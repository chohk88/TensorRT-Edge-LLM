/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime/torchTrtEngineRunner.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include <cmath>
#include <fstream>
#include <sstream>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

namespace
{

//! Binding names for torch_tensorrt built engines with attention plugin
namespace binding_names
{
constexpr char const* kInputIds = "input_ids";
constexpr char const* kContextLengths = "ctx_len";
constexpr char const* kLogits = "output0";  // torch_tensorrt names logits as output0

// KV cache inputs follow pattern: kv_caches_0, kv_caches_1, etc.
std::string formatKVCacheName(int32_t layerIdx)
{
    return "kv_caches_" + std::to_string(layerIdx);
}

// Delta KV outputs follow pattern: output1, output2, etc. (output0 is logits)
std::string formatDeltaKVName(int32_t layerIdx)
{
    return "output" + std::to_string(layerIdx + 1);
}
} // namespace binding_names

std::string formatEngineConfig(TorchTrtEngineConfig const& config)
{
    std::stringstream ss;
    ss << "TorchTrtEngineConfig: "
       << "modelType=" << config.modelType << ", numDecoderLayers=" << config.numDecoderLayers
       << ", numAttentionHeads=" << config.numAttentionHeads << ", numKeyValueHeads=" << config.numKeyValueHeads
       << ", headDim=" << config.headDim << ", hiddenSize=" << config.hiddenSize << ", vocabSize=" << config.vocabSize
       << ", maxBatchSize=" << config.maxSupportedBatchSize << ", maxSeqLen=" << config.maxSequenceLength;
    return ss.str();
}

} // namespace

TorchTrtEngineRunner::TorchTrtEngineRunner(
    std::filesystem::path const& enginePath, std::filesystem::path const& configPath, cudaStream_t stream)
{
    LOG_INFO("Loading config file: %s", configPath.string().c_str());

    // Parse configuration from JSON file
    Json configJson;
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.string().c_str());
        throw std::runtime_error("Failed to open config file: " + configPath.string());
    }
    try
    {
        configJson = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file with error: %s", e.what());
        throw std::runtime_error("Failed to parse config file: " + configPath.string());
    }

    if (!this->initializeConfigFromJson(configJson))
    {
        LOG_ERROR("Failed to initialize TorchTrtEngineRunner from config file: %s", configPath.string().c_str());
        throw std::runtime_error("Failed to initialize TorchTrtEngineRunner from config file: " + configPath.string());
    }

    // Load the engine
    LOG_INFO("Loading engine file: %s", enginePath.string().c_str());
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    if (mmapReader->getData() == nullptr)
    {
        LOG_ERROR("TorchTrtEngineRunner(): Failed to read engine from file: %s", enginePath.string().c_str());
        throw std::runtime_error("Failed to read engine from file: " + enginePath.string());
    }
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));

    if (!mEngine)
    {
        LOG_ERROR("TorchTrtEngineRunner(): Failed to deserialize engine");
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    // Allocate execution context memory
    int64_t const execContextMemoryInBytes = mEngine->getDeviceMemorySizeV2();
    mExecContextMemory = rt::Tensor({execContextMemoryInBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "TorchTrtEngineRunner::mExecContextMemory");

    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    mContext->setDeviceMemoryV2(mExecContextMemory.rawPointer(), execContextMemoryInBytes);

    LOG_INFO("Allocated device memory of %zu bytes for execution context.", execContextMemoryInBytes);

    // Initialize RoPE cache
    if (!this->initializeRopeCache(configJson, stream))
    {
        LOG_ERROR("Failed to initialize RoPE cache");
        throw std::runtime_error("Failed to initialize RoPE cache");
    }

    // Initialize per-layer KV caches
    if (!this->initializeKVCaches(stream))
    {
        LOG_ERROR("Failed to initialize KV caches");
        throw std::runtime_error("Failed to initialize KV caches");
    }

    // Initialize KV cache start index tensor
    mKVCacheStartIdx = rt::Tensor({mConfig.maxSupportedBatchSize}, rt::DeviceType::kGPU, DataType::kINT32,
        "TorchTrtEngineRunner::mKVCacheStartIdx");
    CUDA_CHECK(cudaMemsetAsync(mKVCacheStartIdx.rawPointer(), 0, mKVCacheStartIdx.getMemoryCapacity(), stream));

    // Initialize KV cache lengths tracking
    mKVCacheLengths.resize(mConfig.maxSupportedBatchSize, 0);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_INFO("TorchTrtEngineRunner initialized successfully: %s", formatEngineConfig(mConfig).c_str());
}

TorchTrtEngineRunner::~TorchTrtEngineRunner()
{
    for (auto& [batchSize, graphPair] : mDecodingCudaGraphs)
    {
        CUDA_CHECK(cudaGraphDestroy(graphPair.first));
        CUDA_CHECK(cudaGraphExecDestroy(graphPair.second));
    }
}

bool TorchTrtEngineRunner::initializeConfigFromJson(Json const& configJson)
{
    try
    {
        // Required fields
        std::vector<std::string> const requiredFields
            = {"num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_size", "vocab_size"};

        for (auto const& field : requiredFields)
        {
            if (!configJson.contains(field))
            {
                LOG_ERROR("Missing required field '%s' in config", field.c_str());
                return false;
            }
        }

        mConfig.numDecoderLayers = configJson["num_hidden_layers"].get<int32_t>();
        mConfig.numAttentionHeads = configJson["num_attention_heads"].get<int32_t>();
        mConfig.numKeyValueHeads = configJson["num_key_value_heads"].get<int32_t>();
        mConfig.hiddenSize = configJson["hidden_size"].get<int32_t>();
        mConfig.vocabSize = configJson["vocab_size"].get<int32_t>();

        // Head dimension - may be explicitly specified or computed
        if (configJson.contains("head_dim"))
        {
            mConfig.headDim = configJson["head_dim"].get<int32_t>();
        }
        else
        {
            mConfig.headDim = mConfig.hiddenSize / mConfig.numAttentionHeads;
        }

        // Builder config section for torch_tensorrt specific settings
        if (configJson.contains("torch_trt_config"))
        {
            auto const& trtConfig = configJson["torch_trt_config"];
            mConfig.maxSupportedBatchSize = trtConfig.value("max_batch_size", 1);
            mConfig.maxSequenceLength = trtConfig.value("max_sequence_length", 2048);

            std::string dtypeStr = trtConfig.value("dtype", "fp16");
            if (dtypeStr == "fp16")
            {
                mConfig.dtype = DataType::kHALF;
            }
            else if (dtypeStr == "bf16")
            {
                mConfig.dtype = DataType::kBF16;
            }
            else
            {
                mConfig.dtype = DataType::kFLOAT;
            }
        }
        else
        {
            // Defaults
            mConfig.maxSupportedBatchSize = 1;
            mConfig.maxSequenceLength = 2048;
            mConfig.dtype = DataType::kHALF;
        }

        // Model type detection
        mConfig.modelType = configJson.value("model_type", "unknown");

        // Validation
        if (mConfig.numDecoderLayers <= 0 || mConfig.numAttentionHeads <= 0 || mConfig.numKeyValueHeads <= 0
            || mConfig.headDim <= 0 || mConfig.hiddenSize <= 0 || mConfig.vocabSize <= 0)
        {
            LOG_ERROR("Invalid configuration values (must be positive)");
            return false;
        }

        LOG_INFO("Loaded config: %s", formatEngineConfig(mConfig).c_str());
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Error parsing config: %s", e.what());
        return false;
    }
}

bool TorchTrtEngineRunner::initializeRopeCache(Json const& configJson, cudaStream_t stream)
{
    // RoPE cache shape: [1, max_seq_len, head_dim]
    // This matches the format expected by the attention plugin
    int32_t const ropeLen = mConfig.headDim;

    mRopeCache = rt::Tensor(
        {1, mConfig.maxSequenceLength, ropeLen}, rt::DeviceType::kGPU, DataType::kFLOAT, "TorchTrtEngineRunner::mRopeCache");

    // Get RoPE parameters from config
    float ropeTheta = configJson.value("rope_theta", 10000.0f);
    float attentionScaling = 1.0f;
    if (configJson.contains("rope_scaling") && configJson["rope_scaling"].contains("factor"))
    {
        attentionScaling = 1.0f / std::sqrt(configJson["rope_scaling"]["factor"].get<float>());
    }

    // Compute RoPE cache on CPU then copy to GPU
    int32_t const halfDim = ropeLen / 2;
    std::vector<float> ropeCacheHost(mConfig.maxSequenceLength * ropeLen);

    for (int32_t pos = 0; pos < mConfig.maxSequenceLength; ++pos)
    {
        for (int32_t i = 0; i < halfDim; ++i)
        {
            float invFreq = 1.0f / std::pow(ropeTheta, static_cast<float>(2 * i) / static_cast<float>(ropeLen));
            float angle = static_cast<float>(pos) * invFreq;
            // cos in first half, sin in second half
            ropeCacheHost[pos * ropeLen + i] = std::cos(angle) * attentionScaling;
            ropeCacheHost[pos * ropeLen + halfDim + i] = std::sin(angle) * attentionScaling;
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(mRopeCache.rawPointer(), ropeCacheHost.data(),
        mConfig.maxSequenceLength * ropeLen * sizeof(float), cudaMemcpyHostToDevice, stream));

    LOG_DEBUG("Initialized RoPE cache with shape [1, %d, %d]", mConfig.maxSequenceLength, ropeLen);
    return true;
}

bool TorchTrtEngineRunner::initializeKVCaches(cudaStream_t stream)
{
    // Each layer has its own KV cache with shape [batch, 2, num_kv_heads, max_seq_len, head_dim]
    // This matches the format expected by the attention plugin
    mKVCaches.clear();
    mKVCaches.reserve(mConfig.numDecoderLayers);

    // Delta KV caches for engine outputs - shape matches max input length
    mDeltaKVCaches.clear();
    mDeltaKVCaches.reserve(mConfig.numDecoderLayers);

    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        std::string tensorName = "TorchTrtEngineRunner::mKVCache_" + std::to_string(i);
        mKVCaches.emplace_back(rt::Tensor(
            {mConfig.maxSupportedBatchSize, 2, mConfig.numKeyValueHeads, mConfig.maxSequenceLength, mConfig.headDim},
            rt::DeviceType::kGPU, mConfig.dtype, tensorName));
        CUDA_CHECK(cudaMemsetAsync(mKVCaches.back().rawPointer(), 0, mKVCaches.back().getMemoryCapacity(), stream));

        // Delta KV buffer for engine outputs
        std::string deltaTensorName = "TorchTrtEngineRunner::mDeltaKVCache_" + std::to_string(i);
        mDeltaKVCaches.emplace_back(rt::Tensor(
            {mConfig.maxSupportedBatchSize, 2, mConfig.numKeyValueHeads, mConfig.maxSequenceLength, mConfig.headDim},
            rt::DeviceType::kGPU, mConfig.dtype, deltaTensorName));
    }

    LOG_DEBUG("Initialized %d KV caches with shape [%d, 2, %d, %d, %d]", mConfig.numDecoderLayers,
        mConfig.maxSupportedBatchSize, mConfig.numKeyValueHeads, mConfig.maxSequenceLength, mConfig.headDim);
    return true;
}

TorchTrtEngineConfig TorchTrtEngineRunner::getEngineConfig() const
{
    return mConfig;
}

rt::Tensor& TorchTrtEngineRunner::getRopeCacheTensor()
{
    return mRopeCache;
}

std::vector<rt::Tensor>& TorchTrtEngineRunner::getKVCaches()
{
    return mKVCaches;
}

std::vector<int32_t> const& TorchTrtEngineRunner::getKVCacheLengths() const
{
    return mKVCacheLengths;
}

void TorchTrtEngineRunner::resetKVCaches(int32_t batchSize, cudaStream_t stream)
{
    // Reset KV cache lengths
    std::fill(mKVCacheLengths.begin(), mKVCacheLengths.end(), 0);

    // Reset KV cache start indices to 0
    CUDA_CHECK(cudaMemsetAsync(mKVCacheStartIdx.rawPointer(), 0, batchSize * sizeof(int32_t), stream));

    // Optionally zero out KV caches (may not be necessary if we track lengths correctly)
    for (auto& kvCache : mKVCaches)
    {
        CUDA_CHECK(cudaMemsetAsync(kvCache.rawPointer(), 0, kvCache.getMemoryCapacity(), stream));
    }

    LOG_DEBUG("Reset KV caches for batch size %d", batchSize);
}

void TorchTrtEngineRunner::commitKVCacheLength(int32_t numTokens)
{
    for (auto& len : mKVCacheLengths)
    {
        len += numTokens;
    }
}

bool TorchTrtEngineRunner::bindEngineIO(rt::Tensor const& inputIds, rt::Tensor const& positionIds,
    rt::Tensor const& contextLengths, rt::Tensor& outputLogits, bool isDecoding)
{
    bool status = true;
    int32_t const batchSize = inputIds.getShape()[0];
    int32_t const seqLen = inputIds.getShape()[1];

    // Bind input_ids (INT64 for torch_tensorrt engine)
    status &= mContext->setTensorAddress(binding_names::kInputIds, const_cast<void*>(inputIds.rawPointer()));
    status &= mContext->setInputShape(binding_names::kInputIds, inputIds.getShape().getTRTDims());

    // Note: position_ids is not used in torch_tensorrt engine with attention plugin
    // The plugin handles position encoding internally using ctx_len

    // Bind context_lengths (ctx_len)
    status &= mContext->setTensorAddress(binding_names::kContextLengths, const_cast<void*>(contextLengths.rawPointer()));
    status &= mContext->setInputShape(binding_names::kContextLengths, contextLengths.getShape().getTRTDims());

    // Bind KV caches for each layer
    Dims const kvCacheDims = {5, {batchSize, 2, mConfig.numKeyValueHeads, mConfig.maxSequenceLength, mConfig.headDim}};
    // Delta KV output has dynamic seq_len dimension - matches input seqLen
    Dims const deltaKVDims = {5, {batchSize, 2, mConfig.numKeyValueHeads, seqLen, mConfig.headDim}};
    
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        std::string kvCacheName = binding_names::formatKVCacheName(i);
        status &= mContext->setTensorAddress(kvCacheName.c_str(), mKVCaches[i].rawPointer());
        status &= mContext->setInputShape(kvCacheName.c_str(), kvCacheDims);

        // Bind delta KV output - torch_tensorrt engine outputs new KV for current tokens
        std::string deltaKVName = binding_names::formatDeltaKVName(i);
        status &= mContext->setTensorAddress(deltaKVName.c_str(), mDeltaKVCaches[i].rawPointer());
        // Note: Output shapes are inferred by TensorRT, we just need sufficient buffer
    }

    // Bind output logits - shape is [batch, seq_len, vocab_size]
    // Output shape is inferred from inputs
    status &= mContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());

    if (!status)
    {
        LOG_ERROR("Failed to bind engine I/O tensors");
    }

    return status;
}

bool TorchTrtEngineRunner::executePrefillStep(rt::Tensor const& inputIds, rt::Tensor const& positionIds,
    rt::Tensor const& contextLengths, rt::Tensor& outputLogits, cudaStream_t stream)
{
    int32_t const batchSize = inputIds.getShape()[0];
    int32_t const seqLen = inputIds.getShape()[1];

    // Validate inputs
    if (batchSize > mConfig.maxSupportedBatchSize)
    {
        LOG_ERROR("Batch size %d exceeds max supported %d", batchSize, mConfig.maxSupportedBatchSize);
        return false;
    }
    if (seqLen > mConfig.maxSequenceLength)
    {
        LOG_ERROR("Sequence length %d exceeds max supported %d", seqLen, mConfig.maxSequenceLength);
        return false;
    }

    // Bind I/O
    if (!bindEngineIO(inputIds, positionIds, contextLengths, outputLogits, false))
    {
        LOG_ERROR("Failed to bind engine I/O for prefill step");
        return false;
    }

    // Execute
    bool executeStatus = mContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("TensorRT enqueueV3 failed for prefill step");
        return false;
    }

    // Note: The attention plugin updates KV caches in-place during execution.
    // No need to copy delta KV to KV cache.

    // Commit KV cache lengths
    // For prefill, each sequence contributes seqLen tokens
    for (int32_t i = 0; i < batchSize; ++i)
    {
        mKVCacheLengths[i] = seqLen;
    }

    LOG_DEBUG("Prefill step completed: batch_size=%d, seq_len=%d", batchSize, seqLen);
    return true;
}

bool TorchTrtEngineRunner::executeDecodingStep(rt::Tensor const& inputIds, rt::Tensor const& positionIds,
    rt::Tensor const& contextLengths, rt::Tensor& outputLogits, cudaStream_t stream)
{
    int32_t const batchSize = inputIds.getShape()[0];

    // Validate inputs
    if (batchSize > mConfig.maxSupportedBatchSize)
    {
        LOG_ERROR("Batch size %d exceeds max supported %d", batchSize, mConfig.maxSupportedBatchSize);
        return false;
    }

    // Check if we have a captured CUDA graph for this batch size
    if (mDecodingCudaGraphs.find(batchSize) != mDecodingCudaGraphs.end())
    {
        LOG_DEBUG("Using captured CUDA graph for decoding step");
        cudaGraphExec_t graphExec = mDecodingCudaGraphs[batchSize].second;
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    else
    {
        // Bind I/O
        if (!bindEngineIO(inputIds, positionIds, contextLengths, outputLogits, true))
        {
            LOG_ERROR("Failed to bind engine I/O for decoding step");
            return false;
        }

        // Execute
        bool executeStatus = mContext->enqueueV3(stream);
        if (!executeStatus)
        {
            LOG_ERROR("TensorRT enqueueV3 failed for decoding step");
            return false;
        }

        // Note: The attention plugin updates KV caches in-place during execution.
        // No need to copy delta KV to KV cache.
    }

    // Commit KV cache lengths (1 new token per sequence)
    commitKVCacheLength(1);

    LOG_DEBUG("Decoding step completed: batch_size=%d, new_kv_len=%d", batchSize, mKVCacheLengths[0]);
    return true;
}

bool TorchTrtEngineRunner::captureDecodingCudaGraph(int32_t batchSize, cudaStream_t stream)
{
    if (mDecodingCudaGraphs.find(batchSize) != mDecodingCudaGraphs.end())
    {
        LOG_INFO("CUDA graph already captured for batch size %d", batchSize);
        return true;
    }

    // Create dummy tensors for capture
    rt::Tensor dummyInputIds({batchSize, 1}, rt::DeviceType::kGPU, DataType::kINT32, "dummyInputIds");
    rt::Tensor dummyPositionIds({batchSize, 1}, rt::DeviceType::kGPU, DataType::kINT64, "dummyPositionIds");
    rt::Tensor dummyContextLengths({batchSize}, rt::DeviceType::kGPU, DataType::kINT32, "dummyContextLengths");
    rt::Tensor dummyOutputLogits({batchSize, mConfig.vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT, "dummyOutputLogits");

    // Initialize with dummy data
    CUDA_CHECK(cudaMemsetAsync(dummyInputIds.rawPointer(), 0, dummyInputIds.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(dummyPositionIds.rawPointer(), 0, dummyPositionIds.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(dummyContextLengths.rawPointer(), 0, dummyContextLengths.getMemoryCapacity(), stream));

    // Bind I/O
    if (!bindEngineIO(dummyInputIds, dummyPositionIds, dummyContextLengths, dummyOutputLogits, true))
    {
        LOG_ERROR("Failed to bind engine I/O for CUDA graph capture");
        return false;
    }

    // Warm-up run before capture
    bool executeStatus = mContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("Warm-up enqueueV3 failed for CUDA graph capture");
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Capture CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    executeStatus = mContext->enqueueV3(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    if (!executeStatus)
    {
        LOG_WARNING("enqueueV3 failed during CUDA graph capture");
        return false;
    }

    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    mDecodingCudaGraphs[batchSize] = std::make_pair(graph, graphExec);

    LOG_INFO("CUDA graph captured successfully for batch size %d", batchSize);
    return true;
}

void TorchTrtEngineRunner::updateKVCaches(int32_t batchSize, int32_t numNewTokens, int32_t startPos, cudaStream_t stream)
{
    // The attention plugin outputs delta KV for the new tokens.
    // Delta KV shape: [batch, 2, num_kv_heads, num_new_tokens, head_dim]
    // We need to copy this to KV cache at the appropriate position.
    //
    // KV cache shape: [batch, 2, num_kv_heads, max_seq_len, head_dim]
    // Copy delta_kv[:, :, :, :num_new_tokens, :] to kv_cache[:, :, :, startPos:startPos+num_new_tokens, :]

    // Calculate copy parameters
    size_t const elemSize = (mConfig.dtype == DataType::kHALF) ? 2 : 4;
    size_t const kvPerTokenSize = 2 * mConfig.numKeyValueHeads * mConfig.headDim * elemSize;
    size_t const deltaKVCopySize = static_cast<size_t>(batchSize) * numNewTokens * kvPerTokenSize;

    // For each layer, copy delta KV to appropriate position in KV cache
    for (int32_t layer = 0; layer < mConfig.numDecoderLayers; ++layer)
    {
        // Source: delta KV cache (contiguous for new tokens)
        void const* srcPtr = mDeltaKVCaches[layer].rawPointer();

        // Destination: KV cache at startPos
        // KV cache layout: [batch, 2, num_kv_heads, max_seq_len, head_dim]
        // Need to copy to position startPos in the seq_len dimension
        //
        // For simplicity, we assume batch=1 for now and use strided copy
        // TODO: Support batch > 1 with proper strided memory copy

        if (batchSize == 1)
        {
            // For batch=1, we can use a simple contiguous copy per (2, num_kv_heads) slice
            size_t const seqStride = mConfig.headDim * elemSize;
            size_t const srcOffset = 0;  // Delta starts at position 0
            size_t const dstOffset = static_cast<size_t>(startPos) * seqStride;

            // Copy each (2 * num_kv_heads) plane
            size_t const planeSize = numNewTokens * mConfig.headDim * elemSize;
            size_t const numPlanes = 2 * mConfig.numKeyValueHeads;

            for (size_t plane = 0; plane < numPlanes; ++plane)
            {
                size_t const planeSrcOffset = plane * numNewTokens * mConfig.headDim * elemSize;
                size_t const planeDstOffset = plane * mConfig.maxSequenceLength * mConfig.headDim * elemSize + dstOffset;

                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<char*>(mKVCaches[layer].rawPointer()) + planeDstOffset,
                    static_cast<char const*>(srcPtr) + planeSrcOffset,
                    planeSize,
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
        }
        else
        {
            // For batch > 1, a more complex copy is needed
            LOG_WARNING("Batch size > 1 KV cache update not fully optimized");
            // Fallback: copy entire delta KV buffer (this may overwrite wrong positions)
            CUDA_CHECK(cudaMemcpyAsync(
                mKVCaches[layer].rawPointer(),
                srcPtr,
                deltaKVCopySize,
                cudaMemcpyDeviceToDevice,
                stream));
        }
    }

    LOG_DEBUG("Updated KV caches: layer_count=%d, batch=%d, new_tokens=%d, start_pos=%d",
        mConfig.numDecoderLayers, batchSize, numNewTokens, startPos);
}

} // namespace rt
} // namespace trt_edgellm

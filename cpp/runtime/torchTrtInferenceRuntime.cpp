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

#include "runtime/torchTrtInferenceRuntime.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <filesystem>
#include <limits>

namespace trt_edgellm
{
namespace rt
{

TorchTrtInferenceRuntime::TorchTrtInferenceRuntime(std::string const& engineDir, cudaStream_t stream)
{
    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / "model.engine";
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";

    // Check if engine file exists
    if (!std::filesystem::exists(enginePath))
    {
        LOG_ERROR("Engine file not found: %s", enginePath.string().c_str());
        throw std::runtime_error("Engine file not found: " + enginePath.string());
    }

    // Initialize engine runner
    try
    {
        mEngineRunner = std::make_unique<TorchTrtEngineRunner>(enginePath, configPath, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize TorchTrtEngineRunner: %s", e.what());
        throw std::runtime_error("Failed to initialize TorchTrtEngineRunner: " + std::string(e.what()));
    }
    LOG_INFO("TorchTrtEngineRunner successfully initialized.");

    auto const& config = mEngineRunner->getEngineConfig();

    // Initialize tokenizer
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Loading tokenizer from: %s", engineDir.c_str());
    if (!mTokenizer->loadFromHF(engineDir))
    {
        LOG_ERROR("Failed to load tokenizer from: %s", engineDir.c_str());
        throw std::runtime_error("Failed to load tokenizer from: " + engineDir);
    }
    LOG_INFO("Tokenizer loaded successfully.");

    // Pre-allocate tensors for max batch size
    allocateTensors(config.maxSupportedBatchSize, config.maxSequenceLength, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_INFO("TorchTrtInferenceRuntime initialized successfully.");
}

void TorchTrtInferenceRuntime::allocateTensors(int32_t batchSize, int32_t maxSeqLen, cudaStream_t stream)
{
    auto const& config = mEngineRunner->getEngineConfig();

    // Input tensors - torch_tensorrt engine expects INT64 for input_ids
    mInputIds = rt::Tensor({batchSize, maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
        "TorchTrtInferenceRuntime::mInputIds");
    mPositionIds = rt::Tensor({batchSize, maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
        "TorchTrtInferenceRuntime::mPositionIds");
    mContextLengths = rt::Tensor(
        {batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "TorchTrtInferenceRuntime::mContextLengths");

    // Output tensors - shape is [batch, max_seq_len, vocab_size], dtype matches engine (HALF)
    mOutputLogits = rt::Tensor({batchSize, maxSeqLen, config.vocabSize}, rt::DeviceType::kGPU, config.dtype,
        "TorchTrtInferenceRuntime::mOutputLogits");

    // Sampling tensors
    mSelectedTokens = rt::Tensor(
        {batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "TorchTrtInferenceRuntime::mSelectedTokens");
    mHostSelectedTokens = rt::Tensor(
        {batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "TorchTrtInferenceRuntime::mHostSelectedTokens");

    // Sampling workspace
    SamplingParams defaultParams(batchSize, config.vocabSize, 1.0f, 0, 0.9f);
    int64_t workspaceSize = static_cast<int64_t>(getTopKtopPSamplingWorkspaceSize(batchSize, config.vocabSize, defaultParams));
    mSamplingWorkspace = rt::Tensor(
        {workspaceSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "TorchTrtInferenceRuntime::mSamplingWorkspace");

    // Initialize
    CUDA_CHECK(cudaMemsetAsync(mInputIds.rawPointer(), 0, mInputIds.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(mPositionIds.rawPointer(), 0, mPositionIds.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(mContextLengths.rawPointer(), 0, mContextLengths.getMemoryCapacity(), stream));

    LOG_DEBUG("Allocated tensors for batch_size=%d, max_seq_len=%d", batchSize, maxSeqLen);
}

bool TorchTrtInferenceRuntime::handleRequest(
    TorchTrtGenerationRequest const& request, TorchTrtGenerationResponse& response, cudaStream_t stream)
{
    int32_t const batchSize = static_cast<int32_t>(request.prompts.size());
    auto const& config = mEngineRunner->getEngineConfig();

    if (batchSize == 0)
    {
        LOG_ERROR("Empty request");
        return false;
    }
    if (batchSize > config.maxSupportedBatchSize)
    {
        LOG_ERROR("Batch size %d exceeds max %d", batchSize, config.maxSupportedBatchSize);
        return false;
    }

    // Tokenize prompts
    std::vector<std::vector<int32_t>> batchedInputIds;
    batchedInputIds.reserve(batchSize);

    for (auto const& prompt : request.prompts)
    {
        std::string formattedPrompt = prompt;
        if (request.applyChatTemplate)
        {
            // Simple chat template application - can be enhanced
            formattedPrompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        }
        batchedInputIds.emplace_back(mTokenizer->encode(formattedPrompt, true));
    }

    // Generate
    std::vector<std::vector<int32_t>> outputIds
        = generate(batchedInputIds, request.maxGenerateLength, mTokenizer->getEosId(), stream);

    // Prepare response
    response.outputIds = outputIds;
    response.outputTexts.clear();
    response.numGeneratedTokens.clear();

    for (size_t i = 0; i < outputIds.size(); ++i)
    {
        response.outputTexts.emplace_back(mTokenizer->decode(outputIds[i], true));
        response.numGeneratedTokens.push_back(static_cast<int32_t>(outputIds[i].size()));
    }

    return true;
}

std::vector<std::vector<int32_t>> TorchTrtInferenceRuntime::generate(
    std::vector<std::vector<int32_t>> const& inputIds, int32_t maxNewTokens, int32_t eosTokenId, cudaStream_t stream)
{
    int32_t const batchSize = static_cast<int32_t>(inputIds.size());
    auto const& config = mEngineRunner->getEngineConfig();

    // Find max input length
    int32_t maxInputLen = 0;
    for (auto const& ids : inputIds)
    {
        maxInputLen = std::max(maxInputLen, static_cast<int32_t>(ids.size()));
    }

    if (maxInputLen + maxNewTokens > config.maxSequenceLength)
    {
        LOG_WARNING("Total length %d exceeds max %d, truncating", maxInputLen + maxNewTokens, config.maxSequenceLength);
        maxNewTokens = config.maxSequenceLength - maxInputLen;
    }

    // Reset KV caches
    mEngineRunner->resetKVCaches(batchSize, stream);

    // Prepare input tensors
    mInputIds.reshape({batchSize, maxInputLen});
    mPositionIds.reshape({batchSize, maxInputLen});
    mContextLengths.reshape({batchSize});
    mOutputLogits.reshape({batchSize, maxInputLen, config.vocabSize});

    // Pad and copy input IDs (INT64 for torch_tensorrt engine)
    std::vector<int64_t> paddedInputIds(batchSize * maxInputLen, static_cast<int64_t>(mTokenizer->getPadId()));
    std::vector<int64_t> positionIdsHost(batchSize * maxInputLen, 0);
    std::vector<int32_t> contextLengthsHost(batchSize);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t seqLen = static_cast<int32_t>(inputIds[b].size());
        contextLengthsHost[b] = seqLen;

        for (int32_t s = 0; s < seqLen; ++s)
        {
            paddedInputIds[b * maxInputLen + s] = static_cast<int64_t>(inputIds[b][s]);
            positionIdsHost[b * maxInputLen + s] = static_cast<int64_t>(s);
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), paddedInputIds.data(), batchSize * maxInputLen * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mPositionIds.rawPointer(), positionIdsHost.data(),
        batchSize * maxInputLen * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mContextLengths.rawPointer(), contextLengthsHost.data(), batchSize * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));

    // ===== Prefill Step =====
    {
        TIME_STAGE(metrics::StageNames::kLLM_PREFILL, stream);

        if (!mEngineRunner->executePrefillStep(mInputIds, mPositionIds, mContextLengths, mOutputLogits, stream))
        {
            LOG_ERROR("Prefill step failed");
            return {};
        }

        // Sample first token (use maxInputLen as actual sequence length from prefill)
        std::vector<int32_t> firstTokens = sampleGreedy(mOutputLogits, maxInputLen, 1.0f, stream);

        // Initialize output tracking
        mPrefillMetrics.recordRun(0, maxInputLen * batchSize);
    }

    // ===== Generation Loop =====
    std::vector<std::vector<int32_t>> generatedIds(batchSize);
    std::vector<bool> finished(batchSize, false);
    int32_t numUnfinished = batchSize;

    // Get first token from prefill
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int32_t> currentTokens = sampleGreedy(mOutputLogits, maxInputLen, 1.0f, stream);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        generatedIds[b].push_back(currentTokens[b]);
        if (currentTokens[b] == eosTokenId)
        {
            finished[b] = true;
            numUnfinished--;
        }
    }

    // Reshape for decoding
    mInputIds.reshape({batchSize, 1});
    mPositionIds.reshape({batchSize, 1});

    {
        TIME_STAGE(metrics::StageNames::kLLM_GENERATION, stream);

        for (int32_t step = 1; step < maxNewTokens && numUnfinished > 0; ++step)
        {
            // Prepare input for this step (INT64 for torch_tensorrt engine)
            std::vector<int64_t> stepInputIds(batchSize);
            std::vector<int64_t> stepPositionIds(batchSize);
            std::vector<int32_t> stepContextLengths(batchSize);

            for (int32_t b = 0; b < batchSize; ++b)
            {
                stepInputIds[b] = static_cast<int64_t>(currentTokens[b]);
                // position_ids is not used by the engine (plugin uses ctx_len for position)
                stepPositionIds[b] = static_cast<int64_t>(contextLengthsHost[b] + step);
                // ctx_len = number of tokens processed so far + 1 (current token)
                // For step 1: ctx_len = prefill_len + 1
                stepContextLengths[b] = contextLengthsHost[b] + step;
            }

            CUDA_CHECK(cudaMemcpyAsync(
                mInputIds.rawPointer(), stepInputIds.data(), batchSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mPositionIds.rawPointer(), stepPositionIds.data(), batchSize * sizeof(int64_t),
                cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mContextLengths.rawPointer(), stepContextLengths.data(),
                batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

            // Execute decoding step
            if (!mEngineRunner->executeDecodingStep(mInputIds, mPositionIds, mContextLengths, mOutputLogits, stream))
            {
                LOG_ERROR("Decoding step %d failed", step);
                break;
            }

            // Sample next token (decoding produces 1 token at a time)
            currentTokens = sampleGreedy(mOutputLogits, 1, 1.0f, stream);

            // Update tracking
            for (int32_t b = 0; b < batchSize; ++b)
            {
                if (!finished[b])
                {
                    generatedIds[b].push_back(currentTokens[b]);
                    if (currentTokens[b] == eosTokenId)
                    {
                        finished[b] = true;
                        numUnfinished--;
                    }
                }
            }
        }
    }

    // Record generation metrics
    int32_t totalGenerated = 0;
    for (auto const& ids : generatedIds)
    {
        totalGenerated += static_cast<int32_t>(ids.size());
    }
    if (totalGenerated > 0)
    {
        mGenerationMetrics.recordRun(totalGenerated);
    }

    return generatedIds;
}

std::vector<int32_t> TorchTrtInferenceRuntime::sampleGreedy(
    rt::Tensor const& logits, int32_t actualSeqLen, float temperature, cudaStream_t stream)
{
    // logits shape is [batch, max_seq_len, vocab_size] but we only use actualSeqLen positions
    // We need to sample from the last token's logits for each sequence
    int32_t const batchSize = logits.getShape()[0];
    int32_t const seqLen = actualSeqLen;  // Use actual sequence length, not tensor shape
    int32_t const vocabSize = logits.getShape()[2];

    auto const& config = mEngineRunner->getEngineConfig();

    // Use simple greedy argmax on last token logits
    mSelectedTokens.reshape({batchSize, 1});
    mHostSelectedTokens.reshape({batchSize});

    // For now, implement simple greedy sampling by finding argmax on CPU
    // TODO: Use GPU kernel for efficient sampling
    std::vector<int32_t> selectedTokens(batchSize);
    
    // Determine element size based on dtype
    size_t const elemSize = (config.dtype == nvinfer1::DataType::kHALF) ? 2 : 4;
    size_t const vocabBytes = vocabSize * elemSize;
    
    // Copy last position logits for each batch to host
    std::vector<uint8_t> hostLogits(batchSize * vocabBytes);
    
    for (int32_t b = 0; b < batchSize; ++b)
    {
        // Get pointer to last token logits for this batch
        // Offset: b * seqLen * vocabSize + (seqLen - 1) * vocabSize
        size_t const offset = (static_cast<size_t>(b) * seqLen + (seqLen - 1)) * vocabBytes;
        CUDA_CHECK(cudaMemcpyAsync(
            hostLogits.data() + b * vocabBytes,
            static_cast<uint8_t const*>(logits.rawPointer()) + offset,
            vocabBytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Find argmax for each batch
    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t maxIdx = 0;
        float maxVal = -std::numeric_limits<float>::infinity();

        if (config.dtype == nvinfer1::DataType::kHALF)
        {
            // FP16 logits
            __half const* logitsPtr = reinterpret_cast<__half const*>(hostLogits.data() + b * vocabBytes);
            for (int32_t v = 0; v < vocabSize; ++v)
            {
                float val = __half2float(logitsPtr[v]);
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = v;
                }
            }
        }
        else
        {
            // FP32 logits
            float const* logitsPtr = reinterpret_cast<float const*>(hostLogits.data() + b * vocabBytes);
            for (int32_t v = 0; v < vocabSize; ++v)
            {
                if (logitsPtr[v] > maxVal)
                {
                    maxVal = logitsPtr[v];
                    maxIdx = v;
                }
            }
        }
        selectedTokens[b] = maxIdx;
    }

    return selectedTokens;
}

bool TorchTrtInferenceRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    auto const& config = mEngineRunner->getEngineConfig();
    bool status = true;

    for (int32_t batchSize = 1; batchSize <= config.maxSupportedBatchSize; ++batchSize)
    {
        status &= mEngineRunner->captureDecodingCudaGraph(batchSize, stream);
    }

    if (status)
    {
        LOG_INFO("Successfully captured CUDA graphs for all batch sizes");
    }
    else
    {
        LOG_WARNING("Failed to capture CUDA graphs for some batch sizes");
    }

    return status;
}

} // namespace rt
} // namespace trt_edgellm

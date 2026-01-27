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

#pragma once

#include "common/tensor.h"
#include "runtime/linearKVCache.h"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
using Json = nlohmann::json;

/*!
 * @brief Configuration structure for TorchTrt engine runner
 *
 * Contains all runtime configuration parameters for engines built with torch_tensorrt.
 * This configuration is loaded from a JSON file generated during engine building.
 */
struct TorchTrtEngineConfig
{
    int32_t numDecoderLayers{};      //!< Number of decoder layers
    int32_t numAttentionHeads{};     //!< Number of query attention heads
    int32_t numKeyValueHeads{};      //!< Number of key-value heads
    int32_t headDim{};               //!< Dimension of each attention head
    int32_t hiddenSize{};            //!< Model's hidden dimension
    int32_t vocabSize{};             //!< Vocabulary size
    int32_t maxSupportedBatchSize{}; //!< Maximum supported batch size
    int32_t maxSequenceLength{};     //!< Maximum sequence length for KV cache
    nvinfer1::DataType dtype{};      //!< Data type (FP16 or BF16)
    std::string modelType{};         //!< Model type (e.g., "qwen2", "qwen3", "llama")
};

/*!
 * @brief Engine runner for TensorRT engines built with torch_tensorrt
 *
 * This class handles loading and executing TensorRT engines that were built using
 * torch_tensorrt with the attention plugin backend. It manages:
 * - Engine deserialization and execution context creation
 * - Per-layer KV cache management aligned with plugin expectations
 * - RoPE cache handling
 * - Prefill and decoding execution
 *
 * Unlike the standard LLMEngineRunner which uses a single monolithic KV cache,
 * this runner handles per-layer KV caches as produced by torch_tensorrt compilation.
 */
class TorchTrtEngineRunner
{
public:
    /*!
     * @brief Construct TorchTrt engine runner
     * @param enginePath Path to serialized TensorRT engine file (.engine)
     * @param configPath Path to model configuration JSON file
     * @param stream CUDA stream for operations
     */
    TorchTrtEngineRunner(
        std::filesystem::path const& enginePath, std::filesystem::path const& configPath, cudaStream_t stream);

    //! @brief Destructor
    ~TorchTrtEngineRunner();

    //! @brief Get engine configuration
    //! @return Engine configuration structure
    TorchTrtEngineConfig getEngineConfig() const;

    //! @brief Get reference to the RoPE cache tensor
    //! @return Reference to RoPE cache tensor
    rt::Tensor& getRopeCacheTensor();

    //! @brief Get KV cache tensors for all layers
    //! @return Reference to vector of KV cache tensors
    std::vector<rt::Tensor>& getKVCaches();

    //! @brief Reset KV caches for new sequences
    //! @param batchSize Active batch size
    //! @param stream CUDA stream for operations
    void resetKVCaches(int32_t batchSize, cudaStream_t stream);

    /*!
     * @brief Execute prefill step
     *
     * Process the input sequence and fill the KV cache for the initial context.
     *
     * @param inputIds Input token IDs [batch_size, seq_len]
     * @param positionIds Position IDs [batch_size, seq_len]
     * @param contextLengths Context lengths for each sequence [batch_size]
     * @param outputLogits Output logits tensor [batch_size, vocab_size]
     * @param stream CUDA stream for execution
     * @return True if prefill was successful, false otherwise
     */
    bool executePrefillStep(rt::Tensor const& inputIds, rt::Tensor const& positionIds,
        rt::Tensor const& contextLengths, rt::Tensor& outputLogits, cudaStream_t stream);

    /*!
     * @brief Execute single-token decoding step
     *
     * Generate the next token given the previous token and KV cache state.
     *
     * @param inputIds Input token IDs [batch_size, 1]
     * @param positionIds Position IDs [batch_size, 1]
     * @param contextLengths Cumulative context lengths [batch_size]
     * @param outputLogits Output logits tensor [batch_size, vocab_size]
     * @param stream CUDA stream for execution
     * @return True if decoding was successful, false otherwise
     */
    bool executeDecodingStep(rt::Tensor const& inputIds, rt::Tensor const& positionIds,
        rt::Tensor const& contextLengths, rt::Tensor& outputLogits, cudaStream_t stream);

    /*!
     * @brief Capture CUDA graph for decoding step
     *
     * Capture the decoding step as a CUDA graph for lower latency execution.
     *
     * @param batchSize Batch size to capture
     * @param stream CUDA stream for capture
     * @return True if capture was successful, false otherwise
     */
    bool captureDecodingCudaGraph(int32_t batchSize, cudaStream_t stream);

    //! @brief Get current KV cache lengths
    //! @return Vector of current sequence lengths
    std::vector<int32_t> const& getKVCacheLengths() const;

    //! @brief Commit new tokens to KV cache length tracking
    //! @param numTokens Number of tokens to commit
    void commitKVCacheLength(int32_t numTokens);

private:
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;          //!< TensorRT runtime
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;        //!< TensorRT engine
    std::unique_ptr<nvinfer1::IExecutionContext> mContext; //!< Execution context
    rt::Tensor mExecContextMemory{};                       //!< Device memory for execution context

    TorchTrtEngineConfig mConfig{}; //!< Engine configuration

    //! Per-layer KV cache tensors [batch, 2, num_kv_heads, max_seq_len, head_dim]
    std::vector<rt::Tensor> mKVCaches{};

    //! Per-layer delta KV cache tensors for engine outputs [batch, 2, num_kv_heads, input_len, head_dim]
    std::vector<rt::Tensor> mDeltaKVCaches{};

    //! Current KV cache lengths for each sequence in batch
    std::vector<int32_t> mKVCacheLengths{};

    //! RoPE cache tensor [1, max_seq_len, head_dim]
    rt::Tensor mRopeCache{};

    //! KV cache start index tensor [batch_size]
    rt::Tensor mKVCacheStartIdx{};

    //! CUDA graphs for decoding (keyed by batch size)
    std::unordered_map<int32_t, std::pair<cudaGraph_t, cudaGraphExec_t>> mDecodingCudaGraphs{};

    /*!
     * @brief Initialize configuration from JSON file
     * @param configJson JSON configuration object
     * @return True on success, false on failure
     */
    bool initializeConfigFromJson(Json const& configJson);

    /*!
     * @brief Initialize RoPE cache
     * @param configJson JSON configuration object
     * @param stream CUDA stream for operations
     * @return True on success, false on failure
     */
    bool initializeRopeCache(Json const& configJson, cudaStream_t stream);

    /*!
     * @brief Initialize per-layer KV caches
     * @param stream CUDA stream for operations
     * @return True on success, false on failure
     */
    bool initializeKVCaches(cudaStream_t stream);

    /*!
     * @brief Bind inputs and outputs for engine execution
     * @param inputIds Input token IDs tensor
     * @param positionIds Position IDs tensor
     * @param contextLengths Context lengths tensor
     * @param outputLogits Output logits tensor
     * @param isDecoding Whether this is a decoding step (vs prefill)
     * @return True on success, false on failure
     */
    bool bindEngineIO(rt::Tensor const& inputIds, rt::Tensor const& positionIds, rt::Tensor const& contextLengths,
        rt::Tensor& outputLogits, bool isDecoding);

    /*!
     * @brief Update KV caches with delta outputs from attention plugin
     * @param batchSize Batch size
     * @param numNewTokens Number of new tokens added in this step
     * @param startPos Starting position in KV cache to write new KV values
     * @param stream CUDA stream for operations
     */
    void updateKVCaches(int32_t batchSize, int32_t numNewTokens, int32_t startPos, cudaStream_t stream);
};

} // namespace rt
} // namespace trt_edgellm

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

#include "profiling/metrics.h"
#include "runtime/torchTrtEngineRunner.h"
#include "tokenizer/tokenizer.h"
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief Generation request structure for torch_tensorrt engines
 *
 * Simplified request structure for engines built with torch_tensorrt.
 * Supports text prompts with optional sampling parameters.
 */
struct TorchTrtGenerationRequest
{
    std::vector<std::string> prompts{};       //!< Input prompts (one per batch element)
    int32_t maxGenerateLength{128};           //!< Maximum number of tokens to generate
    float temperature{1.0f};                  //!< Sampling temperature
    int32_t topK{50};                         //!< Top-K sampling parameter
    float topP{0.9f};                         //!< Top-P (nucleus) sampling parameter
    bool applyChatTemplate{true};             //!< Whether to apply chat template
    std::vector<int32_t> inputIds{};          //!< Pre-tokenized input IDs (optional, overrides prompts)
};

/*!
 * @brief Generation response structure
 */
struct TorchTrtGenerationResponse
{
    std::vector<std::vector<int32_t>> outputIds{}; //!< Generated token IDs for each sequence
    std::vector<std::string> outputTexts{};        //!< Decoded output texts
    std::vector<int32_t> numGeneratedTokens{};     //!< Number of tokens generated per sequence
};

/*!
 * @brief High-level inference runtime for torch_tensorrt built engines
 *
 * This runtime provides a complete inference pipeline for LLM models that were
 * built using torch_tensorrt with the attention plugin. It handles:
 *
 * - Tokenization and detokenization
 * - Prefill and autoregressive decoding
 * - Sampling (greedy, top-k, top-p)
 * - KV cache management
 * - Performance profiling
 *
 * Key differences from standard LLMInferenceRuntime:
 * - Designed for engines built with torch_tensorrt (vs ONNX-based engines)
 * - Supports models: Qwen2.5, Qwen3, Llama3.x
 * - Uses attention plugin for efficient attention computation
 */
class TorchTrtInferenceRuntime
{
public:
    /*!
     * @brief Construct inference runtime
     * @param engineDir Directory containing engine and config files
     * @param stream CUDA stream for initialization
     */
    TorchTrtInferenceRuntime(std::string const& engineDir, cudaStream_t stream);

    //! @brief Destructor
    ~TorchTrtInferenceRuntime() = default;

    /*!
     * @brief Handle a generation request
     * @param request Generation request with prompts and parameters
     * @param response Output response with generated text
     * @param stream CUDA stream for execution
     * @return True if request was handled successfully
     */
    bool handleRequest(TorchTrtGenerationRequest const& request, TorchTrtGenerationResponse& response, cudaStream_t stream);

    /*!
     * @brief Generate tokens from pre-tokenized input IDs
     *
     * Lower-level API for when you already have tokenized input.
     *
     * @param inputIds Input token IDs [batch_size, seq_len]
     * @param maxNewTokens Maximum number of new tokens to generate
     * @param eosTokenId End-of-sequence token ID for early stopping
     * @param stream CUDA stream for execution
     * @return Vector of generated token IDs for each batch element
     */
    std::vector<std::vector<int32_t>> generate(
        std::vector<std::vector<int32_t>> const& inputIds, int32_t maxNewTokens, int32_t eosTokenId, cudaStream_t stream);

    /*!
     * @brief Capture CUDA graph for decoding
     * @param stream CUDA stream for capture
     * @return True if capture was successful
     */
    bool captureDecodingCUDAGraph(cudaStream_t stream);

    //! @brief Get prefill metrics
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const
    {
        return mPrefillMetrics;
    }

    //! @brief Get generation metrics
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const
    {
        return mGenerationMetrics;
    }

    //! @brief Get engine configuration
    TorchTrtEngineConfig getEngineConfig() const
    {
        return mEngineRunner->getEngineConfig();
    }

private:
    std::unique_ptr<TorchTrtEngineRunner> mEngineRunner{nullptr}; //!< Engine runner
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer{nullptr};    //!< Tokenizer

    // Pre-allocated tensors for execution
    rt::Tensor mInputIds{};        //!< Input IDs tensor
    rt::Tensor mPositionIds{};     //!< Position IDs tensor
    rt::Tensor mContextLengths{};  //!< Context lengths tensor
    rt::Tensor mOutputLogits{};    //!< Output logits tensor
    rt::Tensor mSamplingWorkspace{}; //!< Workspace for sampling operations
    rt::Tensor mSelectedTokens{};  //!< Selected token IDs tensor
    rt::Tensor mHostSelectedTokens{}; //!< Host tensor for selected tokens

    metrics::LLMPrefillMetrics mPrefillMetrics{};       //!< Prefill stage metrics
    metrics::LLMGenerationMetrics mGenerationMetrics{}; //!< Generation stage metrics

    /*!
     * @brief Perform greedy sampling from logits
     * @param logits Output logits tensor [batch_size, vocab_size]
     * @param temperature Sampling temperature
     * @param stream CUDA stream
     * @return Vector of selected token IDs
     */
    std::vector<int32_t> sampleGreedy(rt::Tensor const& logits, int32_t actualSeqLen, float temperature, cudaStream_t stream);

    /*!
     * @brief Allocate tensors for the given batch size
     * @param batchSize Batch size
     * @param maxSeqLen Maximum sequence length
     * @param stream CUDA stream
     */
    void allocateTensors(int32_t batchSize, int32_t maxSeqLen, cudaStream_t stream);
};

} // namespace rt
} // namespace trt_edgellm

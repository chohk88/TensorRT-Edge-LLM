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

/**
 * @file torch_trt_inference.cpp
 * @brief Example application for running LLM inference with torch_tensorrt built engines
 *
 * This example demonstrates how to:
 * 1. Load a TensorRT engine built with torch_tensorrt (run_llm.py with --backend plugin)
 * 2. Run inference using TensorRT-Edge-LLM's C++ runtime
 * 3. Leverage optimized KV cache management and CUDA graphs
 *
 * Supported models:
 * - Qwen2.5, Qwen3 series
 * - Llama3.x series
 *
 * Usage:
 *   ./torch_trt_inference \
 *       --engineDir=/path/to/engine_directory \
 *       --prompt="What is parallel programming?" \
 *       --maxTokens=128
 *
 * The engine directory should contain:
 * - model.engine: TensorRT engine file (built with torch_tensorrt)
 * - config.json: Model configuration (from HuggingFace or custom)
 * - tokenizer.json, tokenizer_config.json: Tokenizer files
 */

#include "common/logger.h"
#include "common/trtUtils.h"
#include "memoryMonitor.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/torchTrtInferenceRuntime.h"
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

// Command line option IDs
enum TorchTrtInferenceOptionId : int
{
    HELP = 900,
    ENGINE_DIR = 901,
    PROMPT = 902,
    INPUT_FILE = 903,
    OUTPUT_FILE = 904,
    MAX_TOKENS = 905,
    TEMPERATURE = 906,
    TOP_K = 907,
    TOP_P = 908,
    BATCH_SIZE = 909,
    DEBUG = 910,
    DUMP_PROFILE = 911,
    WARMUP = 912,
    BENCHMARK = 913,
    ISL = 914,
    OSL = 915,
};

struct TorchTrtInferenceArgs
{
    bool help{false};
    std::string engineDir;
    std::string prompt;
    std::string inputFile;
    std::string outputFile;
    int32_t maxTokens{128};
    float temperature{1.0f};
    int32_t topK{50};
    float topP{0.9f};
    int32_t batchSize{1};
    bool debug{false};
    bool dumpProfile{false};
    int32_t warmup{0};
    bool benchmark{false};
    int32_t isl{128};  // Input sequence length for benchmark
    int32_t osl{128};  // Output sequence length for benchmark
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " [options]\n\n";
    std::cerr << "Run LLM inference using torch_tensorrt built engines with TensorRT-Edge-LLM C++ runtime.\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --help                    Display this help message\n";
    std::cerr << "  --engineDir=<path>        Path to engine directory (required)\n";
    std::cerr << "  --prompt=<text>           Input prompt for generation\n";
    std::cerr << "  --inputFile=<path>        Path to input JSON file with prompts\n";
    std::cerr << "  --outputFile=<path>       Path to output JSON file\n";
    std::cerr << "  --maxTokens=<n>           Maximum tokens to generate (default: 128)\n";
    std::cerr << "  --temperature=<f>         Sampling temperature (default: 1.0)\n";
    std::cerr << "  --topK=<n>                Top-K sampling (default: 50)\n";
    std::cerr << "  --topP=<f>                Top-P sampling (default: 0.9)\n";
    std::cerr << "  --batchSize=<n>           Batch size (default: 1)\n";
    std::cerr << "  --debug                   Enable debug logging\n";
    std::cerr << "  --dumpProfile             Dump profiling summary\n";
    std::cerr << "  --warmup=<n>              Number of warmup iterations (default: 0)\n";
    std::cerr << "  --benchmark               Run in benchmark mode\n";
    std::cerr << "  --isl=<n>                 Input sequence length for benchmark (default: 128)\n";
    std::cerr << "  --osl=<n>                 Output sequence length for benchmark (default: 128)\n";
    std::cerr << "\nExample:\n";
    std::cerr << "  " << programName << " --engineDir=/models/qwen2.5-1.5b --prompt=\"Hello, how are you?\"\n";
}

bool parseArgs(TorchTrtInferenceArgs& args, int argc, char* argv[])
{
    static struct option options[] = {
        {"help", no_argument, 0, TorchTrtInferenceOptionId::HELP},
        {"engineDir", required_argument, 0, TorchTrtInferenceOptionId::ENGINE_DIR},
        {"prompt", required_argument, 0, TorchTrtInferenceOptionId::PROMPT},
        {"inputFile", required_argument, 0, TorchTrtInferenceOptionId::INPUT_FILE},
        {"outputFile", required_argument, 0, TorchTrtInferenceOptionId::OUTPUT_FILE},
        {"maxTokens", required_argument, 0, TorchTrtInferenceOptionId::MAX_TOKENS},
        {"temperature", required_argument, 0, TorchTrtInferenceOptionId::TEMPERATURE},
        {"topK", required_argument, 0, TorchTrtInferenceOptionId::TOP_K},
        {"topP", required_argument, 0, TorchTrtInferenceOptionId::TOP_P},
        {"batchSize", required_argument, 0, TorchTrtInferenceOptionId::BATCH_SIZE},
        {"debug", no_argument, 0, TorchTrtInferenceOptionId::DEBUG},
        {"dumpProfile", no_argument, 0, TorchTrtInferenceOptionId::DUMP_PROFILE},
        {"warmup", required_argument, 0, TorchTrtInferenceOptionId::WARMUP},
        {"benchmark", no_argument, 0, TorchTrtInferenceOptionId::BENCHMARK},
        {"isl", required_argument, 0, TorchTrtInferenceOptionId::ISL},
        {"osl", required_argument, 0, TorchTrtInferenceOptionId::OSL},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case TorchTrtInferenceOptionId::HELP: args.help = true; return true;
        case TorchTrtInferenceOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case TorchTrtInferenceOptionId::PROMPT: args.prompt = optarg; break;
        case TorchTrtInferenceOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case TorchTrtInferenceOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case TorchTrtInferenceOptionId::MAX_TOKENS:
            try { args.maxTokens = std::stoi(optarg); }
            catch (...) { LOG_ERROR("Invalid maxTokens: %s", optarg); return false; }
            break;
        case TorchTrtInferenceOptionId::TEMPERATURE:
            try { args.temperature = std::stof(optarg); }
            catch (...) { LOG_ERROR("Invalid temperature: %s", optarg); return false; }
            break;
        case TorchTrtInferenceOptionId::TOP_K:
            try { args.topK = std::stoi(optarg); }
            catch (...) { LOG_ERROR("Invalid topK: %s", optarg); return false; }
            break;
        case TorchTrtInferenceOptionId::TOP_P:
            try { args.topP = std::stof(optarg); }
            catch (...) { LOG_ERROR("Invalid topP: %s", optarg); return false; }
            break;
        case TorchTrtInferenceOptionId::BATCH_SIZE:
            try { args.batchSize = std::stoi(optarg); }
            catch (...) { LOG_ERROR("Invalid batchSize: %s", optarg); return false; }
            break;
        case TorchTrtInferenceOptionId::DEBUG: args.debug = true; break;
        case TorchTrtInferenceOptionId::DUMP_PROFILE: args.dumpProfile = true; break;
        case TorchTrtInferenceOptionId::WARMUP:
            try { args.warmup = std::stoi(optarg); }
            catch (...) { LOG_ERROR("Invalid warmup: %s", optarg); return false; }
            break;
        case TorchTrtInferenceOptionId::BENCHMARK: args.benchmark = true; break;
        case TorchTrtInferenceOptionId::ISL:
            try { args.isl = std::stoi(optarg); }
            catch (...) { LOG_ERROR("Invalid isl: %s", optarg); return false; }
            break;
        case TorchTrtInferenceOptionId::OSL:
            try { args.osl = std::stoi(optarg); }
            catch (...) { LOG_ERROR("Invalid osl: %s", optarg); return false; }
            break;
        default: return false;
        }
    }

    // Validate required arguments
    if (args.engineDir.empty())
    {
        LOG_ERROR("--engineDir is required");
        return false;
    }

    if (!args.benchmark && args.prompt.empty() && args.inputFile.empty())
    {
        LOG_ERROR("Either --prompt, --inputFile, or --benchmark is required");
        return false;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    return true;
}

std::vector<std::string> loadPromptsFromFile(std::string const& inputFile)
{
    std::vector<std::string> prompts;
    std::ifstream file(inputFile);
    if (!file.is_open())
    {
        LOG_ERROR("Failed to open input file: %s", inputFile.c_str());
        return prompts;
    }

    try
    {
        Json inputData = Json::parse(file);
        if (inputData.contains("prompts") && inputData["prompts"].is_array())
        {
            for (auto const& p : inputData["prompts"])
            {
                prompts.push_back(p.get<std::string>());
            }
        }
        else if (inputData.contains("requests") && inputData["requests"].is_array())
        {
            // Support llm_inference.cpp format
            for (auto const& req : inputData["requests"])
            {
                if (req.contains("messages") && req["messages"].is_array())
                {
                    for (auto const& msg : req["messages"])
                    {
                        if (msg["role"] == "user")
                        {
                            if (msg["content"].is_string())
                            {
                                prompts.push_back(msg["content"].get<std::string>());
                            }
                            else if (msg["content"].is_array())
                            {
                                for (auto const& c : msg["content"])
                                {
                                    if (c["type"] == "text")
                                    {
                                        prompts.push_back(c["text"].get<std::string>());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse input file: %s", e.what());
    }

    return prompts;
}

int main(int argc, char* argv[])
{
    TorchTrtInferenceArgs args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    LOG_INFO("=== Torch-TensorRT LLM Inference with TensorRT-Edge-LLM Runtime ===");
    LOG_INFO("Engine directory: %s", args.engineDir.c_str());

    // Load plugin library
    auto pluginHandles = loadEdgellmPluginLib();

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Memory monitoring
    MemoryMonitor memoryMonitor;
    if (args.dumpProfile)
    {
        memoryMonitor.start();
    }

    // Initialize runtime
    std::unique_ptr<rt::TorchTrtInferenceRuntime> runtime;
    try
    {
        runtime = std::make_unique<rt::TorchTrtInferenceRuntime>(args.engineDir, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize runtime: %s", e.what());
        return EXIT_FAILURE;
    }

    auto const& config = runtime->getEngineConfig();
    LOG_INFO("Model type: %s", config.modelType.c_str());
    LOG_INFO("Layers: %d, Heads: %d/%d, Hidden: %d, Vocab: %d", config.numDecoderLayers, config.numAttentionHeads,
        config.numKeyValueHeads, config.hiddenSize, config.vocabSize);

    // Capture CUDA graphs for decoding (optional, can improve performance)
    // Note: CUDA graph capture may not work well with the attention plugin
    // that updates KV caches in-place. Keep disabled for now.
    LOG_INFO("CUDA graph capture disabled for attention plugin compatibility");

    // Prepare prompts
    std::vector<std::string> prompts;
    if (args.benchmark)
    {
        // Generate random prompts for benchmarking
        LOG_INFO("Benchmark mode: ISL=%d, OSL=%d, batch_size=%d", args.isl, args.osl, args.batchSize);
        for (int32_t i = 0; i < args.batchSize; ++i)
        {
            std::string dummyPrompt(args.isl, 'a'); // Dummy prompt
            prompts.push_back(dummyPrompt);
        }
        args.maxTokens = args.osl;
    }
    else if (!args.inputFile.empty())
    {
        prompts = loadPromptsFromFile(args.inputFile);
        if (prompts.empty())
        {
            LOG_ERROR("No prompts loaded from input file");
            return EXIT_FAILURE;
        }
    }
    else
    {
        prompts.push_back(args.prompt);
    }

    LOG_INFO("Processing %zu prompts with max_tokens=%d", prompts.size(), args.maxTokens);

    // Warmup
    if (args.warmup > 0)
    {
        LOG_INFO("Running %d warmup iterations...", args.warmup);
        setProfilingEnabled(false);

        for (int32_t w = 0; w < args.warmup; ++w)
        {
            rt::TorchTrtGenerationRequest warmupReq;
            warmupReq.prompts = {prompts[0]};
            warmupReq.maxGenerateLength = std::min(args.maxTokens, 32);
            warmupReq.temperature = args.temperature;

            rt::TorchTrtGenerationResponse warmupResp;
            runtime->handleRequest(warmupReq, warmupResp, stream);
        }

        LOG_INFO("Warmup complete");
    }

    // Enable profiling for actual runs
    if (args.dumpProfile)
    {
        setProfilingEnabled(true);
    }

    // Process prompts in batches
    std::vector<std::string> allOutputs;
    size_t numBatches = (prompts.size() + args.batchSize - 1) / args.batchSize;

    for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx)
    {
        size_t startIdx = batchIdx * args.batchSize;
        size_t endIdx = std::min(startIdx + args.batchSize, prompts.size());

        rt::TorchTrtGenerationRequest request;
        for (size_t i = startIdx; i < endIdx; ++i)
        {
            request.prompts.push_back(prompts[i]);
        }
        request.maxGenerateLength = args.maxTokens;
        request.temperature = args.temperature;
        request.topK = args.topK;
        request.topP = args.topP;

        rt::TorchTrtGenerationResponse response;
        bool success = runtime->handleRequest(request, response, stream);

        if (!success)
        {
            LOG_ERROR("Request failed for batch %zu", batchIdx);
            continue;
        }

        // Collect outputs
        for (auto const& output : response.outputTexts)
        {
            allOutputs.push_back(output);
            if (!args.benchmark)
            {
                LOG_INFO("Generated: %s", output.c_str());
            }
        }

        if (!args.benchmark)
        {
            LOG_INFO("Batch %zu/%zu completed: %zu tokens generated", batchIdx + 1, numBatches,
                response.numGeneratedTokens.empty() ? 0 : response.numGeneratedTokens[0]);
        }
    }

    // Disable profiling
    if (args.dumpProfile)
    {
        setProfilingEnabled(false);
        memoryMonitor.stop();
    }

    // Print profiling results
    if (args.dumpProfile)
    {
        std::ostringstream profileOutput;
        profileOutput << "\n=== Performance Summary ===" << std::endl;
        outputPrefillProfile(profileOutput, runtime->getPrefillMetrics());
        outputGenerationProfile(profileOutput, runtime->getGenerationMetrics());
        outputMemoryProfile(profileOutput, memoryMonitor);
        profileOutput << "=====================================" << std::endl;
        LOG_INFO("%s", profileOutput.str().c_str());
    }

    // Save output to file if requested
    if (!args.outputFile.empty())
    {
        Json outputData;
        outputData["engine_dir"] = args.engineDir;
        outputData["model_type"] = config.modelType;
        outputData["responses"] = Json::array();

        for (size_t i = 0; i < allOutputs.size(); ++i)
        {
            Json resp;
            resp["prompt"] = i < prompts.size() ? prompts[i] : "";
            resp["output"] = allOutputs[i];
            outputData["responses"].push_back(resp);
        }

        std::ofstream outFile(args.outputFile);
        if (outFile.is_open())
        {
            outFile << outputData.dump(2);
            LOG_INFO("Output saved to: %s", args.outputFile.c_str());
        }
        else
        {
            LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    LOG_INFO("=== Inference Complete ===");
    return EXIT_SUCCESS;
}

# Running Torch-TensorRT Engines with TensorRT-Edge-LLM C++ Runtime

## Table of Contents
1. [Overview](#1-overview)
2. [Overall Architecture](#2-overall-architecture)
3. [Understanding TensorRT Binding](#3-understanding-tensorrt-binding)
4. [Engine Build and Save (Python)](#4-engine-build-and-save-python)
5. [C++ Runtime Structure](#5-c-runtime-structure)
6. [Input/Output Tensor Shapes](#6-inputoutput-tensor-shapes)
7. [KV Cache Management](#7-kv-cache-management)
8. [Generation Loop](#8-generation-loop)
9. [Key Findings](#9-key-findings)
10. [How to Run](#10-how-to-run)
11. [Frequently Asked Questions (FAQ)](#11-frequently-asked-questions-faq)
    - [11.1 Differences in Engine Storage Methods](#111-differences-in-engine-storage-methods)
    - [11.2 TensorRT Engine Build and Serialization](#112-tensorrt-engine-build-and-serialization)
    - [11.3 TensorRT Version Issues](#113-tensorrt-version-issues)
    - [11.4 KV Cache Dimension Details](#114-kv-cache-dimension-details)
    - [11.5 In-place KV Cache Update Details](#115-in-place-kv-cache-update-details)
    - [11.6 What is mKVCaches?](#116-what-is-mkvcaches)

---

## 1. Overview

### 1.1 Purpose
This document explains how to run LLM (Large Language Model) TensorRT engines built with `torch-tensorrt` using the `TensorRT-Edge-LLM` C++ runtime.

### 1.2 Why Use C++ Runtime?

| Advantage | Description |
|------|------|
| **Performance** | Pure C++ inference without Python overhead |
| **Deployment** | Lightweight deployment to edge devices |
| **KV Cache Management** | Optimized memory management |
| **Integration** | Easy integration with existing C++ applications |

### 1.3 Overall Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   HuggingFace   │────▶│  torch-tensorrt │────▶│   .engine file  │
│     Model       │     │     compile     │     │   config.json   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Generated     │◀────│   C++ Runtime   │◀────│  Engine Load    │
│     Text        │     │   (Inference)   │     │   & Binding     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 2. Overall Architecture

### 2.1 System Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    torch_trt_inference.cpp                      │
│                    (Application Entry Point)                    │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│              TorchTrtInferenceRuntime                          │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ TorchTrtEngine   │  │    Tokenizer     │                    │
│  │    Runner        │  │                  │                    │
│  └────────┬─────────┘  └──────────────────┘                    │
│           │                                                     │
│  ┌────────▼─────────────────────────────────────────────────┐  │
│  │                  TensorRT Engine                          │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │            Attention Plugin (libNvInfer_edgellm)    │ │  │
│  │  │  - KV Cache management                              │ │  │
│  │  │  - RoPE (Rotary Position Embedding)                 │ │  │
│  │  │  - In-place KV Cache updates                        │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Main Classes

| Class | File | Role |
|--------|------|------|
| `TorchTrtInferenceRuntime` | `torchTrtInferenceRuntime.cpp/h` | High-level inference API, tokenizer management |
| `TorchTrtEngineRunner` | `torchTrtEngineRunner.cpp/h` | TensorRT engine execution, KV cache management |

---

## 3. Understanding TensorRT Binding

### 3.1 What is Binding?

**Binding** connects input/output tensors of a TensorRT engine with GPU memory.

```
┌─────────────────────────────────────────────────────────────┐
│                     TensorRT Engine                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Input Bindings           Output Bindings           │    │
│  │  ┌──────────────┐        ┌──────────────┐          │    │
│  │  │  input_ids   │        │   output0    │ (logits) │    │
│  │  │  ctx_len     │        │   output1    │ (delta_kv)│   │
│  │  │  kv_caches_0 │        │   output2    │          │    │
│  │  │  kv_caches_1 │        │   ...        │          │    │
│  │  │  ...         │        └──────────────┘          │    │
│  │  └──────────────┘                                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│   GPU Memory    │            │   GPU Memory    │
│  (Input Data)   │            │  (Output Data)  │
└─────────────────┘            └─────────────────┘
```

### 3.2 Binding Process

1. **Set Address**: `setTensorAddress()` - Tell the engine the GPU buffer address
2. **Set Shape**: `setInputShape()` - Specify actual shape for dynamic shapes
3. **Execute**: `enqueueV3()` - Run inference with data at the set addresses

### 3.3 Code Example

```cpp
// 1. Connect GPU buffer address to engine
bool status = mContext->setTensorAddress("input_ids", inputIds.rawPointer());

// 2. Set dynamic shape (tell input_ids shape)
status &= mContext->setInputShape("input_ids", inputIds.getShape().getTRTDims());

// 3. Execute after setting all bindings
bool executeStatus = mContext->enqueueV3(stream);
```

### 3.4 Torch-tensorrt Engine Binding Structure

```
Engine Bindings (Total 59)
├── Inputs (30)
│   ├── input_ids      : [batch, seq_len]         INT64
│   ├── ctx_len        : [batch]                  INT32
│   ├── kv_caches_0    : [batch, 2, num_kv_heads, max_seq_len, head_dim]  FP16
│   ├── kv_caches_1    : [batch, 2, num_kv_heads, max_seq_len, head_dim]  FP16
│   └── ... (28 layers)
│
└── Outputs (29)
    ├── output0        : [batch, seq_len, vocab_size]  FP16  (logits)
    ├── output1        : [batch, 2, num_kv_heads, seq_len, head_dim]  FP16  (delta_kv)
    └── ... (28 layers)
```

---

## 4. Engine Build and Save (Python)

### 4.1 Role of save_engine_for_cpp.py

```python
# Main flow
1. Load HuggingFace model
2. Replace Attention with Plugin
3. Compile with torch_tensorrt
4. Save Engine (.engine)
5. Save Config (config.json)
6. Save Tokenizer
7. Save Chat Template
```

### 4.2 Core Code Analysis

#### Attention Plugin Replacement

```python
# In plugin_utils.py
model = replace_attention_with_plugin(model, config, max_seq_len, device, dtype)
wrapper = LLMPluginWrapper(model)
```

**Role**: Replace standard PyTorch attention with TensorRT-Edge-LLM's optimized attention plugin.

**Why is it needed?**
- Efficient KV cache management
- RoPE (Rotary Position Embedding) optimization
- Memory usage optimization

#### torch_tensorrt Compilation

```python
trt_model = torch_tensorrt.dynamo.compile(
    ep,                                    # Exported Program
    inputs=[dummy_input_ids, dummy_pos_ids, dummy_kvs, dummy_ctx_len],
    enabled_precisions={torch.float32},    # Compilation precision
    use_explicit_typing=True,              # Use explicit typing
    device=device,
    min_block_size=1,
)
```

#### Engine Save

```python
# Extract serialized engine from TorchTensorRTModule
serialized_engine = trt_model._run_on_acc_0.serialized_engine

with open(engine_path, "wb") as f:
    f.write(serialized_engine)
```

### 4.3 config.json Structure

```json
{
  "model_type": "qwen2",
  "num_hidden_layers": 28,
  "num_attention_heads": 12,
  "num_key_value_heads": 2,
  "hidden_size": 1536,
  "head_dim": 128,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "torch_trt_config": {
    "max_batch_size": 1,
    "max_sequence_length": 2048,
    "dtype": "fp16",
    "use_attention_plugin": true
  }
}
```

---

## 5. C++ Runtime Structure

### 5.1 TorchTrtEngineRunner

#### Constructor Flow

```cpp
TorchTrtEngineRunner::TorchTrtEngineRunner(engineDir, stream)
{
    // 1. Load plugin library
    loadEdgellmPluginLib();  // libNvInfer_edgellm_plugin.so

    // 2. Parse config
    initializeConfigFromJson(configJson);  // Read config.json

    // 3. Load and deserialize engine
    mEngine = mRuntime->deserializeCudaEngine(engineData, engineSize);

    // 4. Create execution context
    mContext = mEngine->createExecutionContext();

    // 5. Initialize KV cache
    initializeKVCaches(stream);

    // 6. Initialize RoPE cache
    initializeRopeCache(configJson, stream);
}
```

#### Core Methods

| Method | Role |
|--------|------|
| `executePrefillStep()` | Process initial prompt (parallel) |
| `executeDecodingStep()` | Generate tokens one by one (sequential) |
| `bindEngineIO()` | Bind tensors to engine |
| `resetKVCaches()` | Initialize KV cache |

### 5.2 TorchTrtInferenceRuntime

#### Constructor Flow

```cpp
TorchTrtInferenceRuntime::TorchTrtInferenceRuntime(engineDir, config)
{
    // 1. Create engine runner
    mEngineRunner = std::make_unique<TorchTrtEngineRunner>(engineDir, stream);

    // 2. Load tokenizer
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    mTokenizer->loadFromHF(engineDir);

    // 3. Allocate working tensors
    mInputIds = rt::Tensor({batch, maxSeqLen}, GPU, INT64);
    mOutputLogits = rt::Tensor({batch, maxSeqLen, vocabSize}, GPU, HALF);
    // ... other tensors
}
```

#### Core Methods

| Method | Role |
|--------|------|
| `handleRequest()` | Receive text prompt and generate response |
| `generate()` | Perform generation with token IDs |
| `sampleGreedy()` | Sample next token from logits |

---

## 6. Input/Output Tensor Shapes

### 6.1 Prefill Stage

| Tensor | Shape | dtype | Description |
|--------|-------|-------|------|
| **input_ids** | `[batch, seq_len]` | INT64 | Input token IDs |
| **ctx_len** | `[batch]` | INT32 | Sequence length |
| **kv_caches_i** | `[batch, 2, num_kv_heads, max_seq_len, head_dim]` | FP16 | KV cache for layer i |
| **output0** (logits) | `[batch, seq_len, vocab_size]` | FP16 | Next token probabilities |
| **output{i+1}** (delta_kv) | `[batch, 2, num_kv_heads, seq_len, head_dim]` | FP16 | New KV values |

**Example (Qwen2.5-1.5B, batch=1, 7 input tokens):**

```
input_ids:    [1, 7]              # 7 token input
ctx_len:      [1]                 # value: 7
kv_caches_0:  [1, 2, 2, 2048, 128] # Store Key and Value separately
output0:      [1, 7, 151936]      # Logits at each position
output1:      [1, 2, 2, 7, 128]   # KV for 7 new tokens
```

### 6.2 Decoding Stage

| Tensor | Shape | dtype | Description |
|--------|-------|-------|------|
| **input_ids** | `[batch, 1]` | INT64 | Previously generated 1 token |
| **ctx_len** | `[batch]` | INT32 | Cumulative length (prefill + generated tokens) |
| **kv_caches_i** | `[batch, 2, num_kv_heads, max_seq_len, head_dim]` | FP16 | Updated KV cache |
| **output0** (logits) | `[batch, 1, vocab_size]` | FP16 | Next token probabilities |
| **output{i+1}** (delta_kv) | `[batch, 2, num_kv_heads, 1, head_dim]` | FP16 | KV for 1 token |

**Example (first decoding step):**

```
input_ids:    [1, 1]              # 1 generated token
ctx_len:      [1]                 # value: 8 (7+1)
kv_caches_0:  [1, 2, 2, 2048, 128] # Data at positions 0~6
output0:      [1, 1, 151936]      # Logits for 1 token
output1:      [1, 2, 2, 1, 128]   # KV for 1 token
```

### 6.3 KV Cache Dimension Explanation

```
kv_cache shape: [batch, 2, num_kv_heads, max_seq_len, head_dim]
                  │     │       │            │           │
                  │     │       │            │           └── 128 (head dimension)
                  │     │       │            └── 2048 (max sequence length)
                  │     │       └── 2 (Number of KV heads in Qwen2.5-1.5B)
                  │     └── 2 (Key and Value, stored separately)
                  └── 1 (batch size)
```

---

## 7. KV Cache Management

### 7.1 What is KV Cache?

Memory that stores previously computed Keys and Values in LLMs to prevent recomputation.

```
Prefill (prompt processing):
┌─────────────────────────────────────────────────┐
│ "What is 2+2?" (7 tokens)                       │
│  ↓                                               │
│ KV Cache: [Store KV at pos 0~6]                 │
└─────────────────────────────────────────────────┘

Decoding (token generation):
┌─────────────────────────────────────────────────┐
│ Input: 1 new token                               │
│ KV Cache read: [pos 0~6] (previous context)     │
│ KV Cache write: [pos 7] (new token's KV)        │
│  ↓                                               │
│ Output: next token                               │
└─────────────────────────────────────────────────┘
```

### 7.2 In-place Update

**Key Finding**: TensorRT-Edge-LLM's attention plugin updates KV cache **in-place**.

```cpp
// Wrong approach (unnecessary copying)
// for (int i = 0; i < numLayers; ++i) {
//     copyDeltaKVToCache(mDeltaKVCaches[i], mKVCaches[i], startPos);
// }

// Correct approach (plugin auto-updates)
// When mKVCaches is bound in bindEngineIO(),
// the plugin directly updates mKVCaches during execution
```

### 7.3 Meaning of ctx_len

`ctx_len` tells the attention plugin how many tokens have been processed so far.

| Stage | ctx_len value | Meaning |
|------|------------|------|
| Prefill | `seq_len` (e.g., 7) | 7 tokens processed |
| Decode 1 | `seq_len + 1` (8) | Store new KV at position 8 |
| Decode 2 | `seq_len + 2` (9) | Store new KV at position 9 |
| ... | ... | ... |

---

## 8. Generation Loop

### 8.1 Overall Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    generate() function                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Input preparation                                        │
│     ├── Tokenize: "What is 2+2?" → [token IDs]             │
│     ├── Pad to max_input_len                                │
│     └── Copy to GPU                                         │
│                                                              │
│  2. Prefill Step                                             │
│     ├── bindEngineIO(input_ids, ctx_len, kv_caches)         │
│     ├── mContext->enqueueV3(stream)                         │
│     ├── [Plugin: KV cache updated]                          │
│     └── sampleGreedy(logits) → first generated token       │
│                                                              │
│  3. Decoding Loop                                            │
│     for step in range(max_new_tokens):                      │
│         ├── input_ids = [previous generated token]          │
│         ├── ctx_len = prefill_len + step                    │
│         ├── bindEngineIO(...)                               │
│         ├── mContext->enqueueV3(stream)                     │
│         ├── [Plugin: Add new KV to cache]                   │
│         ├── sampleGreedy(logits) → next token               │
│         └── if EOS token: break                             │
│                                                              │
│  4. Return results                                           │
│     └── [generated token IDs]                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Detailed Code Analysis

#### Prefill Step

```cpp
// torchTrtInferenceRuntime.cpp

// 1. Initialize KV cache
mEngineRunner->resetKVCaches(batchSize, stream);

// 2. Prepare input tensors and copy to GPU
mInputIds.reshape({batchSize, maxInputLen});
cudaMemcpyAsync(mInputIds.rawPointer(), paddedInputIds.data(), ...);

// 3. Execute prefill
mEngineRunner->executePrefillStep(mInputIds, mPositionIds, mContextLengths, mOutputLogits, stream);

// 4. Sample first token
currentTokens = sampleGreedy(mOutputLogits, maxInputLen, 1.0f, stream);
```

#### Decoding Loop

```cpp
for (int32_t step = 1; step < maxNewTokens && numUnfinished > 0; ++step)
{
    // 1. Prepare input (previously generated token)
    stepInputIds[b] = static_cast<int64_t>(currentTokens[b]);
    stepContextLengths[b] = contextLengthsHost[b] + step;  // Cumulative length

    // 2. Copy to GPU
    cudaMemcpyAsync(mInputIds.rawPointer(), stepInputIds.data(), ...);

    // 3. Execute decoding
    mEngineRunner->executeDecodingStep(mInputIds, mPositionIds, mContextLengths, mOutputLogits, stream);

    // 4. Sample next token
    currentTokens = sampleGreedy(mOutputLogits, 1, 1.0f, stream);

    // 5. Check EOS
    if (currentTokens[b] == eosTokenId) {
        finished[b] = true;
    }
}
```

### 8.3 sampleGreedy Function

```cpp
std::vector<int32_t> sampleGreedy(const Tensor& logits, int32_t actualSeqLen, float temperature, cudaStream_t stream)
{
    // logits shape: [batch, max_seq_len, vocab_size]
    // actualSeqLen: actual valid sequence length
    
    int32_t batchSize = logits.getShape()[0];
    int32_t vocabSize = logits.getShape()[2];
    
    // Extract logits at last token position
    // offset = (batch * actualSeqLen + (actualSeqLen - 1)) * vocabSize * sizeof(half)
    for (int32_t b = 0; b < batchSize; ++b) {
        size_t offset = (b * actualSeqLen + (actualSeqLen - 1)) * vocabBytes;
        cudaMemcpyAsync(hostLogits + b * vocabBytes, logits.ptr + offset, ...);
    }
    
    // Compute argmax (select token with highest probability)
    for (int32_t b = 0; b < batchSize; ++b) {
        int32_t maxIdx = 0;
        float maxVal = -infinity;
        for (int32_t v = 0; v < vocabSize; ++v) {
            float val = __half2float(logitsPtr[v]);
            if (val > maxVal) { maxVal = val; maxIdx = v; }
        }
        selectedTokens[b] = maxIdx;
    }
    
    return selectedTokens;
}
```

---

## 9. Key Findings

### 9.1 Plugin's In-place KV Cache Update

```cpp
// ❌ Wrong understanding: need to copy delta KV separately
updateKVCaches(batchSize, seqLen, startPos, stream);  // Unnecessary!

// ✅ Correct understanding: Plugin directly updates kv_caches
// When mKVCaches is bound in bindEngineIO,
// mKVCaches is automatically updated during plugin execution!
```

### 9.2 Input Data Type

```cpp
// ❌ Wrong
mInputIds = Tensor({batch, seq}, GPU, INT32);  // torch-tensorrt expects INT64

// ✅ Correct
mInputIds = Tensor({batch, seq}, GPU, INT64);  // Use INT64
```

### 9.3 Binding Names

Binding names for engines built with torch-tensorrt:

| Role | Name |
|------|------|
| Input IDs | `input_ids` |
| Context Length | `ctx_len` |
| KV Cache (layer i) | `kv_caches_i` (e.g., `kv_caches_0`) |
| Logits Output | `output0` |
| Delta KV (layer i) | `output{i+1}` (e.g., `output1`) |

### 9.4 Using Actual Sequence Length

```cpp
// ❌ Wrong: Using tensor shape
int32_t seqLen = logits.getShape()[1];  // Reshaped max size

// ✅ Correct: Pass actual processed sequence length
sampleGreedy(logits, actualSeqLen, temperature, stream);
```

---

## 10. How to Run

### 10.1 Engine Build (Python)

```bash
cd /develop/TensorRT/tools/llm

python save_engine_for_cpp.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir /path/to/engine_output \
    --precision FP16 \
    --max-seq-len 2048
```

**Output files:**
```
/path/to/engine_output/
├── model.engine                  # TensorRT engine
├── config.json                   # Model configuration
├── tokenizer.json                # Tokenizer
├── tokenizer_config.json
├── vocab.json (or other format)
└── processed_chat_template.json  # Chat template
```

### 10.2 C++ Build

```bash
cd /develop/TensorRT/TensorRT-Edge-LLM-release/build
cmake ..
make torch_trt_inference -j8
```

### 10.3 Execution

```bash
# Set plugin library path
export EDGELLM_PLUGIN_PATH=/develop/TensorRT/TensorRT-Edge-LLM-release/build/libNvInfer_edgellm_plugin.so

# Run
./examples/llm/torch_trt_inference \
    --engineDir=/path/to/engine_output \
    --prompt="What is parallel programming?" \
    --maxTokens=100
```

### 10.4 Example Output

```
[INFO] === Torch-TensorRT LLM Inference with TensorRT-Edge-LLM Runtime ===
[INFO] Engine directory: /path/to/engine_output
[INFO] Loaded config: modelType=qwen2, numDecoderLayers=28, ...
[INFO] TorchTrtEngineRunner initialized successfully
[INFO] Tokenizer loaded successfully
[INFO] Processing 1 prompts with max_tokens=100

Generated: Parallel programming is a programming paradigm that involves 
executing multiple tasks simultaneously on multiple processors or computing 
resources. The goal of parallel programming is to take advantage of the 
computational power of modern multi-core processors...

[INFO] Batch 1/1 completed: 100 tokens generated
[INFO] === Inference Complete ===
```

---

## Appendix: File Structure

```
TensorRT-Edge-LLM-release/
├── cpp/
│   └── runtime/
│       ├── torchTrtEngineRunner.h      # Engine execution class header
│       ├── torchTrtEngineRunner.cpp    # Engine execution implementation
│       ├── torchTrtInferenceRuntime.h  # High-level API header
│       └── torchTrtInferenceRuntime.cpp # High-level API implementation
├── examples/
│   └── llm/
│       └── torch_trt_inference.cpp     # Example application
└── build/
    ├── libNvInfer_edgellm_plugin.so    # Attention plugin
    └── examples/llm/
        └── torch_trt_inference         # Built executable

TensorRT/ (torch-tensorrt)
└── tools/
    └── llm/
        ├── save_engine_for_cpp.py      # Engine save utility
        ├── plugin_utils.py             # Plugin utilities
        └── run_llm.py                  # Python execution example
```

---

## 11. Frequently Asked Questions (FAQ)

This section answers advanced questions that may arise while reading the guide.

---

### 11.1 Differences in Engine Storage Methods

#### Q: What's the difference between saving with `serialized_engine` and `torch.jit.save`?

**Two storage methods:**

| Method | Storage Format | Content |
|------|----------|------|
| `serialized_engine` | `.engine` (binary) | Pure TensorRT engine only |
| `torch.jit.save` | `.ts` (TorchScript) | TensorRT engine + PyTorch wrapper |

**1. Raw Engine Save (`.engine`)**
```python
# Save pure TensorRT engine binary only
serialized_engine = trt_model._run_on_acc_0.serialized_engine
with open("model.engine", "wb") as f:
    f.write(serialized_engine)
```

**Features:**
- Pure TensorRT engine binary
- Can be loaded directly with C++ TensorRT API
- Can be used without PyTorch
- Smaller size

**2. TorchScript Save (`.ts`)**
```python
# Save with PyTorch wrapper
torch.jit.save(trt_model, "model.ts")
# or
torch_tensorrt.save(trt_model, "model.ts", output_format="torchscript")
```

**Features:**
- TensorRT engine included in TorchScript container
- Load with `torch.jit.load()`
- Requires PyTorch environment
- Includes input/output processing logic

**Visual Comparison:**
```
┌─────────────────────────────────────────────────────────────────┐
│ .engine file                                                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │            TensorRT Engine Binary (pure)                    │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ .ts file (TorchScript)                                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                   TorchScript Container                     │ │
│ │  ┌────────────────────────────────────────────────────────┐ │ │
│ │  │ Pre-processing (input conversion)                      │ │ │
│ │  ├────────────────────────────────────────────────────────┤ │ │
│ │  │ TensorRT Engine Binary                                  │ │ │
│ │  ├────────────────────────────────────────────────────────┤ │ │
│ │  │ Post-processing (output conversion)                    │ │ │
│ │  └────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Q: Is the saved TensorRT engine serialized?**

**Yes, correct.** `serialized_engine` is already a serialized byte array.

```python
# Type of serialized_engine
type(serialized_engine)  # <class 'bytes'>

# Already serialized, can write directly to file
with open("model.engine", "wb") as f:
    f.write(serialized_engine)  # Write byte array directly
```

---

### 11.2 TensorRT Engine Build and Serialization

#### Q: Is the TensorRT engine built specifically for a GPU?

**Yes, correct.** TensorRT engines are **optimized for specific GPU architectures** during build.

**Engine Build Process:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Engine Build Process                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Network definition (ONNX, TorchScript, etc.)                │
│     └── Layers, operations, connections                         │
│                                                                  │
│  2. Builder configuration                                        │
│     └── Precision (FP16, INT8), max batch size, workspace, etc. │
│                                                                  │
│  3. GPU profiling (GPU dependent!)                               │
│     ├── Check current GPU's SM count                            │
│     ├── Measure memory bandwidth                                │
│     ├── Kernel benchmarking                                     │
│     └── Select optimal algorithms                               │
│                                                                  │
│  4. Generate optimized engine                                    │
│     ├── Layer fusion                                             │
│     ├── Tensor layout optimization                              │
│     ├── Kernel auto-tuning                                      │
│     └── Memory allocation planning                              │
│                                                                  │
│  5. Serialization                                                │
│     └── Convert optimized engine to byte stream                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**GPU Dependency Examples:**

| GPU | Optimization Factors |
|-----|------------|
| RTX 3090 (Ampere) | Tensor Core utilization, FP16 optimization |
| RTX 4090 (Ada) | FP8 support, different kernel selection |
| Jetson Orin | Memory constraints considered, DLA utilization |

**Serialize/Deserialize Relationship:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  Build → Serialize → Deserialize                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BUILD (slow, one-time)                                          │
│  ┌───────────────┐                                               │
│  │ ONNX/PyTorch  │                                               │
│  │    Model      │                                               │
│  └───────┬───────┘                                               │
│          │ trt.Builder.build_serialized_network()                │
│          │ (minutes to tens of minutes, GPU profiling)           │
│          ▼                                                       │
│  ┌───────────────┐                                               │
│  │  In-memory    │                                               │
│  │   Engine      │                                               │
│  └───────┬───────┘                                               │
│          │                                                       │
│  SERIALIZE (fast)                                                │
│          │ engine.serialize() or direct save                     │
│          ▼                                                       │
│  ┌───────────────┐                                               │
│  │  .engine file │◀─────── Save byte stream                     │
│  │  (binary)     │                                               │
│  └───────┬───────┘                                               │
│          │                                                       │
│  DESERIALIZE (fast)                                              │
│          │ runtime.deserialize_cuda_engine()                     │
│          │ (seconds)                                             │
│          ▼                                                       │
│  ┌───────────────┐                                               │
│  │  In-memory    │◀─────── Restore executable engine            │
│  │   Engine      │                                               │
│  └───────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**

1. **Build**: Profile on GPU and select optimal implementation (slow)
2. **Serialize**: Save optimized engine to file (fast)
3. **Deserialize**: Restore engine from file (fast)

**Cautions:**
- Engine may not work on different GPU than build GPU
- Deserialize fails with different TensorRT version

```cpp
// Deserialize in C++
IRuntime* runtime = createInferRuntime(logger);
ICudaEngine* engine = runtime->deserializeCudaEngine(engineData, engineSize);

if (engine == nullptr) {
    // Failed due to GPU mismatch or TensorRT version mismatch
    LOG_ERROR("Failed to deserialize engine");
}
```

---

### 11.3 TensorRT Version Issues

#### Q: Why do TensorRT version issues occur and how to match them?

**Version Mismatch Problem:**
```
[ERROR] IRuntime::deserializeCudaEngine: Error Code 6: API Usage Error 
(The engine plan file is not compatible with this version of TensorRT, 
expecting library version 10.12.0.36 got 10.13.3.9, please rebuild.)
```

**Two TensorRT Installation Methods:**

| Installation Method | Description | Use |
|----------|------|------|
| **pip install** | Python package (`pip install tensorrt`) | Build engine in Python |
| **System install** | deb/rpm or tar install | C++ development, provides headers/libraries |

**Version Check Methods:**

```bash
# Python TensorRT version
python -c "import tensorrt as trt; print(trt.__version__)"
# Output: 10.13.3.9

# System TensorRT version (C++ library)
ls -la /usr/lib/x86_64-linux-gnu/libnvinfer.so*
# Output: libnvinfer.so.10 -> libnvinfer.so.10.12.0
```

**Why are two versions needed?**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Environment                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  pip install tensorrt                                      │ │
│  │  - Python bindings                                         │ │
│  │  - Includes own TensorRT library                           │ │
│  │  - Version: 10.13.3.9                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                  Build engine                                    │
│                           │                                      │
│                           ▼                                      │
│                   .engine file                                   │
│                  (built with 10.13.3.9)                          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Load attempt
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    C++ Environment                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  System installed TensorRT                                 │ │
│  │  - Header files (/usr/include/...)                         │ │
│  │  - Libraries (libnvinfer.so)                               │ │
│  │  - Version: 10.12.0.36                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                  ❌ Version mismatch!                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Solutions:**

**Method 1: Match Python TensorRT version to system (recommended)**
```bash
# Check system TensorRT version
ls /usr/lib/x86_64-linux-gnu/libnvinfer.so.*
# libnvinfer.so.10.12.0 → version 10.12

# Install same version in Python
pip install tensorrt==10.12.0.36 tensorrt-cu12==10.12.0.36 --force-reinstall
```

**Method 2: Upgrade system TensorRT**
```bash
# Install latest version from NVIDIA official repository
# (requires root, affects entire system)
apt-get update
apt-get install tensorrt
```

**Method 3: Separate environments**
```bash
# Configure Python to use system TensorRT
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# Use system library instead of pip tensorrt
```

**Version Matching Checklist:**

| Check Item | Command |
|----------|--------|
| Python TensorRT | `python -c "import tensorrt; print(tensorrt.__version__)"` |
| System libnvinfer | `ls -la /usr/lib/*/libnvinfer.so*` |
| CUDA version | `nvcc --version` |
| cuDNN version | `cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR` |

---

### 11.4 KV Cache Dimension Details

#### Q: How do max_sequence_length, head_dim, and number of layers relate to KV cache?

**Transformer Basics Review:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transformer Decoder Layer                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: hidden_states [batch, seq_len, hidden_size]             │
│                │                                                 │
│                ▼                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Self-Attention                                │ │
│  │                                                            │ │
│  │    Q = W_q * hidden_states  [batch, seq, num_heads * d]   │ │
│  │    K = W_k * hidden_states  [batch, seq, num_kv_heads * d]│ │
│  │    V = W_v * hidden_states  [batch, seq, num_kv_heads * d]│ │
│  │                                                            │ │
│  │    Attention = softmax(Q @ K^T / sqrt(d)) @ V             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                │                                                 │
│                ▼                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              MLP (Feed Forward)                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                │                                                 │
│                ▼                                                 │
│  Output: hidden_states [batch, seq_len, hidden_size]            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why is KV Cache needed?**

LLMs generate tokens one by one in an **autoregressive** manner:

```
┌─────────────────────────────────────────────────────────────────┐
│ Without KV Cache (inefficient)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Step 1: Input "What"                                             │
│         → Compute Q, K, V → Attention → Output "is"             │
│                                                                  │
│ Step 2: Input "What is" (from scratch again!)                    │
│         → Recompute Q, K, V → Attention → Output "2"            │
│                                                                  │
│ Step 3: Input "What is 2" (from scratch again!)                  │
│         → Recompute Q, K, V again → Attention → Output "+"      │
│                                                                  │
│ 💥 Redundantly compute K, V for previous tokens every step!     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ With KV Cache (efficient)                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Step 1: Input "What"                                             │
│         → Compute Q, K, V                                        │
│         → Store K, V in cache ✅                                 │
│         → Attention → Output "is"                                │
│                                                                  │
│ Step 2: Input "is" (only 1 new token!)                           │
│         → Compute Q, K, V for new token only                     │
│         → Add new K, V to cache ✅                               │
│         → Attention with cached K, V → Output "2"                │
│                                                                  │
│ Step 3: Input "2"                                                │
│         → Compute Q, K, V for new token only                     │
│         → Add to cache → Attention → Output "+"                 │
│                                                                  │
│ ✅ Reuse K, V from cache for previous tokens!                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**KV Cache Dimension Details:**

```
kv_cache shape: [batch, 2, num_kv_heads, max_seq_len, head_dim]
```

| Dimension | Name | Example Value | Meaning |
|------|------|---------|------|
| 0 | `batch` | 1 | Number of sequences processed simultaneously |
| 1 | `2` | 2 | Store Key and Value separately |
| 2 | `num_kv_heads` | 2 | Number of KV heads (for GQA/MQA) |
| 3 | `max_seq_len` | 2048 | Maximum number of tokens that can be stored |
| 4 | `head_dim` | 128 | Dimension of each head |

**Memory Calculation Example (Qwen2.5-1.5B):**

```python
# Configuration
batch = 1
num_layers = 28         # Number of decoder layers
num_kv_heads = 2        # Number of KV heads (GQA)
max_seq_len = 2048      # Max sequence length
head_dim = 128          # Head dimension
dtype_size = 2          # FP16 = 2 bytes

# KV cache size for 1 layer
single_layer_size = batch * 2 * num_kv_heads * max_seq_len * head_dim * dtype_size
# = 1 * 2 * 2 * 2048 * 128 * 2 = 2,097,152 bytes ≈ 2 MB

# Total KV cache size
total_kv_cache = single_layer_size * num_layers
# = 2 MB * 28 = 56 MB
```

**Visual Representation:**

```
┌─────────────────────────────────────────────────────────────────┐
│ KV Cache Structure (1 layer)                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [batch=1, 2, num_kv_heads=2, max_seq_len=2048, head_dim=128]   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Key Cache [1, 1, 2, 2048, 128]                              ││
│  │ ┌─────────────────────────────────────────────────────────┐ ││
│  │ │ Head 0: [2048 positions × 128 dims] ▓▓▓▓▓░░░░░░░░░░░░░░ │ ││
│  │ │         pos 0-6: has data (▓)                            │ ││
│  │ │         pos 7-2047: empty (░)                            │ ││
│  │ └─────────────────────────────────────────────────────────┘ ││
│  │ ┌─────────────────────────────────────────────────────────┐ ││
│  │ │ Head 1: [2048 positions × 128 dims] ▓▓▓▓▓░░░░░░░░░░░░░░ │ ││
│  │ └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Value Cache [1, 1, 2, 2048, 128]                            ││
│  │ ┌─────────────────────────────────────────────────────────┐ ││
│  │ │ Head 0: [2048 positions × 128 dims] ▓▓▓▓▓░░░░░░░░░░░░░░ │ ││
│  │ └─────────────────────────────────────────────────────────┘ ││
│  │ ┌─────────────────────────────────────────────────────────┐ ││
│  │ │ Head 1: [2048 positions × 128 dims] ▓▓▓▓▓░░░░░░░░░░░░░░ │ ││
│  │ └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**GQA (Grouped Query Attention) Explanation:**

Qwen2.5-1.5B uses **GQA**:
- `num_attention_heads` = 12 (number of Query heads)
- `num_kv_heads` = 2 (number of Key/Value heads)

```
┌─────────────────────────────────────────────────────────────────┐
│ GQA: Query heads share KV heads                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query Heads (12)            KV Heads (2)                        │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐    ┌───────────────────┐  │
│  │Q0 │ │Q1 │ │Q2 │ │Q3 │ │Q4 │ │Q5 │───▶│ K0, V0            │  │
│  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘    └───────────────────┘  │
│                                                                  │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐    ┌───────────────────┐  │
│  │Q6 │ │Q7 │ │Q8 │ │Q9 │ │Q10│ │Q11│───▶│ K1, V1            │  │
│  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘    └───────────────────┘  │
│                                                                  │
│  6 Query heads share 1 KV head                                   │
│  → 6x KV cache memory savings!                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 11.5 In-place KV Cache Update Details

#### Q: How exactly does in-place update work?

**Overall Flow:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Before Engine Execution                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  mKVCaches (GPU memory)                                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Layer 0: [1, 2, 2, 2048, 128]                               │ │
│  │          ▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░            │ │
│  │          pos 0-6: previous data                             │ │
│  │          pos 7+: empty                                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  (Layers 1-27 same)                                              │
│                                                                  │
│  mDeltaKVCaches (GPU memory) - output buffer                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Layer 0: [1, 2, 2, 2048, 128]                               │ │
│  │          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░            │ │
│  │          (empty)                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ bindEngineIO()
                               │ mContext->enqueueV3(stream)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Attention Plugin Executing                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Plugin internal operations:                                     │
│  1. Read existing K, V from mKVCaches (pos 0-6)                 │
│  2. Compute Q, K, V for new token                               │
│  3. Perform Attention operation                                  │
│  4. Write new K, V directly to mKVCaches (pos 7) ⬅️ IN-PLACE!  │
│  5. Also write new K, V to mDeltaKVCaches (pos 0)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    After Engine Execution                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  mKVCaches (GPU memory) - automatically updated! ✅              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Layer 0: [1, 2, 2, 2048, 128]                               │ │
│  │          ▓▓▓▓▓▓▓█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░            │ │
│  │          pos 0-6: previous data                             │ │
│  │          pos 7: newly added (█) ⬅️ Plugin wrote directly   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  mDeltaKVCaches (GPU memory) - only has new token's KV           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Layer 0: [1, 2, 2, 2048, 128]                               │ │
│  │          █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░            │ │
│  │          pos 0: new token's KV (not actually used)          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Code View:**

```cpp
// torchTrtEngineRunner.cpp - bindEngineIO function

bool TorchTrtEngineRunner::bindEngineIO(...)
{
    // ...
    
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        // 1. Bind KV Cache input
        //    Tell engine the GPU memory address of mKVCaches[i]
        std::string kvCacheName = "kv_caches_" + std::to_string(i);
        mContext->setTensorAddress(kvCacheName.c_str(), mKVCaches[i].rawPointer());
        
        // 2. Bind Delta KV output
        //    Tell engine the GPU memory address of mDeltaKVCaches[i]
        std::string deltaKVName = "output" + std::to_string(i + 1);
        mContext->setTensorAddress(deltaKVName.c_str(), mDeltaKVCaches[i].rawPointer());
    }
    
    // ...
}

bool TorchTrtEngineRunner::executePrefillStep(...)
{
    // 1. Bind I/O
    bindEngineIO(inputIds, positionIds, contextLengths, outputLogits, false);
    
    // 2. Execute engine
    //    At this point, Attention Plugin executes and
    //    mKVCaches is updated IN-PLACE!
    bool executeStatus = mContext->enqueueV3(stream);
    
    // 3. No separate KV cache copying needed!
    //    Plugin already updated mKVCaches
    
    return executeStatus;
}
```

**Why does it work this way?**

TensorRT-Edge-LLM's Attention Plugin is implemented as follows:

```cpp
// AttentionPlugin internals (conceptual code)

int AttentionPlugin::enqueue(...) 
{
    // inputs[1] = kv_cache (pointer to mKVCaches)
    void* kv_cache_ptr = inputs[1];
    
    // ctx_len = number of tokens processed so far
    int ctx_len = getCtxLen(inputs[2]);
    
    // 1. Read existing KV (0 ~ ctx_len-1)
    // 2. Compute Attention
    // 3. Write new KV directly to kv_cache (IN-PLACE)
    writeNewKV(kv_cache_ptr, new_k, new_v, ctx_len);
    
    // 4. Output Delta KV (optional)
    writeDeltaKV(outputs[layer_idx], new_k, new_v);
    
    return 0;
}
```

---

### 11.6 What is mKVCaches?

#### Q: Is mKVCaches an input? Output? Is it the full KV cache?

**mKVCaches is the "full KV Cache".**

```cpp
// torchTrtEngineRunner.h

class TorchTrtEngineRunner
{
private:
    // Per-layer KV cache tensors [batch, 2, num_kv_heads, max_seq_len, head_dim]
    std::vector<rt::Tensor> mKVCaches{};      // Full KV cache (28 layers)
    std::vector<rt::Tensor> mDeltaKVCaches{}; // New token's KV (28 layers)
};
```

**mKVCaches vs mDeltaKVCaches:**

| Tensor | Shape | Role | Input/Output |
|------|-------|------|----------|
| `mKVCaches[i]` | `[1, 2, 2, 2048, 128]` | Store full KV (all tokens) | **Both input and output** (in-place) |
| `mDeltaKVCaches[i]` | `[1, 2, 2, 2048, 128]` | Store only new token's KV | Output only |

**Visual Comparison:**

```
┌─────────────────────────────────────────────────────────────────┐
│ mKVCaches[0] - Full KV Cache (Layer 0)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  max_seq_len = 2048                                              │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ pos 0   pos 1   pos 2   ...   pos 7   pos 8   ...   pos 2047││
│  │ ┌─────┐ ┌─────┐ ┌─────┐      ┌─────┐ ┌─────┐      ┌─────┐   ││
│  │ │ K,V │ │ K,V │ │ K,V │ .... │ K,V │ │     │ .... │     │   ││
│  │ │tok 0│ │tok 1│ │tok 2│      │tok 7│ │     │      │     │   ││
│  │ └─────┘ └─────┘ └─────┘      └─────┘ └─────┘      └─────┘   ││
│  │  ▓▓▓▓▓   ▓▓▓▓▓   ▓▓▓▓▓        ▓▓▓▓▓   ░░░░░        ░░░░░   ││
│  │                                                               ││
│  │  ◀────── In use (8 tokens) ────────▶  ◀──── Empty ──────▶   ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Stores K, V for all tokens                                      │
│  Automatically updated during engine execution                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ mDeltaKVCaches[0] - Delta KV (Layer 0)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Receives only new token's KV as engine output                   │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ pos 0                                                        ││
│  │ ┌─────┐                                                      ││
│  │ │ K,V │  (K, V for 1 newly generated token)                 ││
│  │ │tok 8│                                                      ││
│  │ └─────┘                                                      ││
│  │  ▓▓▓▓▓   ░░░░░   ░░░░░        ░░░░░   ░░░░░        ░░░░░   ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ※ Current implementation doesn't use mDeltaKVCaches            │
│     Not needed since Plugin directly updates mKVCaches           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Code Verification:**

```cpp
// initializeKVCaches - allocate with full max_seq_len size
bool TorchTrtEngineRunner::initializeKVCaches(cudaStream_t stream)
{
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        // Allocate buffer for full sequence length
        mKVCaches.emplace_back(rt::Tensor(
            {mConfig.maxSupportedBatchSize, 
             2,                              // Key, Value
             mConfig.numKeyValueHeads, 
             mConfig.maxSequenceLength,      // 2048 (full!)
             mConfig.headDim},
            rt::DeviceType::kGPU, mConfig.dtype, tensorName));
            
        // Initialize to 0
        cudaMemsetAsync(mKVCaches.back().rawPointer(), 0, ...);
        
        // Delta KV also allocated with same size (for output buffer)
        mDeltaKVCaches.emplace_back(rt::Tensor(...));
    }
}
```

**Actual Usage Pattern:**

```
Prefill (process 7 prompt tokens):
┌────────────────────────────────────────────────────────────┐
│ Before: mKVCaches = [░░░░░░░░░░░░░░░░░░░░░░...] (empty)   │
│ After:  mKVCaches = [▓▓▓▓▓▓▓░░░░░░░░░░░░░░...] (0-6 filled)│
└────────────────────────────────────────────────────────────┘

Decoding Step 1:
┌────────────────────────────────────────────────────────────┐
│ Before: mKVCaches = [▓▓▓▓▓▓▓░░░░░░░░░░░░░░...]            │
│ After:  mKVCaches = [▓▓▓▓▓▓▓█░░░░░░░░░░░░░...] (7 added)  │
└────────────────────────────────────────────────────────────┘

Decoding Step 2:
┌────────────────────────────────────────────────────────────┐
│ Before: mKVCaches = [▓▓▓▓▓▓▓█░░░░░░░░░░░░░...]            │
│ After:  mKVCaches = [▓▓▓▓▓▓▓██░░░░░░░░░░░░...] (8 added)  │
└────────────────────────────────────────────────────────────┘
```

**Summary:**

| Question | Answer |
|------|------|
| Is mKVCaches input? Output? | **Both!** (in-place update) |
| Full KV? Only recent 1? | **Full KV** (max_seq_len size) |
| When updated? | Plugin auto-updates during engine execution |
| What about mDeltaKVCaches? | Stores only new token KV (currently unused) |

---

## Conclusion

Through this guide, I hope you've understood the complete flow of running LLM engines built with torch-tensorrt using the TensorRT-Edge-LLM C++ runtime. The key points are:

1. **Binding**: Connect GPU memory with engine inputs/outputs
2. **KV Cache**: Attention plugin updates in-place
3. **Generation**: Two stages - Prefill (parallel) → Decoding (sequential)

If you have questions, please check the relevant parts of the code directly.

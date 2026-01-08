# TensorRT Edge-LLM Python/Torch-TensorRT Integration

This directory contains utilities for using TensorRT Edge-LLM attention plugins with Python and torch_tensorrt.

## Overview

The standard TensorRT Edge-LLM attention plugin is designed for C++ runtime where in-place buffer updates work seamlessly. However, Python runtime (including torch_tensorrt) cannot retrieve in-place updated buffers directly. 

This integration adds **Delta KV Output Mode** to solve this problem:

- **Context Phase**: Output KV has shape `[B, 2, H, SeqLen, D]` (all processed tokens)
- **Generation Phase**: Output KV has shape `[B, 2, H, 1, D]` (only the new token)

The Python runtime must explicitly merge this delta into the main KV cache buffer.

## Building the Plugin

```bash
cd /path/to/TensorRT-Edge-LLM
mkdir -p build && cd build
cmake .. -DTRT_PACKAGE_DIR=/usr -DCUDA_VERSION=12.8
make -j$(nproc)
```

The plugin will be built at `build/libNvInfer_edgellm_plugin.so`

## Usage

### 1. Load the Plugin

```python
from tools.llm.plugin_utils import load_plugin, set_plugin_config

# Load the plugin library
load_plugin("/path/to/libNvInfer_edgellm_plugin.so")

# Configure plugin parameters
set_plugin_config(
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    max_seq_len=2048,
    enable_delta_kv_output=True  # Required for Python/torch_tensorrt
)
```

### 2. Replace Model Attention with Plugin

```python
from tools.llm.plugin_utils import replace_attention_with_plugin, LLMPluginWrapper

# Replace attention modules
model = replace_attention_with_plugin(model, config, max_seq_len=2048, device=device)

# Wrap for proper forward pass handling
wrapped_model = LLMPluginWrapper(model)
```

### 3. Create TensorRT-compiled Model

```python
import torch_tensorrt

# Export and compile
ep = torch.export.export(wrapped_model, args=(dummy_inputs,), dynamic_shapes=dynamic_shapes, strict=False)
trt_model = torch_tensorrt.dynamo.compile(ep, inputs=dummy_inputs, ...)
```

### 4. Generate with Delta KV Merging

```python
from tools.llm.plugin_utils import (
    create_kv_caches, 
    generate_with_plugin,
    merge_delta_kv
)

# Create KV caches
kv_caches = create_kv_caches(config, max_seq_len=2048, batch_size=1, device=device)

# Generate tokens (automatically handles delta merging)
output_ids, updated_kv_caches = generate_with_plugin(
    trt_model,
    input_ids,
    kv_caches,
    max_new_tokens=100,
    eos_token_id=tokenizer.eos_token_id
)
```

### Manual Delta Merging

If you're implementing custom generation logic:

```python
# After each forward pass
logits, delta_kvs = trt_model(input_ids, position_ids, kv_caches, ctx_len)

# Merge delta into main KV caches
merge_delta_kv(kv_caches, delta_kvs, cur_pos)
cur_pos += seq_len_processed
```

## Debug Logging

Enable debug logging to trace plugin execution:

```bash
export TRT_EDGELLM_DEBUG_PLUGIN=1
python your_script.py
```

## Plugin Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_q_heads` | Number of query attention heads | Required |
| `num_kv_heads` | Number of key/value heads (GQA) | Required |
| `head_size` | Dimension per attention head | Required |
| `enable_tree_attention` | Enable EAGLE speculative decoding | 0 |
| `enable_delta_kv_output` | Enable delta KV output mode | 0 |

## Architecture

```
+--------------------+     +---------------------+
|   Python Runtime   |     |    C++ Plugin       |
+--------------------+     +---------------------+
         |                          |
         v                          v
+--------------------+     +---------------------+
| torch.ops.xqa.attn |---->| AttentionPlugin     |
+--------------------+     +---------------------+
         |                          |
         |                 +--------+--------+
         |                 |                 |
         |                 v                 v
         |        +--------------+  +--------------+
         |        | FMHA Kernel  |  | XQA Kernel   |
         |        | (Prefill)    |  | (Decode)     |
         |        +--------------+  +--------------+
         |                 |                 |
         v                 v                 v
+--------------------+     +---------------------+
| merge_delta_kv()   |<----| Delta KV Output     |
+--------------------+     +---------------------+
         |
         v
+--------------------+
| Main KV Cache      |
+--------------------+
```

## Files

- `plugin_utils.py` - Main Python utilities
- `../../cpp/plugins/attentionPlugin/` - C++ plugin source
- `../../cpp/kernels/contextAttentionKernels/utilKernels.cu` - Delta copy kernel


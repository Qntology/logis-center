# 🚀 **vLLM.rs** – 用 Rust 实现的极简 vLLM

一个极速 ⚡、轻量的 🦀**Rust 实现版 vLLM**。

---

<p align="center">
  <a href="./ReadMe.md">English</a> |
  <a href="./ReadMe-CN.md">简体中文</a>
</p>

## ✨ 主要特性

* 🔧 **纯 Rust 后端** – 完全**不依赖 PyTorch**
* 🚀 **高性能** (支持**前缀缓存、PD分离**)
* 🧠 **极简核心** – 核心逻辑仅 **<3000 行** Rust 代码
* 💻 **跨平台支持** – 支持 **CUDA**（Linux/Windows）与 **Metal**（macOS）
* 🤖 **内置API 服务与ChatGPT风格网页** – Rust 原生实现的聊天与 API/Web 服务
* 🔌 **MCP集成** – Model Context Protocol 工具调用支持
* 📊 **Embedding与分词器API** – 完整的文本处理支持
* 🐍 **轻量 Python 接口** – 使用 PyO3 构建的 Python 聊天接口

---

## 📈 性能

### 💬 对话性能

> **A100** (单卡, 40G)

| 模型 | 格式 | 大小 | 输出速度 |
|------------------|---------------|----------|------------------------|
| Ministral-3-3B (Multimodal) | BF16 | 3B | **118.49** tokens/s |
| Ministral-3-3B (Multimodal) | ISQ (BF16->Q4K) | 3B | **171.92** tokens/s |
| Qwen3-VL-8B-Instruct (**Multimodal**) | Q8_0 | 8B | **105.31** tokens/s |
| Llama-3.1-8B | ISQ (BF16->Q4K) | 8B | **120.74** tokens/s |
| DeepSeek-R1-0528-Qwen3-8B | Q4_K_M | 8B | **124.87** tokens/s |
| GLM-4-9B-0414 | Q4_K_M | 9B | **70.38** tokens/s |
| QwQ-32B | Q4_K_M | 32B | **41.36** tokens/s |
| **Qwen3-30B-A3B** | Q4_K_M | **30B (MoE)**| **97.16** tokens/s  |
| **Qwen3.5-27B** | Q4_K_M | **27B (Dense)**| **45.20** tokens/s  |
| **Qwen3.5-27B** | FP8 | **27B (Dense)**| **42** tokens/s (**Hopper**)  |

> vLLM.rs 在 **Metal (Apple Silicon, M4)** 上的性能

  <details>

   | 模型 | 并发数 | 输出Tokens | 耗时 (s) | 吞吐量 (tokens/s) |
   |------------------|--------|--------|---------|-------------|
   | Qwen3-0.6B (BF16) |  128  | 63488       | 83.13s    | 763.73     |
   | Qwen3-0.6B (BF16) |  32      | 15872       | 23.53s    | 674.43    |
   | Qwen3-0.6B (BF16) | 1       | 456       | 9.23s    | 49.42       |
   | Qwen3-4B (Q4_K_M)  | 1       | 1683       | 52.62s    | 31.98     |
   | Qwen3-8B (Q2_K)  | 1       | 1300       | 80.88s    | 16.07     |
  </details>

查看 [**完整性能测试 →**](docs/performance.md)

## 🧠 支持的模型架构

* ✅ LLaMa 系列（LLaMa2、LLaMa3, IQuest-Coder）
* ✅ Qwen 系列（Qwen2、Qwen3）
* ✅ Qwen2/Qwen3 Moe 系列
* ✅ Qwen3-Next 系列
* ✅ Qwen3.5 Dense/MoE 系列（27B, 35B, 122B, 397B, 多模态）
* ✅ Mistral v1, v2
* ✅ Mistral-3 VL Reasoning (3B, 8B, 14B, 多模态)
* ✅ GLM4 (0414版本, **非ChatGLM**)
* ✅ GLM4 MoE (4.6/4.7)
* ✅ Phi3 / Phi4 (Phi-3, Phi-4, Phi-4-mini等)
* ✅ Gemma3 (多模态，不支持Flash Attention)
* ✅ Qwen3-VL (Dense, 多模态)
* ✅ MiroThinker-v1.5 (30B, 235B)

支持 **Safetensor** (包含GPTQ, AWQ, FP8-blockwise 量化格式) 和 **GGUF** 格式。

所有模型均支持硬件FP8 KvCache加速（需SM90+及关闭`flashinfer` 或 `flashattn` 特性）。

---
## 📚 文档
- [快速开始](docs/get_started.md)
- [Docker构建](docs/docker.md)
- [工具调用解析](docs/tool_parsing.md)
- [MCP集成与工具调用](docs/mcp_tool_calling.md)
- [Claude Code使用vLLM.rs后端](docs/claude_code.md)
- [OpenCode使用vLLM.rs后端](docs/open_code.md)
- [Goose AI Agent使用vLLM.rs后端](docs/goose.md)
- [Embedding](docs/embeddings.md)
- [多模态 (Qwen3-VL, Gemma3, Mistral3-VL)](docs/multimodal.md)
- [前缀缓存](docs/prefix-cache.md)
- [Rust库](docs/rust_crate.md)
- [Tokenize/Detokenize](docs/tokenize.md)
- [性能测试](docs/performance.md)

## 📘 使用方法（Python）
### 📦 使用 pip 安装
- 💡 **CUDA 计算能力 < 8.0**（例如 V100）需要**手动编译** （不支持 `flashattn`；或可使用 **Rust 模式**）。
- 💡 **预编译包** 默认启用了`flashattn` 或 `flashinfer` 特性，若使用 **FP8 KV Cache**，须将其移除后手动编译。

> 🍎 Metal（macOS）
```shell
python3 -m pip install vllm_rs
````

> 🟩 CUDA（Linux）

#### Ampere / Ada（SM80+）

```shell
#（可选）安装 NCCL
apt-get install -y libnccl2 libnccl-dev
python3 -m pip install vllm_rs
```

#### Hopper（SM90+）/ Blackwell（SM120+）

从 [Release Assets](https://github.com/guoqingbao/vllm.rs/releases/tag/v0.9.8) 下载 wheel，解压后安装 `.whl` 包。


### 🌐✨ API Server + ChatGPT风格内置网页
   💡使用`--ui-server`会同时启动ChatGPT风格网页, 此时无需其它客户端。

   💡如长文本请求导致当前生成过程卡顿，请使用 **Rust PD Server**方案 （见**PD分离**）

   💡前缀缓存为自动匹配公共前缀，无需 `session_id`。

  <details open>
    <summary>单卡 + GGUF模型</summary>

  ```bash
  # CUDA
  python3 -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --kv-fraction 0.7 --ui-server --prefix-cache
  # Metal/MacOS (MacOS Tahoe之前的系统可能会存在生成过慢问题)
  python3 -m vllm_rs.server --m unsloth/Qwen3-4B-GGUF --f Qwen3-4B-Q4_K_M.gguf --ui-server --max-model-len 32768 --prefix-cache
   ```
  </details>

   <details open>
    <summary>多卡 + 本地GGUF模型</summary>

   ```bash
   python3 -m vllm_rs.server --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --d 0,1 --ui-server --prefix-cache
   ```
  </details>

  </details>

   <details open>
    <summary>将未量化模型加载为GGUF模型</summary>

   ```bash
   # 同时将权重量化为Q4K格式，启用最长上下文：
   python3 -m vllm_rs.server --w /path/Qwen3.5-122B-A10B --isq q4k --d 0,1 --port 8000 --max-model-len 262144 --max-num-seqs 1 --ui-server --prefix-cache
   ```
  </details>


  <details open>
    <summary>FP8模型</summary>

```bash
# CUDA (MoE, Dense) sm90+ 设备需打开`cutlass`特性以支持FP8硬件加速
vllm-rs --m Qwen/Qwen3.5-27B-FP8 --ui-server --prefix-cache
# MacOS/Metal (Dense)
vllm-rs --m Qwen/Qwen3-4B-Instruct-2507-FP8 --ui-server --prefix-cache
```

  </details>

<details open>
    <summary>多模态模型 (Qwen3 VL, +图片)</summary>

```bash
# 使用内置ChatUI上传或提及图片url (格式 '.bmp', '.gif', '.jpeg', '.png', '.tiff', or '.webp')
python3 -m vllm_rs.server --m Qwen/Qwen3.5-35B-A3B-FP8 --ui-server --prefix-cache
```

  <details>
    <summary>运行GPTQ/AWQ Marlin兼容模型</summary>

```bash
python3 -m vllm_rs.server --w /home/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-Marlin
```
  </details>

查看 [**更多Python示例 →**](python/ReadMe.md)



## 📘 使用方法（Rust）

### CUDA平台安装 (CUDA 11+, 12+, 13.0)

> 方案 1：安装进Docker：
   <details>

```bash
cd vllm.rs
# 将 `sm_80` 更改至你当前的硬件特性，如 sm_75 (V100), sm_80 (A100), sm_86 (RTX4090), sm_90 (Hopper), sm_100/sm_120 (Blackwell); 将 CUDA 版本号 `12.9.0` 更改为与当前主机驱动匹配的版本; 将最后一个参数 `0` 更改为 `1` 启用Rust中国区镜像（适用于中国大陆）
./build_docker.sh "cuda,nccl,graph,flashinfer,cutlass,python" sm_80 12.9.0 0

# 还可以使用 `flash attention` 后端, 以及传入 `--prod` 以构建生产镜像
./build_docker.sh --prod "cuda,nccl,graph,flashattn,cutlass,python" sm_90 13.0.0
```
   </details>

参考 [**如何通过Docker运行 vLLM.rs 服务 →**](docs/docker.md)

> 方案 2：手动安装：

   <details open>

安装 Rust 工具链
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

安装构建依赖：
```sh
sudo apt-get update
sudo apt-get install -y git build-essential libssl-dev pkg-config
```

安装 CUDA Toolkit：
```sh
# CUDA 12.9 （版本号<= 本机驱动版本）
apt-get update
apt-get install -y \
  cuda-nvcc-12-9 \
  cuda-nvrtc-dev-12-9 \
  libcublas-dev-12-9 \
  libcurand-dev-12-9

# NCCL
apt-get install -y libnccl2 libnccl-dev
```
编译 vLLM.rs
```shell
# 只有单卡的情况下去掉 `nccl`
# 使用FP8 KVCache 或 V100及较老的机型去掉 `flashattn/flashinfer` 和 `cutlass`特性
# 默认安装进/usr/local/bin，使用`--dst`更改安装目录
./build.sh --install --features cuda,nccl,graph,flashinfer,cutlass

# 使用Flash Attention后端
./build.sh --install --features cuda,nccl,graph,flashattn,cutlass
```
  </details>

### MacOS/Metal平台安装

安装 [Xcode 命令行工具](https://mac.install.guide/commandlinetools/)

使用`metal`特性安装
```shell
cargo install --features metal
```

### 运行方式

使用 `--i` 启用交互模式 🤖，`--ui-server` 或 `--server` 启用服务模式 🌐，`--m`指定Huggingface模型，或`--w` 指定本地Safetensors模型路径 或`--f` 指定GGUF模型文件：


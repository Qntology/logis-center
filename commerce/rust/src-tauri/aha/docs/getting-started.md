# Getting Started

Welcome to AHA! This guide will help you get up and running quickly.

## Quick Start (5 Minutes)

### 1. Check Available Models

```bash
aha list
```

### 2. Download Your First Model

```bash
# Download a small text model to start
aha download -m qwen3-0.6b
```

### 3. Start the Service

```bash
# Start the HTTP API server
aha cli -m qwen3-0.6b
```

The service will start on `http://127.0.0.1:10100`

### 4. Make Your First API Call

In a new terminal:

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "Hello, AHA!"}
    ]
  }'
```

## Basic Concepts

### What is AHA?

AHA is a local AI inference engine that:
- Runs models on your machine (no cloud API)
- Supports multiple model types (text, vision, audio, OCR, ASR)
- Provides an OpenAI-compatible API
- Works offline once models are downloaded

### Model Categories

| Category | Description | Example Models |
|----------|-------------|----------------|
| **Text** | Text generation and chat | Qwen3, MiniCPM4 |
| **Vision** | Image understanding | Qwen2.5VL, Qwen3VL |
| **OCR** | Text extraction from images | DeepSeek-OCR, Hunyuan-OCR |
| **ASR** | Speech-to-text | GLM-ASR, Fun-ASR, Qwen3-ASR |
| **Audio** | Text-to-speech | VoxCPM, VoxCPM1.5 |
| **Image** | Image processing | RMBG2.0 (background removal) |

### CLI Commands

| Command | Purpose |
|---------|---------|
| `aha cli` | Download model and start service |
| `aha serv` | Start service with existing model |
| `aha download` | Download model only |
| `aha run` | Direct inference without server |
| `aha list` | List available models |

## Common Workflows

### Text Generation

```bash
# Start the service
aha cli -m qwen3-0.6b

# In another terminal, make a request
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Vision Understanding

```bash
# Start a vision model
aha cli -m qwen3vl-2b

# Analyze an image
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3vl-2b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image in detail."},
          {"type": "image", "image_url": {"url": "file:///path/to/image.jpg"}}
        ]
      }
    ],
    "stream": false
  }'
```

### OCR (Text Extraction)

```bash
# Start an OCR model
aha cli -m deepseek-ocr

# Extract text from an image
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Extract all text from this image."},
          {"type": "image", "image_url": {"url": "file:///path/to/document.jpg"}}
        ]
      }
    ]
  }'
```

### Speech Recognition (ASR)

```bash
# Start an ASR model
aha cli -m glm-asr-nano-2512

# Transcribe audio
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-asr-nano-2512",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Transcribe this audio."},
          {"type": "audio", "audio_url": {"url": "file:///path/to/audio.wav"}}
        ]
      }
    ]
  }'
```

### Text-to-Speech

```bash
# Start a TTS model
aha cli -m voxcpm1.5

# Generate speech
curl http://127.0.0.1:10100/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm1.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Hello, this is AHA speaking."},
          {"type": "audio", "audio_url": {"url": "https://package-release.coderbox.cn/aiway/test/other/%E5%93%AA%E5%90%92.wav"}}
        ]
      }
    ]
  }'
```

### Background Removal

```bash
# Start RMBG2.0 model
aha cli -m rmbg2.0

# Remove background from image
curl http://127.0.0.1:10100/images/remove_background \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rmbg2.0",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image_url": {"url": "file:///path/to/document.jpg"}}
        ]
      }
    ]
  }'
```

### Direct Inference (Without Server)

```bash
# Run inference directly without starting HTTP server
aha run -m qwen3-0.6b \
  -i "Write a haiku about AI" \
  --weight-path ~/.aha/Qwen/Qwen3-0.6B
```

## Configuration Options

### Change Port

```bash
# Use port 8080 instead of default 10100
aha cli -m qwen3-0.6b -p 8080
```

### Bind to All Interfaces

```bash
# Allow external access (use with caution)
aha cli -m qwen3-0.6b -a 0.0.0.0 -p 8080
```

### Use Local Model

```bash
# Skip download, use existing model
aha serv -m qwen3-0.6b \
  --weight-path /path/to/model \
  -p 8080
```

### Custom Save Directory

```bash
# Download model to specific directory
aha download -m qwen3vl-2b -s /data/models
```

## Streaming Responses

For chat/completions, using no "stream" field or "stream": true enables streaming, while "stream": false is for non-streaming responses:

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": false
  }'
```

## Model Selection Guide

### For Text Generation
- **qwen3-0.6b**: Fast, lightweight (~1.2 GB)
- **minicpm4-0.5b**: Small, efficient (~1 GB)

### For Vision Tasks
- **qwen3vl-2b**: Balanced performance (~4 GB)
- **qwen3vl-8b**: Better quality (~16 GB)

### For OCR
- **deepseek-ocr**: General purpose
- **hunyuan-ocr**: Good for Chinese text
- **paddleocr-vl**: Lightweight option

### For Speech Recognition
- **glm-asr-nano-2512**: Fast, accurate
- **fun-asr-nano-2512**: Good for Chinese
- **qwen3asr-0.6b**: Lightweight

### For Text-to-Speech
- **voxcpm1.5**: High quality Chinese

### For Background Removal
- **rmbg2.0**: State-of-the-art results

## Tips & Best Practices

### 1. Start Small

Begin with smaller models to understand the workflow:
```bash
aha download -m qwen3-0.6b
```

### 2. Use GPU Acceleration

Build with GPU support for better performance:
```bash
# NVIDIA GPUs
cargo build --release --features cuda

# Apple Silicon
cargo build --release --features metal
```

### 3. Pre-download Models

Download models when you have good internet:
```bash
aha download -m qwen3vl-2b
```

Then use them later without internet:
```bash
aha serv -m qwen3vl-2b --weight-path ~/.aha/Qwen/Qwen3-VL-2B-Instruct
```

### 4. Manage Disk Space

Models are stored in `~/.aha/` by default. Clean up if needed:
```bash
# Check disk usage
du -sh ~/.aha/*

# Remove old models
rm -rf ~/.aha/old-model-name
```

### 5. Monitor Resources

For large models, monitor your resources:
```bash
# Linux
htop
nvidia-smi  # For NVIDIA GPUs

# macOS
Activity Monitor
```

## Troubleshooting

### Port Already in Use

```bash
# Use a different port
aha cli -m qwen3-0.6b -p 8080
```

### Model Download Failed

```bash
# Retry with more attempts
aha download -m qwen3vl-2b --download-retries 5
```

### Out of Memory

```bash
# Use a smaller model
aha cli -m qwen3-0.6b
```

## Next Steps

1. Explore the [API Reference](./api.md) for detailed endpoint documentation
2. Read the [CLI Reference](./cli.md) for all command options
3. Check [Architecture & Design](./concepts.md) to understand how AHA works
4. See [Development](./development.md) if you want to contribute

## Examples Repository

For more examples, check out the [tests](../tests/) directory in the repository.

## See Also

- [API Reference](./api.md) - Complete API documentation
- [CLI Reference](./cli.md) - Command-line reference
- [Installation Guide](./installation.md) - Installation instructions
- [Development Guide](./development.md) - Contributing guide

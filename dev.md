# easy-asr-server 开发文档

## 1. 项目概述

easy-asr-server 是一个基于 FunASR 库构建的简易高并发语音识别服务包。它提供 REST API 接口 (`/asr/recognize`, `/asr/health`)，支持上传标准音频格式进行语音识别，并返回识别结果文本。该服务利用 FunASR 的 `AutoModel` 进行 VAD+ASR 处理，支持多用户并发请求 (通过 Uvicorn workers 实现)，并能自动利用可用的 GPU 或 CPU 资源。

支持通过命令行接口 (CLI) 启动和配置服务。

## 2. 包结构

```
src/
├── easy_asr_server/
│   ├── __init__.py           # 包初始化和版本信息
│   ├── api.py                # FastAPI 应用, REST API 路由, CLI 入口
│   ├── asr_engine.py         # ASR 处理引擎 (含设备检测, VAD+ASR pipeline)
│   ├── model_manager.py      # 模型下载和管理 (使用 modelscope)
│   └── utils.py              # 工具函数(音频处理、日志、错误处理)
├── pyproject.toml            # 项目配置和依赖 (使用 uv 管理, 含 CLI 入口)
└── README.md                 # 使用文档
```

## 3. 模块说明

### 3.1 模型管理模块 (model_manager.py)

**功能**：
- 负责 **默认模型 (ASR: `iic/SenseVoiceSmall`, VAD: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`)** 的下载、缓存和加载。
- 使用 `modelscope` 库提供的 `snapshot_download` API 确保模型文件的稳定获取和本地存储 (参见: https://www.modelscope.cn/docs/Models/Download-Model)。
- 提供线程安全的模型下载机制，防止多线程并发导致的重复下载。
- 实现模型本地缓存管理。
- 提供获取模型本地路径的功能。

**关键组件**：
- `ModelManager` 类：单例模式，确保全局只有一个模型管理实例。
- `download_model` 方法：线程安全的模型下载功能，使用 `modelscope.snapshot_download` 下载指定模型 ID。
- `get_model_path` 方法：获取指定模型 ID 的本地缓存路径。
- 文件锁机制：防止并发下载冲突。

### 3.2 ASR 引擎模块 (asr_engine.py)

**功能**：
- 封装 FunASR 的核心功能 (`AutoModel`)。
- **初始化 `AutoModel` 时，主要使用 ASR 模型 (`iic/SenseVoiceSmall`) 的本地路径，并依赖 FunASR 自动查找和使用同目录下由 `ModelManager` 提供的 VAD 模型 (`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`) 文件，以实现 VAD+ASR 处理流程。**
- 负责完整的 VAD 切分和语音识别。
- 自动检测并使用 GPU (CUDA) 如果可用，否则回退到 CPU。支持通过 CLI 参数手动指定设备。
- 提供简单的健康检查机制。

**关键组件**：
- `ASREngine` 类：封装 FunASR `AutoModel` 的使用，初始化时接收 ASR 模型路径和设备，并配置 `AutoModel` 以执行 VAD+ASR。
- `recognize` 方法：接收 `utils` 模块处理过的音频数据 (16kHz 单声道 WAV)，执行 VAD+ASR pipeline，并返回最终识别文本。
- `test_health` 方法：检查引擎是否正常运行 (例如，通过处理一段静音或预定义短音频，确认模型加载和基础推理正常)。

### 3.3 API 服务模块 (api.py)

**功能**：
- 提供 REST API 接口 (`/asr/recognize`, `/asr/health`)。
- 处理 HTTP 请求和响应。
- 管理服务生命周期 (启动/关闭事件)，包括 ASR 引擎的实例化和资源清理。
- 提供命令行接口 (CLI) 用于启动服务，使用 `typer` 实现。
- 处理 CLI 参数 (`host`, `port`, `workers`, `device`) 并传递给 Uvicorn 和 ASR 引擎。

**关键组件**：
- FastAPI 应用实例。
- `/asr/recognize` 端点：接收音频文件，调用 `utils` 进行验证/转换，然后调用 `ASREngine` 处理，并返回结果。
- `/asr/health` 端点：调用 `ASREngine` 的 `test_health` 方法并返回服务状态。
- `typer` 应用实例和命令 (`run`)：解析命令行参数。
- CLI 参数处理逻辑：将解析的参数传递给 `uvicorn.run` 和 ASR 引擎初始化。
- 启动和关闭事件处理：管理 ASR 引擎的初始化 (包括模型加载、设备选择) 和清理。

**3.3.1 Error Handling**

服务必须实现健壮的错误处理机制，并向客户端返回明确的 HTTP 状态码和错误信息：

*   **`400 Bad Request`**: 输入无效。原因可能包括：
    *   上传的不是有效的音频文件。
    *   音频文件格式无法通过 `utils` 模块成功转换 (例如，严重损坏或完全不支持的编解码器)。
*   **`422 Unprocessable Entity`**: 请求格式正确，但语义错误。主要用于 FastAPI 的自动验证失败（例如，缺少 `audio` 文件部分）。
*   **`500 Internal Server Error`**: 服务器内部发生意外错误。例如：
    *   ASR 引擎推理失败。
    *   模型加载失败 (发生在启动后)。
    *   其他未预料到的代码异常。
*   **`503 Service Unavailable`**: 服务暂时无法处理请求。例如：
    *   服务正在启动，模型尚未加载完成。
    *   ASR 引擎健康检查失败。

错误响应体应包含一个 `detail` 字段，提供关于错误的简要说明。

### 3.4 工具模块 (utils.py)

**功能**：
- 提供音频处理工具：
    *   **验证上传文件的类型是否为音频。**
    *   **将输入音频转换为 ASR 引擎所需的标准格式：16kHz 采样率, 16-bit PCM, 单声道 (mono) WAV 格式。** 使用 `torchaudio` 或类似库进行处理。
    *   处理转换过程中可能出现的错误。
- 日志记录设置 (配置标准日志格式和级别)。
- 定义通用的错误/异常类 (如果需要)。

**关键组件**：
- `process_audio` 函数：接收上传的文件对象，执行验证、转换，并返回处理后的音频数据 (例如，numpy array 或 bytes) 或引发异常。
- 日志配置函数。

### 3.5 包初始化 (__init__.py)

**功能**：
- 定义包的公共接口 (如果需要导出类或函数)。
- 提供版本信息 (`__version__`)。

## 4. 核心流程

### 4.1 服务启动流程 (CLI Only)

1.  用户在终端运行 `easy-asr-server run [OPTIONS]`。
2.  `typer` 解析命令行参数 (`host`, `port`, `workers`, `device`)。
3.  服务启动事件触发：
    a.  `ModelManager` 实例被创建。调用其方法确保 **ASR (`iic/SenseVoiceSmall`) 和 VAD (`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`) 模型均已下载** 到本地缓存，并获取 ASR 模型的路径。
    b.  根据 `device` 参数 (或自动检测) 确定计算设备。
    c.  `ASREngine` 实例被创建，传入 ASR 模型路径和设备，配置其内部 `AutoModel` 使用 VAD+ASR pipeline。
    d.  执行一次健康检查以确保引擎正常。
4.  FastAPI 应用完成初始化，配置日志。
5.  调用 `uvicorn.run` 启动服务器，传入 `host`, `port`, `workers` 和 FastAPI 应用实例。

### 4.2 请求处理流程 (`/asr/recognize`)

1.  客户端发送包含音频文件的 POST 请求到 `/asr/recognize`。
2.  FastAPI 接收请求，获取上传的文件。
3.  调用 `utils.process_audio` 处理上传的文件：
    *   验证文件类型。
    *   转换为 16kHz 单声道 WAV 格式 (内存中处理)。
    *   如果验证或转换失败，捕获异常并返回 `400 Bad Request`。
4.  获取处理后的音频数据。
5.  调用 `ASREngine` 实例的 `recognize` 方法处理音频数据。
    *   如果引擎内部出错，捕获异常并返回 `500 Internal Server Error`。
6.  返回包含识别文本的 JSON 响应 (e.g., `{"text": "..."}`), 状态码 `200 OK`。

### 4.3 模型下载流程

1.  服务启动时，`ModelManager` 检查默认的 **ASR (`iic/SenseVoiceSmall`) 和 VAD (`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`) 模型** 是否均已存在于本地缓存。
2.  对于任何不存在的模型，使用 `modelscope.snapshot_download` 库下载。
3.  使用文件锁确保只有一个进程/线程 (在多 worker 启动时) 对同一个模型执行下载。
4.  下载完成后，`ModelManager` 可以返回模型的本地路径供 `ASREngine` 使用。

## 5. 安装和使用

### 5.1 安装方式

```bash
# 推荐使用 uv 进行环境和依赖管理
# https://github.com/astral-sh/uv

# 创建虚拟环境 (可选但推荐)
uv venv

# 激活环境 (Linux/macOS)
source .venv/bin/activate
# 激活环境 (Windows)
# .venv\Scripts\activate

# 安装依赖 (包括 typer, modelscope, funasr, torch, torchaudio)
uv pip install -e .

# (pyproject.toml 应包含 typer 和脚本入口)
# [project]
# dependencies = [
#    "fastapi",
#    "uvicorn[standard]",
#    "funasr",
#    "modelscope",
#    "torch",
#    "torchaudio", # For audio processing in utils
#    "typer[all]",
#    "python-multipart", # For FastAPI file uploads
#    # ...其他依赖
# ]
#
# [project.scripts]
# easy-asr-server = "easy_asr_server.api:cli_app" # 假设 typer app 在 api.py 中叫 cli_app
```

### 5.2 基本使用 (CLI)

```bash
# 启动服务 (使用默认配置: host=127.0.0.1, port=8000, workers=1, device=auto)
easy-asr-server run

# 指定主机和端口
easy-asr-server run --host 0.0.0.0 --port 9000

# 指定 worker 数量 (用于 uvicorn)
easy-asr-server run --workers 4

# 手动指定使用 CPU
easy-asr-server run --device cpu

# 手动指定使用 GPU (如果可用)
easy-asr-server run --device cuda

# 组合选项
easy-asr-server run --host 0.0.0.0 --port 8080 --workers 2 --device auto
```

**CLI 选项:**

*   `--host TEXT`: 服务器监听的主机地址 [默认: "127.0.0.1"]
*   `--port INTEGER`: 服务器监听的端口 [默认: 8000]
*   `--workers INTEGER`: Uvicorn worker 进程数 [默认: 1]
*   `--device TEXT`: 指定计算设备 ('auto', 'cpu', 'cuda') [默认: "auto"]

### 5.3 API 使用示例

```python
import requests

# 发送音频文件识别请求 (服务会尝试转换格式)
with open("audio.wav", "rb") as f: # or audio.mp3, etc.
    audio_data = f.read()

# 假设服务运行在 127.0.0.1:8000
response = requests.post(
    "http://127.0.0.1:8000/asr/recognize",
    files={"audio": ("input_audio.wav", audio_data)} # 文件名随意, 服务内部处理
)

if response.status_code == 200:
    print(response.json()["text"])
else:
    print(f"Error: {response.status_code}")
    try:
        print(response.json()) # Print detailed error if available
    except requests.exceptions.JSONDecodeError:
        print(response.text) # Print raw text if not JSON

```

## 6. 配置

- **设备选择**: 通过 CLI 的 `--device` 参数 (`auto`, `cpu`, `cuda`) 控制。默认值为 `auto`。
- **模型选择**: 当前固定使用 ASR 模型 `iic/SenseVoiceSmall` 和 VAD 模型 `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`。未来可考虑增加 CLI 参数 `--asr-model-id` 和 `--vad-model-id` 进行配置。
- **并发处理**: 服务并发能力主要由 Uvicorn worker 进程数决定 (通过 `--workers` CLI 参数设置)。每个 worker 独立处理请求。对于 CPU 密集型的 ASR 任务，并发数受限于 CPU 核心数和 worker 数量。对于 GPU 加速，并发瓶颈可能在 GPU 内存或计算单元。**注意：简单的多 worker 模型可能导致每个 worker 都加载一份模型副本 (ASR+VAD)，需关注内存占用，特别是 GPU 内存。** （未来可考虑更高级的请求队列/共享模型实例方案以优化资源使用）。
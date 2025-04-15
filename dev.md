# easy-asr-server 开发文档

## 1. 项目概述

easy-asr-server 是一个基于 FunASR 库构建的简易高并发语音识别服务包。它提供 REST API 接口 (`/asr/recognize`, `/asr/health`)，支持上传标准音频格式进行语音识别，并返回识别结果文本。该服务利用 FunASR 的 `AutoModel` 进行 VAD+ASR+标点恢复处理，支持多用户并发请求 (通过 Uvicorn workers 实现)，并能自动利用可用的 GPU 或 CPU 资源。

支持通过命令行接口 (CLI) 启动和配置服务，可以选择不同的 ASR 核心 pipeline (例如 `sensevoice` 或 `paraformer`)。

## 2. 包结构

```
src/
├── easy_asr_server/
│   ├── __init__.py           # 包初始化和版本信息
│   ├── api.py                # FastAPI 应用, REST API 路由, CLI 入口 (使用 Typer)
│   ├── asr_engine.py         # ASR 处理引擎 (现在主要委托给 ModelManager)
│   ├── model_manager.py      # 模型下载、缓存、加载 (AutoModel) 和推理接口
│   └── utils.py              # 工具函数(音频处理、日志、错误处理)
├── pyproject.toml            # 项目配置和依赖 (使用 uv 管理, 含 CLI 入口)
└── README.md                 # 使用文档
```

## 3. 模块说明

### 3.1 模型管理模块 (model_manager.py)

**功能**：
- **管理多个 ASR pipelines (例如 `sensevoice`, `paraformer`) 的配置**，包括各组件 (ASR, VAD, PUNC) 的 ModelScope ID、推理参数 (`params`) 和后处理函数 (`postprocess`)，定义在模块级的 `MODEL_CONFIGS` 字典中。
- 负责 **下载、缓存** 指定 pipeline 所需的所有模型组件 (ASR, VAD, PUNC)，使用 `modelscope` 库提供的 `snapshot_download` API 确保模型文件的稳定获取和本地存储。
- 提供线程安全和进程安全的模型下载机制 (使用 `threading.Lock` 和 `filelock.FileLock`)，防止并发下载冲突。
- **在服务启动时 (worker 进程的 lifespan 启动阶段)，根据 CLI 参数选择的 pipeline 类型，使用各组件的本地缓存路径加载 `funasr.AutoModel` 实例。**
- 封装加载的 `AutoModel` 实例，并提供一个统一的 **`generate` 方法** 作为稳定的推理接口。
- `generate` 方法负责调用底层 `AutoModel` 的推理方法，并应用在 `MODEL_CONFIGS` 中为该 pipeline 配置的特定 **推理参数和后处理函数**，最终返回处理后的识别文本字符串。

**关键组件**：
- `ModelManager` 类：单例模式，确保全局只有一个实例管理模型状态。
- `MODEL_CONFIGS` 字典：定义不同 pipeline 的模型 ID、参数和后处理逻辑。
- `download_model` 方法：线程/进程安全的模型组件下载功能。
- `get_model_path` 方法：获取模型组件本地缓存路径 (按需触发下载)。
- `load_pipeline` 方法：核心方法，在服务启动时调用。负责确保所有组件下载完成，并使用本地路径初始化 `AutoModel` 实例。
- `generate` 方法：封装 `AutoModel` 推理，应用配置的参数和后处理，返回 `str` 结果。

### 3.2 ASR 引擎模块 (asr_engine.py)

**功能**：
- **接收一个已初始化的 `ModelManager` 实例**。
- **将核心的语音识别任务委托给 `ModelManager` 的 `generate` 方法**。
- 不再直接处理模型加载、路径管理或设备选择。
- 提供简单的健康检查 (`test_health`)，通过调用 `ModelManager` 的 `generate` 处理一段静音来验证底层 pipeline 是否正常工作。

**关键组件**：
- `ASREngine` 类：构造函数接收 `ModelManager` 实例。
- `recognize` 方法：接收 `utils` 模块处理过的音频数据，调用 `self.model_manager.generate()` 获取识别结果字符串。
- `test_health` 方法：调用 `self.recognize` (间接调用 `model_manager.generate`) 进行健康检查。

### 3.3 API 服务模块 (api.py)

**功能**：
- 提供 REST API 接口 (`/asr/recognize`, `/asr/health`)。
- 使用 `typer` 提供命令行接口 (CLI) 用于启动服务。
- 处理 CLI 参数 (`host`, `port`, `workers`, `device`, `pipeline`, `log_level`)。
- 使用 FastAPI 的 **`lifespan` 上下文管理器** 处理服务启动和关闭事件。
    - **启动时**: 在每个 Uvicorn worker 进程中，根据主进程传递的配置 (通过 `app_state`)：
        1.  获取 `ModelManager` 单例。
        2.  调用 `model_manager.load_pipeline()` 加载选定的 pipeline 和模型到指定设备。
        3.  实例化 `ASREngine` 并传入 `ModelManager` 实例。
        4.  将 `ModelManager` 和 `ASREngine` 实例存储在 `app_state` 中供后续请求使用。
    - **关闭时**: 清理 `app_state` 中的实例引用。
- 使用 **依赖注入 (`Depends`)** 将健康的 `ASREngine` 实例提供给 API 端点处理函数。
    - `get_asr_engine` 依赖项负责从 `app_state` 获取引擎实例，并调用其 `test_health` 方法确认服务可用性。
- 处理 HTTP 请求和响应，包括文件上传和 JSON 响应。

**关键组件**：
- FastAPI 应用实例 (`app`)，配置了 `lifespan`。
- `app_state` 字典：用于在主进程和 worker 进程间传递配置，并存储 worker 内的单例实例 (`ModelManager`, `ASREngine`)。
- `lifespan` 异步上下文管理器：处理 worker 进程的启动和关闭逻辑。
- `get_asr_engine` 依赖函数：提供健康的 `ASREngine` 实例给端点。
- `/asr/recognize` 端点：接收音频文件，调用 `utils.process_audio` 处理，调用 `ASREngine.recognize`，返回 `{"text": "..."}` JSON 响应。
- `/asr/health` 端点：通过 `get_asr_engine` 依赖确认健康状态，返回包含状态、pipeline 类型和设备的 JSON 响应。
- `typer` 应用实例 (`cli_app`) 和 `run` 命令：解析命令行参数。

**3.3.1 Error Handling**

服务必须实现健壮的错误处理机制，并向客户端返回明确的 HTTP 状态码和错误信息：

*   **`400 Bad Request`**: 输入无效。原因可能包括：
    *   上传的不是有效的音频文件。
    *   音频文件格式无法通过 `utils` 模块成功转换 (例如，严重损坏或完全不支持的编解码器)。
*   **`422 Unprocessable Entity`**: 请求格式正确，但语义错误。主要用于 FastAPI 的自动验证失败（例如，缺少 `audio` 文件部分）。
*   **`500 Internal Server Error`**: 服务器内部发生意外错误。例如：
    *   ASR 引擎推理失败 (从 `ModelManager.generate` 抛出的异常)。
    *   音频处理中发生未预料的错误。
    *   其他未预料到的代码异常。
*   **`503 Service Unavailable`**: 服务暂时无法处理请求。例如：
    *   服务正在启动，模型尚未加载完成 (依赖 `get_asr_engine` 失败)。
    *   ASR 引擎健康检查失败 (依赖 `get_asr_engine` 失败)。

错误响应体应包含一个 `detail` 字段，提供关于错误的简要说明。

### 3.4 工具模块 (utils.py)

**功能**：
- 提供音频处理工具：
    *   **验证上传文件的类型是否为音频 (`is_valid_audio_file`)。**
    *   **将输入音频转换为 ASR 引擎所需的标准格式（16kHz 采样率, 单声道 WAV），保存到临时文件，并返回文件路径。** 使用 `torchaudio` 进行转换和 `soundfile` (或类似库) 进行保存。
    *   处理转换和保存过程中可能出现的错误 (`AudioProcessingError`)。
- **提供读取热词文件的功能 (`read_hotwords`)，返回包含热词的字符串。**
- 日志记录设置 (`setup_logging`)。
- 获取音频时长 (`get_audio_duration`，从 waveform 计算，可能已移除或更改)。
- 保存音频 (`save_audio_to_file`，现在集成到 `process_audio` 中)。

**关键组件**：
- `process_audio` 函数：接收上传的文件对象 (`.file` 属性)，执行验证、转换和保存到临时 WAV 文件，**返回临时文件的路径**，或引发 `AudioProcessingError`。
- `read_hotwords` 函数：读取指定路径的热词文件，处理异常，返回格式化后的字符串。
- `is_valid_audio_file` 函数。
- `setup_logging` 函数。

### 3.5 包初始化 (__init__.py)

**功能**：
- 定义包的公共接口 (导出关键类和函数)。
- 提供版本信息 (`__version__`)。

## 4. 核心流程

### 4.1 服务启动流程 (CLI Only)

1.  用户在终端运行 `easy-asr-server run [OPTIONS]`。
2.  `typer` 解析命令行参数 (`host`, `port`, `workers`, `device`, `pipeline`, `log_level`)。
3.  **主进程将选择的 `pipeline` 和 `device` 存储到 `app_state` 字典中。**
4.  主进程配置并启动 `uvicorn`。
5.  Uvicorn 启动指定数量的 worker 进程。
6.  **每个 worker 进程独立执行 `lifespan` 启动事件：**
    a.  从 `app_state` 读取 `pipeline_type` 和 `device` 配置。
    b.  获取 `ModelManager` 单例实例。
    c.  **调用 `model_manager.load_pipeline(pipeline_type, device)`:**
        i.  该方法内部根据 `pipeline_type` 查找 `MODEL_CONFIGS`。
        ii. **为 pipeline 所需的每个组件 (ASR, VAD, PUNC) 调用 `model_manager.get_model_path(model_id)`，该方法会按需调用 `model_manager.download_model` (使用 `snapshot_download`) 来下载并缓存模型文件，最终返回本地路径。**
        iii. 使用获取到的 **本地模型文件路径** 初始化 `funasr.AutoModel` 实例。
        iv. 将加载的 `AutoModel` 存储在 `ModelManager` 实例内部。
    d.  实例化 `ASREngine`，将 `ModelManager` 实例传入其构造函数。
    e.  将 `ModelManager` 和 `ASREngine` 实例存储在 `app_state` 中 (供该 worker 进程内的请求使用)。
    f.  执行一次健康检查以确保引擎正常 (通过 `ASREngine.test_health`)。
7.  Worker 进程初始化完成，FastAPI 应用开始监听请求。

### 4.2 请求处理流程 (`/asr/recognize`)

1.  客户端发送包含音频文件的 POST 请求到 `/asr/recognize`。
2.  FastAPI 接收请求，获取 `UploadFile` 对象。
3.  **FastAPI 依赖注入系统调用 `get_asr_engine` 依赖:**
    a.  `get_asr_engine` 从 `app_state` 获取 `ASREngine` 实例。
    b.  如果实例不存在 (启动失败)，抛出 503 异常。
    c.  调用 `asr_engine_instance.test_health()` 进行健康检查。
    d.  如果健康检查失败，抛出 503 异常。
    e.  返回健康的 `ASREngine` 实例 (`engine`)。
4.  `/asr/recognize` 端点函数执行：
    a.  调用 `utils.process_audio(audio.file)` 处理上传的文件：
        *   验证文件类型 (`is_valid_audio_file`)。
        *   加载、转换为 16kHz 单声道 `torch.Tensor`。
        *   如果验证或转换失败，捕获 `AudioProcessingError` 并返回 `400 Bad Request`。
    b.  获取处理后的音频数据 (`audio_data`)。
    c.  调用注入的 `engine.recognize(audio_data)`。
        *   **内部调用 `self.model_manager.generate(input_audio=audio_data)`:**
            *   `ModelManager.generate` 调用内部加载的 `AutoModel` 实例的推理方法，并传入 `MODEL_CONFIGS` 中配置的 `params`。
            *   对 `AutoModel` 的原始输出应用 `MODEL_CONFIGS` 中配置的 `postprocess` 函数。
            *   返回最终的文本字符串。
    d.  如果引擎内部出错 (例如 `ModelManager.generate` 抛出异常)，捕获 `RuntimeError` 并返回 `500 Internal Server Error`。
5.  返回包含识别文本的 JSON 响应 `{"text": "..."}`，状态码 `200 OK`。

### 4.3 模型下载流程

1.  **服务启动时，在每个 worker 的 `lifespan` 启动阶段，`ModelManager.load_pipeline` 被调用。**
2.  `load_pipeline` 根据选择的 `pipeline` 类型，查找 `MODEL_CONFIGS` 中定义的 ASR, VAD, PUNC 组件的 ModelScope ID。
3.  **对于每个组件 ID，调用 `ModelManager.get_model_path`。**
4.  `get_model_path` 检查该 ID 是否已有缓存路径。如果没有，则调用 `ModelManager.download_model`。
5.  `download_model` 使用线程锁和文件锁确保只有一个线程/进程对同一个模型 ID 执行下载。
6.  `download_model` 调用 `modelscope.snapshot_download` 下载模型文件到缓存目录 (`~/.cache/easy_asr_server/models`)。
7.  下载完成后，创建 `download_complete` 标记文件，存储实际下载路径。
8.  `get_model_path` 返回模型的本地路径。
9.  `load_pipeline` 使用这些路径初始化 `AutoModel`。

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
# -e . 表示以可编辑模式安装当前项目
uv pip install -e '.[dev]' # 安装核心依赖和开发依赖 (dev 包含 pytest-cov等)
# 或者仅安装核心依赖:
# uv pip install -e .

# (pyproject.toml 应包含 typer 和脚本入口)
# [project]
# dependencies = [
#    "fastapi",
#    "uvicorn[standard]",
#    "funasr",
#    "modelscope",
#    "torch",
#    "torchaudio", # For audio processing in utils
#    "typer[all]", # Changed from click
#    "python-multipart", # For FastAPI file uploads
#    "filelock", # For model download locking
#    # ...其他依赖
# ]
#
# [project.scripts]
# easy-asr-server = "easy_asr_server.api:cli_app" # Updated entry point for Typer
```

### 5.2 基本使用 (CLI)

```bash
# 启动服务 (使用默认配置: host=127.0.0.1, port=8000, workers=1, device=auto, pipeline=sensevoice, log_level=info)
# 由于只有一个命令，直接执行脚本名加选项即可
easy-asr-server

# 指定主机和端口
easy-asr-server --host 0.0.0.0 --port 9000

# 指定 worker 数量 (用于 uvicorn)
easy-asr-server --workers 4

# 手动指定使用 CPU
easy-asr-server --device cpu

# 手动指定使用 GPU (如果可用)
easy-asr-server --device cuda

# 选择不同的 ASR pipeline
easy-asr-server --pipeline paraformer

# 设置日志级别
easy-asr-server --log-level debug

# 指定热词文件
easy-asr-server --hotword-file /path/to/hotwords.txt

# 组合选项
easy-asr-server --host 0.0.0.0 --port 8080 --workers 2 --pipeline sensevoice --device auto --log-level info
```

**CLI 选项 (使用 Typer):**

这些是启动服务器主命令的可用选项：

*   `--host TEXT`: 服务器监听的主机地址 [默认: "127.0.0.1"]
*   `--port INTEGER`: 服务器监听的端口 [默认: 8000]
*   `--workers INTEGER`: Uvicorn worker 进程数 [默认: 1]
*   `--device TEXT`: 指定计算设备 ('auto', 'cpu', 'cuda', 'cuda:0', etc.) [默认: "auto"]
*   `--pipeline TEXT`: 指定要使用的 ASR pipeline 类型 (例如 'sensevoice', 'paraformer') [默认: "sensevoice"]
*   `--log-level TEXT`: 指定日志级别 ('debug', 'info', 'warning', 'error') [默认: "info"]
*   `--hotword-file`, `-hf TEXT`: 指定热词文件路径 (一行一个热词) [默认: None]

### 5.3 API 使用示例

```python
import requests

# Example for /asr/recognize
with open("audio.wav", "rb") as f:
    audio_data = f.read()

response = requests.post(
    "http://127.0.0.1:8000/asr/recognize",
    files={"audio": ("input_audio.wav", audio_data)}
)

if response.status_code == 200:
    print(response.json()["text"])
else:
    print(f"Error: {response.status_code}")
    print(response.json())

# Example for GET /asr/health
response = requests.get("http://127.0.0.1:8000/asr/health")
if response.status_code == 200:
    print(f"Health check: {response.json()}")
else:
    print(f"Health check failed: {response.status_code} - {response.text}")

# Example for GET /asr/hotwords
response = requests.get("http://127.0.0.1:8000/asr/hotwords")
if response.status_code == 200:
    print(f"Current hotwords: {response.json()}")
else:
    print(f"Failed to get hotwords: {response.status_code} - {response.text}")

# Example for PUT /asr/hotwords
new_list = ["OpenAI", "ChatGPT", "FunASR"]
response = requests.put("http://127.0.0.1:8000/asr/hotwords", json=new_list)
if response.status_code == 204:
    print("Hotwords updated successfully.")
else:
    print(f"Failed to update hotwords: {response.status_code} - {response.text}")
```
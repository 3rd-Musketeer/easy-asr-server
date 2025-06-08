# Live ASR Demo

交互式语音识别演示程序，支持实时录音和识别。

## 功能特性

- 🎤 实时录音：按 Enter 开始/结束录音
- 🤖 多种模型：支持 `sensevoice` 和 `paraformer` 管道
- 🔤 热词支持：可配置热词提升识别准确率
- 🖥️ 设备选择：支持 CPU、GPU 等不同设备
- 🚀 直接调用：无需启动服务器，直接在 Python 中使用

## 依赖安装

```bash
# 安装音频录制库
pip install sounddevice

# 如果还没有安装 easy_asr_server 的其他依赖
pip install -r ../requirements.txt
```

## 使用方法

### 基本用法

```bash
# 使用默认设置（sensevoice 模型，自动设备检测）
python live_asr_demo.py

# 或者直接执行（如果有执行权限）
./live_asr_demo.py
```

### 高级配置

```bash
# 指定不同的模型
python live_asr_demo.py --pipeline paraformer

# 指定设备
python live_asr_demo.py --device cpu
python live_asr_demo.py --device cuda

# 添加热词（用空格分隔）
python live_asr_demo.py --hotwords "你好 谢谢 再见"

# 组合使用
python live_asr_demo.py --pipeline sensevoice --device cuda --hotwords "智能助手 语音识别"

# 开启详细日志
python live_asr_demo.py --log-level INFO
```

### 参数说明

- `--pipeline, -p`: ASR 管道类型
  - `sensevoice` (默认): SenseVoice 模型，支持多语言
  - `paraformer`: Paraformer 模型，中文识别
  
- `--device, -d`: 推理设备 (默认: auto)
  - `auto`: 自动检测最佳设备
  - `cpu`: 使用 CPU
  - `cuda`: 使用 GPU (需要CUDA支持)
  - `mps`: 使用 Apple Silicon GPU
  
- `--hotwords, -w`: 热词字符串，用空格分隔
  
- `--log-level, -l`: 日志级别 (DEBUG/INFO/WARNING/ERROR)

## 交互说明

运行程序后：

1. **🔴 录音**：在 `[Ready]` 状态下按 Enter 开始录音
2. **⏹️ 停止**：在 `[Recording]` 状态下按 Enter 停止录音并识别
3. **📝 结果**：识别结果会立即显示
4. **🔤 热词**：输入 `h` 查看当前热词配置
5. **👋 退出**：输入 `q` 退出程序

## 示例会话

```
🎤 Live ASR Demo
Pipeline: sensevoice
Device: auto
Hotwords: '你好 智能助手' (empty if none)
--------------------------------------------------
🔄 Initializing ASR components...
📱 Using device: cuda
📦 Loading pipeline: sensevoice
🏥 Performing health check...
✅ ASR engine is healthy and ready!

📋 Instructions:
• Press Enter to start recording
• Press Enter again to stop recording and get transcription
• Type 'q' and press Enter to quit
• Type 'h' and press Enter to show hotwords
--------------------------------------------------
⚪ [Ready] Press Enter to record (q=quit, h=hotwords): 
🔴 Recording started... Press Enter to stop
🔴 [Recording] Press Enter to stop: 
⏹️  Recording stopped. Duration: 3.24s, Samples: 51840
🤔 Recognizing speech...
⚡ Recognition completed in 0.89s
📝 Result: 你好，我想测试一下语音识别功能。
--------------------------------------------------
⚪ [Ready] Press Enter to record (q=quit, h=hotwords): q
👋 Goodbye!
```

## 注意事项

1. **首次运行**：第一次使用会自动下载模型，可能需要较长时间
2. **音频设备**：确保系统有可用的音频输入设备（麦克风）
3. **权限**：某些系统可能需要麦克风权限
4. **网络**：模型下载需要网络连接
5. **内存**：模型加载需要一定的内存空间（建议 4GB+ RAM）

## 故障排除

### 音频录制问题
```bash
# 检查音频设备
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### 模型下载问题
```bash
# 预先下载模型
python -m easy_asr_server.cli download sensevoice
```

### CUDA 支持问题
```bash
# 检查 CUDA 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
``` 
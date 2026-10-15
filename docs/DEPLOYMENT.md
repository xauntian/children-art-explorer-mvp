# 本地部署

本文以 Windows PowerShell、Python 3.12 为基准。开发机器使用 RTX 5080 16 GB 和约 96 GB 内存；这是已有环境，不是最低配置承诺。其他硬件需要重新检查显存、耗时与稳定性。

项目后台与 ComfyUI 可以使用不同的 Python 环境。后台只通过本地 HTTP 接口调用服务，不需要将 PyTorch 安装进项目环境。

## 1. 项目后台

在仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.main doctor --offline
.\.venv\Scripts\python.exe -m pytest tests/unit
```

预期启用 `fauvism`、`cubism`、`impressionism`，最大尝试数 2，打印关闭。`--offline` 不检查服务在线状态，也不表示模型已安装。

## 2. ComfyUI 与 GGUF 节点

在自行选择的运行时父目录执行以下命令。将大型模型保存在适合的位置，避免上传进源码仓库。

```powershell
git clone https://github.com/Comfy-Org/ComfyUI.git
cd ComfyUI
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/city96/ComfyUI-GGUF.git custom_nodes/ComfyUI-GGUF
.\.venv\Scripts\python.exe -m pip install -r custom_nodes/ComfyUI-GGUF/requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDA 12.8 构建用于已有 RTX 50 系环境；其他设备按 PyTorch 官方支持情况选择。GGUF 节点的依赖需完整安装，包括其声明的 `gguf`、`sentencepiece` 和 `protobuf`。已有试作使用 ComfyUI 0.39.0；上游版本更新后先运行 `doctor` 并重新验证。

## 3. 模型文件

从以下来源下载，并按表中的文件名放置；仓库不包含权重。下载前阅读对应来源的许可和使用条件。

| 用途 | ComfyUI 下的目录与文件名 | 来源 |
|---|---|---|
| 扩散模型 | `models/unet/Qwen-Image-Edit-2509-Q3_K_M.gguf` | [QuantStack](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF) |
| 编码器 | `models/text_encoders/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | [Unsloth](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) |
| 视觉投影 | `models/text_encoders/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf` | 同上，源文件 `mmproj-BF16.gguf` 需重命名 |
| VAE | `models/vae/qwen_image_vae.safetensors` | [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae) |
| 默认加速 LoRA | `models/loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors` | [Lightx2v](https://huggingface.co/lightx2v/Qwen-Image-Lightning/tree/main/Qwen-Image-Edit-2509) |

`mmproj` 必须与编码器同目录，文件名包含编码器基础名称 `Qwen2.5-VL-7B-Instruct`；仅保存为上游的 `mmproj-BF16.gguf` 可能导致节点找不到投影文件。

默认路线是 Qwen GGUF。`workflows/image_edit.json` 是兼容的 SD 1.5 工作流；其 checkpoint 字段不代表默认还需要下载 SD 1.5 或 ControlNet。

在 ComfyUI 目录启动服务，终端保持运行：

```powershell
.\.venv\Scripts\python.exe main.py --listen 127.0.0.1 --port 8188 --reserve-vram 1.0
```

## 4. Ollama 检查模型

安装 [Ollama](https://ollama.com/)，下载准确模型标签：

```powershell
ollama pull qwen3-vl:8b-instruct
ollama list
```

若桌面服务尚未运行，在单独终端执行 `ollama serve`。项目默认地址为 `http://127.0.0.1:11434`。

当前 `configs/app.json` 使用 `mode="comparison"`，`temperature=0`、`seed=0`、`num_predict=768`、`num_ctx=8192`、`num_gpu=0`。检查放在 CPU 上执行，以减少与生成模型争用显存；改变模型或设备策略后需重新验证，不能自动换成云端服务。

## 5. 验证并运行

返回项目仓库根目录，在后台项目环境中执行：

```powershell
.\.venv\Scripts\python.exe -m app.main doctor
.\.venv\Scripts\python.exe -m app.main import --photo samples/children_originals/01_garden.png --artwork
```

`doctor` 应能连接两个服务并找到配置中的模型。导入后每个任务写入 `runs/<job_id>/`，只在粗检通过且文件保存成功时创建 `output.png`。如果失败，请查看任务原因与检查记录；不要把存在候选图片当作任务成功。

默认应用工作流的 `FluxKontextImageScale` 会按预设比例缩放并可能中心裁切，`--artwork` 只关闭前处理裁边。要复现完整画面的展示结果，请使用 [examples 的独立配方](../examples/README.md)，它同时关闭了模型侧裁切。

## 6. 常见问题

| 情况 | 处理 |
|---|---|
| 服务连接失败 | 核对两个服务是否已启动、端口与 `configs/app.json` 地址是否一致 |
| 找不到 UnetLoaderGGUF / CLIPLoaderGGUF | 检查 GGUF 自定义节点及完整依赖，重启 ComfyUI |
| 找不到 mmproj | 核对文件名与编码器名称的对应关系，以及是否同目录 |
| 显存不足 | 检查其他 GPU 程序；保持检查器在 CPU，按实际设备调整输入尺寸与 ComfyUI 显存余量后再验证 |
| 响应截断或 JSON 类型错误 | 该任务为技术失败；检查原始响应、准确模型标签和长度上限，不放宽通过规则 |
| 候选有图但无 output.png | 可能未通过粗检；查看 `meta.json` 与 `check_*.json` |
| 已有活动任务 | 使用 `list --all` 查看，再用 `resume` 按已保存状态恢复 |
| 相机打不开 | 使用 `doctor --camera` 探测；检查设备索引与系统权限。默认 `dshow` 用于 Windows，其他平台改为受支持的后端 |

保持模型与服务版本、实际 Prompt、种子及参数记录完整。检查耗时和缓存命中不能混作重新生成耗时；不同环境不承诺逐像素复现。

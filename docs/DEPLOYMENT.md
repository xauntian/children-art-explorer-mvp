# Windows 本地部署

以下命令从仓库根目录的 PowerShell 执行，基准为 Python 3.12。已验证设备为 RTX 5080 16 GB、约 96 GB 内存；这不是最低配置要求，也不是跨设备的速度承诺。当前流程为 Windows GUI、华为原生相机截图、FLUX.2 Klein 4B 生成，风格审查与实体打印关闭。

项目 Python 与 ComfyUI Python 分开管理。项目通过本地 HTTP 调用生成服务，不需要在项目 `.venv` 中安装 PyTorch；生产不依赖 Ollama。

## 1. 项目环境

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.main doctor --offline
.\.venv\Scripts\python.exe -m pytest tests/unit
```

离线检查应识别全部八个启用画派、FLUX.2 工作流、10 秒 AI 预算和关闭的审查/打印。它不检查手机或模型服务在线状态，也不表示权重已安装。GUI 依赖 Tkinter、Pillow；屏幕视频解码依赖 `av`（PyAV），已列入项目依赖。

## 2. ComfyUI 环境

GUI 默认查找仓库内的 `comfyui/` 和 `comfyui/.venv/Scripts/python.exe`。该目录由部署者自行安装，Git 忽略其中的程序环境与权重；亦可修改 `configs/gui.json` 中的相对路径。

```powershell
git clone https://github.com/Comfy-Org/ComfyUI.git comfyui
py -3.12 -m venv comfyui/.venv
.\comfyui\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu130
.\comfyui\.venv\Scripts\python.exe -m pip install -r comfyui/requirements.txt
.\comfyui\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

已有实测为 ComfyUI 0.39.0、Torch 2.11.0+cu130、CUDA 可用。上游安装版本可能变化，安装后须核对节点与实际设备。RTX 50 系需要支持该 GPU 的 PyTorch 构建；其他设备按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择。生产 FLUX.2 工作流不要求 ComfyUI-GGUF 自定义节点；历史 Qwen GGUF 配方的依赖另计。

## 3. 三份生产模型

仓库不包含模型权重。阅读模型来源的许可和使用条件，下载后放在以下路径，保持文件名一致：

| 组件 | 仓库根目录下的目标路径 | 官方来源 |
|---|---|---|
| 扩散模型 | `comfyui/models/diffusion_models/flux-2-klein-4b-fp8.safetensors` | [black-forest-labs/FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) |
| 文本编码器 | `comfyui/models/text_encoders/qwen_3_4b_fp4_flux2.safetensors` | [Comfy-Org/Flux-2-Klein-4B · text_encoders](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/tree/main/split_files/text_encoders) |
| VAE | `comfyui/models/vae/flux2-vae.safetensors` | [Comfy-Org/Flux-2-Klein-4B · vae](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/tree/main/split_files/vae) |

八种画派只切换 Prompt，使用相同权重与采样配置，不需要下载八套模型，也不需要下载训练 LoRA。`configs/app.json` 中保留的其他路线字段不代表当前还需安装 SD 1.5、ControlNet、Qwen 图像编辑模型或检查模型。实际生产工作流为 `workflows/image_edit_flux2_klein.json`。

如需手动启动服务，从仓库根目录执行：

```powershell
.\comfyui\.venv\Scripts\python.exe comfyui/main.py --listen 127.0.0.1 --port 8189 --reserve-vram 1.0 --disable-auto-launch
```

GUI 可以复用这个服务，也可在「开始」时按 `configs/gui.json` 启动。服务地址须与 `configs/app.json` 的 `comfy.base_url` 一致。GUI 关闭后保留 ComfyUI，方便下次复用缓存。系统内存用于缓存和卸载，不能等同于 GPU 显存带宽；不要把 96 GB 内存理解为一定能达到 10 秒成品。

## 4. Android 工具与手机

下载并解压 [Google 官方 SDK Platform-Tools](https://developer.android.com/tools/releases/platform-tools)，使 ADB 位于：

```text
tools/android/platform-tools/adb.exe
```

从 [Genymobile 官方 scrcpy 5.0.1 发布页](https://github.com/Genymobile/scrcpy/releases/tag/v5.0.1)取得 `scrcpy-server-v5.0.1`，保持文件名不变，放在：

```text
tools/android/scrcpy-server-v5.0.1
```

ADB 所需的同包 DLL 应与可执行文件一同保留。仓库不打包第三方二进制，不要求安装桌面 scrcpy 客户端；项目自行调用官方 server 与 PyAV 解码。server 版本须与当前协议实现一致。

手机开启 USB 调试、USB 连接电脑并确认授权，然后检查：

```powershell
.\tools\android\platform-tools\adb.exe devices -l
```

预期仅有一台处于 `device` 状态的手机。默认 `phone.serial=null`，使用该设备；连接多台时须在 `configs/phone_capture_adb.json` 中显式指定序列号。不要把个人序列号提交到公共仓库。

当前 `preview_screen_size` 和 `preview_roi` 已针对 Mate 40 Pro 的原生相机标定。使用不同分辨率、显示设置或设备时，须以真实截图核对并调整 ROI；程序不会自动猜测取景框。GUI 不读倍率，也不打开、切换或调整相机。解锁手机，手动进入华为原生相机并设好取景，使四条纸边及少量底板完整可见；保持屏幕亮起。

GUI 待机预览是屏幕视频，不是周期截图；P 使用独立 ADB 原生截图保存正式输入。GUI 强制禁用拍照回退，不会因截图失败触发快门。Camera2 命令行入口是另一条显式采集路线，详见[手机采集规格](phone-capture-spec.md)。

## 5. 启动 GUI

```powershell
.\.venv\Scripts\python.exe -m app.main doctor
.\start_gui.cmd
```

`doctor` 在服务已启动时应找到当前模型及节点；无需启动 Ollama。点击 GUI 的「开始」后，程序检查手机和服务，逐一预热八派，准备完成后才接受 GUI 窗口中的 P。

每轮显示原图、校正图与 AI 图。处理中 P 失效，完成或失败后重新激活。单轮 AI 预算为 10 秒，采集、纸面校正和预热另计。超时只在 GUI 提示，不生成带标注或原图替代的假成品。原图和校正图可在 `runs/phone/<session>/` 查看，生成记录在 `runs/phone-<session>/meta.json`，GUI 日志在 `runs/runtime/gui.log`。

当前打印关闭，会跳过设备检查、系统打印与出纸等待。要启用实体打印，先安装 Windows 打印机驱动，再配置 `configs/gui.json` 的 `printer.enabled=true` 与可选 `printer.name`，重启 GUI。此功能仍需真实设备联调；故障时查询原作业，不反复提交或重新生成图片。

## 6. 检查与故障处理

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit
.\.venv\Scripts\python.exe -m ruff check app tools tests conftest.py
```

| 情况 | 核对内容 |
|---|---|
| 找不到项目 Python | 先创建 `.venv` 并安装 `requirements.txt` |
| GUI 无法启动 ComfyUI | `configs/gui.json` 的 Python/目录相对仓库根目录存在，且 ComfyUI 环境完整 |
| 服务连接失败 | 地址为 `127.0.0.1:8189`，配置一致；查看本机服务日志 |
| 缺 FLUX.2 节点或模型 | 更新兼容的 ComfyUI、核对三份文件的目录与名称，再运行 `doctor` |
| 找不到 ADB 或 server | 核对第四节的路径、文件名及同包 DLL；server 为 5.0.1 |
| 手机 `unauthorized` / 锁屏 / 不在相机 | 手机手动授权、解锁并打开原生相机，再重新准备 |
| 截图尺寸不匹配 | 核对实际屏幕尺寸与标定 ROI，不自动套用其他分辨率 |
| 检测不到完整纸边 | 保留四边及少量底板，避免纸边遮挡、反光或背景混淆，再按 P |
| AI 超时 | 本轮结束，不输出迟到结果；检查服务队列、GPU 占用与模型缓存，再开始新一轮 |
| 有原图但无最终图 | 查看会话与任务的错误；候选存在不代表成功，不手动伪造成功状态 |
| 预览静止 | 屏幕画面静止时可能不产生新帧；确认手机亮屏、相机前景与连接状态 |

依赖和权重就绪后的断网流程、GUI 连续多件、长时间运行与真实打印尚需现场验证。单元测试、预热或少量真实调用不能替代独立画质与儿童体验评估。

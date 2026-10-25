# Children’s Art Explorer

面向儿童艺术展览的本地生成原型：**实体绘画 → 手机采集 → 整纸透视校正 → 随机画派 → 本地 AI 重绘 → 保存 A4 比例图片**。儿童使用实体材料，工作人员通过 Windows 桌面 GUI 操作后台。

当前应用配置为 `1.10.0`，生成 Prompt 为 `1.7.0`。生产流程采用 **FLUX.2 Klein 4B、八画派、单轮 AI 10 秒截止预算**；风格审查和实体打印均关闭。八种风格共用同一套模型权重，各自使用画派 Prompt，项目没有训练八套模型或画派 LoRA。

## 文档入口

| 文档 | 用途 |
|---|---|
| [MVP_SPEC.md](MVP_SPEC.md) | 当前功能、设备约束、模型、输出和超时规则 |
| [部署说明](docs/DEPLOYMENT.md) | Windows、ComfyUI、模型和手机工具安装 |
| [手机采集规格](docs/phone-capture-spec.md) | 原生相机截图与 Camera2 命令行入口的区别 |
| [验证记录](docs/VALIDATION.md) | 真实测量、历史证据及待验收内容 |
| [agent.md](agent.md) · [AGENTS.md](AGENTS.md) | 开发约定与 Agent 入口 |
| [SPEC.md](SPEC.md) · [MVP_V1.0.md](MVP_V1.0.md) | 当前规格入口与项目研究背景 |
| [历史样例与配方](examples/README.md) | 三张输入、八画派二十四张历史 Qwen 结果及认可状态 |

## 启动与操作

在 Windows PowerShell 中准备项目环境：

```powershell
git clone https://github.com/xauntian/children-art-explorer-mvp.git
cd children-art-explorer-mvp
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.main doctor --offline
```

按[部署说明](docs/DEPLOYMENT.md)另行安装 ComfyUI、三份模型权重、ADB 和官方 scrcpy server。项目环境不需要安装 PyTorch，生产流程不需要 Ollama。仓库没有打包这些环境、权重或工具二进制。

双击 `start_gui.cmd`，或在仓库根目录运行：

```powershell
.\start_gui.cmd
```

1. USB 连接并授权手机调试，解锁手机，手动打开华为原生相机，设好位置、方向和倍率，使完整纸边可见。当前取景框配置针对已标定的 Mate 40 Pro；其他手机须重新核对屏幕尺寸与 ROI。
2. 点击 GUI 的「开始」。程序检查手机、服务与模型，复用已运行的 ComfyUI，或按配置在后台启动；逐一预热全部八个启用画派后，P 才激活。
3. 待机时第一栏显示手机屏幕实时视频。在 GUI 窗口按 **P**，执行原始截图、整纸检测与透视校正、随机画派、AI 生成和保存。界面显示步骤与原图、校正图、AI 结果。
4. 处理中忽略 P，不排队、不重复触发。本轮完成或失败后，P 重新激活；待机时恢复实时视频，可换画后再按 P。

GUI 不读取倍率、不定时截图、不调整相机设置，也不触发相机快门。预览采用独立的屏幕 H.264 视频，默认长边 1280、最高 15 fps、4 Mbps，关闭音频与远程控制。正式原图来自 P 触发的原生屏幕截图，不用低分辨率视频帧代替。GUI 截图路径不调用 Camera2、不自动切换拍照路径；尺寸不符、锁屏或纸边不完整时提示本轮失败。

## 当前生成配置

| 项目 | 当前值 |
|---|---|
| 生成服务 | 本地 ComfyUI，`http://127.0.0.1:8189` |
| 扩散模型 | `flux-2-klein-4b-fp8.safetensors` |
| 编码器 / VAE | `qwen_3_4b_fp4_flux2.safetensors` / `flux2-vae.safetensors` |
| 工作流 | `workflows/image_edit_flux2_klein.json`；原图经 VAE 与 ReferenceLatent 提供图像条件 |
| 采样 | 4 steps、CFG 1、Euler、Flux2Scheduler、denoise 1 |
| 分配 | 八个启用画派等概率随机，单轮选择一派并生成一次 |
| A4 原生生成 | 横向 1088×768；竖向 768×1088 |
| 最终图片 | 横向 1018×720；竖向 720×1018，Lanczos4 下采样 |
| 检查 / 打印 | `checker.enabled=false`；GUI `printer.enabled=false` |

八种画派为：野兽派、立体主义、印象派、后印象派、表现主义、超现实主义、新造型主义、波普艺术。原作提供主题或可辨认线索；Prompt 总结画派语言，允许重组构图，不插入逐图识别生成的场景描述。4B 指扩散模型参数规模，独立编码器另计；FP8 和 FP4 是量化格式。

**10 秒预算覆盖 AI 阶段的准备、上传、推理、读取、完整性检查、下采样与保存；手机采集、纸面校正和启动预热另计。** 到达截止时间仍无完整 AI 成品时，结束本轮并仅在 GUI 显示超时提示，保留原图和校正图，不用原图冒充 AI 结果、不添加图片标注，P 随后重新激活。程序只取消自身请求并拒收迟到结果；预算用于结束等待，不保证每次都能在 10 秒内生成成品。

生产流程只检查图片可解码、尺寸等文件完整性，不执行主题或画派审查；记录 `review_status=not_performed`、`validation_scope=file_integrity_only`。存在最终图不表示通过独立画质验收。

## 打印与文件

当前跳过实体打印机检查、提交与出纸等待，无需连接打印机。需要接入时在 `configs/gui.json` 显式启用 `printer.enabled`，安装 Windows 驱动并配置实体 A4 打印机，再进行真实设备联调。启用后，P 要等驱动确认已打印或工作人员确认出纸才重新激活；故障不自动重印。打印关闭时不改写此前的待打印记录。

```text
app/                    GUI、采集、纸边提取、生成、截止预算、状态与持久化
configs/                当前应用、八画派、手机和 GUI 配置及备用配方
workflows/              当前 FLUX.2 与保留模型的 ComfyUI API 工作流
prompts/                生成模板及流程外检查模板
tools/                  采集、生成验证与校准工具；Android 二进制需另装
tests/                  单元测试、集成测试及固定回归素材
examples/               保留的历史 Qwen 样图、参数、Prompt 与哈希
docs/                   部署、规格和选定验证记录
runs/phone/<session>/   原始屏幕、取景框、校正图与会话记录（本机生成）
runs/phone-<session>/   AI 候选、最终 output.png 与 meta.json（本机生成）
runs/runtime/           本机日志与预热记录，不提交
```

配置使用仓库相对路径，手机序列号默认不固定。模型、环境、运行日志、密钥和个人机器路径不进入源码提交；已有公开样例按原字节保留。独立的 Camera2 命令行入口和导入入口仍保留，其取景策略与 GUI 不同，见[手机采集规格](docs/phone-capture-spec.md)。例如导入完整画作：

```powershell
.\.venv\Scripts\python.exe -m app.main doctor
.\.venv\Scripts\python.exe -m app.main import --photo samples/children_originals/01_garden.png --artwork
```

`--artwork` 用于已经裁好的原作；需要重新处理时建立新任务，不覆盖历史结果。

## 验证范围

本机 RTX 5080 16 GB、约 96 GB 内存、ComfyUI 0.39.0 / Torch 2.11.0+cu130 环境中，FLUX.2 对既有校正图的一次完整生成管线耗时 **3.89 秒**。该记录产生于三画派配置阶段，不是当前八派全部真实输入测速，也不包含手机采集或完整冷启动。八派预热另有 **22.792 秒**完成记录，输入是合成纯色图，不能作为八派画质或全流程速度证据。详情见[验证记录](docs/VALIDATION.md)。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit
.\.venv\Scripts\python.exe -m ruff check app tools tests conftest.py
```

单元测试包含模拟服务与设备；只有明确需要真实服务时才运行 `python -m pytest tests/integration -v -m integration`。工程检查不能替代八派独立质量、GUI 连续多件、断网、实体打印和儿童体验验收。历史样例使用不同的 Qwen 配方，不作为当前 FLUX.2 画质或速度的证明。

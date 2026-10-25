# 手机采集与作品提取规格

## 1. 目标与设备

桌面 GUI 使用华为原生相机的实时屏幕视频预览，P 独立截取当前已标定的取景框，再完成整纸校正、本地生成和 A4 数字保存。工作人员需手动解锁手机、打开原生相机并设好取景；软件不读取或调整倍率。

命令行另保留 Camera2 USB 相机串流，完成“触发后取新帧 → 完整纸边检测 → 透视与方向校正 → 随机画派生成 → 文件完整性检查 → 直接保存数字结果”。下述 Camera2 参数与前台要求仅适用于这条显式命令行路线。图像处理和 AI 在电脑上执行。

默认方法为 `extraction.method="paper"`。完整矩形画纸平放在纯色深灰哑光底板上，当前底色为 **#3D3D3D**。画作可接近铺满取景，四边只需保留能清楚区分纸边与底板的小幅留边，没有固定百分比或像素留边要求，也不需要标记、棋盘或固定放置区。纸张四角和四条边必须完整处于相机帧内。

手机固定俯拍，当前 Camera2 取流为后置 `camera_id=0`、zoom **1.5**、2048 × 1536 / 30 fps / H.264 16 Mbps，输出方向配置为 `rotation_degrees=0`。该取景比例独立于原生相机界面 2×；以四边完整可见为准。手离开纸面后再采集，避免反光和遮挡。

Camera2 路线的采集、确认与生成使用 CLI 终端，不启动采集 GUI、不控制桌面鼠标，也不自动打开结果窗口。相关配置为 `configs/phone_capture.json`；生成、模型输入尺寸和启用画派继续使用 `configs/app.json` 与 `configs/styles.json`。

## 2. 命令行持续取流与新帧采集

默认 `capture_backend="camera_stream"`，使用官方 scrcpy-server 5.0.1 的 Camera2 源，经 USB 接收 H.264 并由 PyAV 解码。手机须开启 USB 调试并授权当前电脑。后端启动会停止原生相机以避免占用，并临时设置 USB 常亮；退出时恢复原常亮设置，不需要前台原生相机。不读取手机屏幕、不发送解锁手势，也不普遍要求解锁；设备系统若限制相机权限则报告启动错误。

模型预热完成后启动串流，收到足够帧并稳定至少一秒后进入可采集状态。启动与预热分别记录，不混入之后的 P/auto 作品任务耗时。

| 条件 | 行为 |
|---|---|
| P、T/R 或 auto 触发采集 | 等待序号更新且本地解码时间晚于本次触发的帧，不返回触发前缓存 |
| 新帧满足本地帧龄与超时要求 | PNG 保存解码帧，随后整纸提取 |
| 串流断开、解码失败、帧过期或取帧超时 | 本轮失败，不读旧帧、不调用原生快门、不切换后端 |
| 四边不完整、证据不足或存在多个可信边界 | 提取失败，不提交生成 |

默认不读取屏幕、不处理相机 UI、不触发快门，也不查询 DCIM。PNG 来源是有损 H.264 解码帧，不代表原生相机 JPEG 或无损传感器原始图像。

记录 `frame_sequence`、`trigger_sequence`、触发及本地解码时间，要求取到触发后新解码的帧。`frame_age_scope="local_decode_only"`；帧龄仅表示电脑解码后的时间，不等于传感器曝光到电脑的完整链路延迟。

旧 ADB 采集可显式选择 `--config configs/phone_capture_adb.json`：保留原生相机 2×、预览 ROI／UI 标定、截图优先和最多一次本次新照片回退。该兼容路径的秒级快门与相册读取不属于当前默认串流采集。

## 3. CLI 操作

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m app.phone --probe
.\.venv\Scripts\python.exe -m app.phone --mode confirm --cli
.\.venv\Scripts\python.exe -m app.phone --mode direct --cli
.\.venv\Scripts\python.exe -m app.phone --mode auto --headless
```

也可分别使用 `.\start_phone.cmd`、`.\direct_phone.cmd`、`.\debug_phone.cmd`。默认串流的 `--probe` 只检查设备与 server 文件，不启动相机、不消费视频帧；生成服务按 README 启动。

当前 `generation_warmup=true`：入口先用合成纯色输入和各画派静态生成 Prompt，分别完整预热全部八个启用画派，再连接串流。每项种子不同，不创建用户作品任务，也不消耗随机画派分配；任一项失败即停止启动。
当前 FLUX.2 Klein 4B 配置以横向 A4 原生尺寸 1088×768 逐个预热八个画派，结果写入 `runs/runtime/generation-warmup.json` 的 `style_prewarms`。八派共用同一套 4B 权重和 4 步采样，各自使用不同 Prompt；[本次八派预热报告](validation/eight-style-warmup-20261025.json)记录全部完成，耗时及测量范围见 [MVP_SPEC.md](../MVP_SPEC.md)；输入为合成纯色，不是用户拍照全流程、完整冷启动或画质验收。启动准备独立于作品任务及单轮 10 秒预算，不能把就绪后的生成速度当作冷启动速度。

`confirm` 与 `direct` 都是交互入口，按键含义完全相同；`auto` 仍为无需输入的一轮自动流程。

| 按键或入口 | 行为 |
|---|---|
| P | 采集新图，完成提取、一次生成和保存 |
| T | 仅采集与提取，输出图像路径并等待确认，不生成 |
| Enter | 确认当前待确认输入并生成，不重新拍照 |
| R | 仅重新采集与提取，等待确认，不生成 |
| Esc / Q | 取消待确认输入／退出 |
| `auto` | 启动后无需输入，执行一次真实全流程，成功或失败后停止 |

T/R 后可自行查看终端输出的 `artwork.png` 和 `input.png`，再按 Enter；程序不自动打开图像窗口。P 始终采集新图并运行全流程，不因入口名为 `confirm` 而改变。输入确认只用于采集与提取结果；生成后不进行主题或风格审查。

每次只处理一件作品，忙碌时不排队；连按或长按 P 不创建重复任务。重新拍摄建立新的采集记录。

## 4. 纸面提取与颜色保留

实现参考成熟 OpenCV 文档扫描流程及 [jscanify](https://github.com/puffinsoft/jscanify)（[MIT 许可](https://github.com/puffinsoft/jscanify/blob/master/LICENSE)）的算法思路，采用 Python/OpenCV 独立实现，未直接复制其源代码。jscanify 官方也建议平面与纯色背景。

| 阶段 | 要求 |
|---|---|
| 纸边检测 | 在缩小的预览或照片中，以边缘、阈值和轮廓生成四边形候选 |
| 可靠性检查 | 验证凸四边形、四侧连续性、明暗对比及候选唯一性；旧 ADB 配置另排除 UI 干扰 |
| 透视校正 | 在原彩图像上按可靠纸边拉正，去除整张纸面以外的底板 |
| 方向校正 | 显式配置角度；当前串流为 `rotation_degrees=0`，保留的 ADB 配置为 `270` |
| 输出 | 保留整张纸面，按输出尺寸上限缩放，再准备模型输入 |

灰度、Canny 和阈值处理只用于检测几何，不用于最终图像的黑白化。输出保留原彩、黑白笔画、浅色内容及纸内留白，不进行逐笔触颜色删除或语义补画。该方法处理完整不透明矩形画纸，不承诺从任意不规则物件中分离笔触。

`artwork.png` 带 alpha 通道，纸面内部保持不透明；外部背景通过四边形裁切移除。检测到的整张纸面直接进入 `for_artwork` 路径，不再次执行画框检测或默认四边 6% 裁边。模型输入长边从 `configs/app.json` 读取，保持原作比例。

检测长边上限为 1000，校正输出长边上限为 1536。原生大照片缩小时先做面积采样以抑制混叠；纸边置信度不足时失败，不通过扩大或固定比例裁边掩盖问题。

`aruco` 和 `checkerboard` 仍可作为兼容配置选项使用，各自遵守对应背景条件；默认 `paper` 路径不使用标记或放置区。

## 5. 验证范围

当前八派合成输入预热及扩充八派前的单件 4B 生成／超时验证见 [验证记录](VALIDATION.md)。预热不等于当前八派真实作品的完整流程、固定速度或画质验收。公开的 [八画派样例](../examples/README.md) 使用各自记录的历史 Qwen 配方，不能作为当前 FLUX.2 输出证据。

## 6. 一次生成后直接输出

提取成功且满足当前按键的提交条件后，才创建正式任务，从八种启用画派中等概率随机选择一种，完整列表见 [MVP_SPEC.md](../MVP_SPEC.md)。原作直接作为图像条件，同画派跨输入共用 Prompt，不加入逐图识别得到的场景描述。原有三派沿用共用模板，五派通过可选 generate_prompt 准确复用历史研究目录的完整 Prompt；不是加载八个独立模型。

当前生成为 FLUX.2 Klein 4B FP8，独立编码器为 Qwen3 4B FP4，VAE 为 `flux2-vae.safetensors`。4B 指视觉生成模块，独立编码器另计；FP8／FP4 是量化格式。权重来源及哈希见[模型清单](validation/model-sources.json)。工作流 `workflows/image_edit_flux2_klein.json` 使用 A4 原生尺寸（横向 1088×768、竖向 768×1088）、4 steps、CFG 1、Euler 和 `Flux2Scheduler`；原作经 VAE 编码，由 `ReferenceLatent` 提供图像条件，并使用独立 Flux2 目标 latent，不额外运行 Prompt 增强。`configs/app_qwen21_a4.json` 保留此前 Qwen A4 配置，`configs/app_qwen21_fast.json` 保留 Qwen 768 长边配方；`configs/app_flux2.json`、`configs/app_lcm.json`、`configs/app_qwen.json` 保留其他配方。

最终输出采用 A4 比例、短边 720 像素：横向 1018×720、竖向 720×1018。完整原图以 contain 等比放入原生 A4 画布，必要时少量补白，不裁内容、不拉伸；生成后Lanczos4 下采样至最终尺寸。A4 原纸可自然填满对应画幅。这里的 720 指纸张画幅短边，不是 16:9 视频规格。

已验证 ComfyUI 0.39.0、Torch 2.11.0+cu130 与 CUDA backend，服务端口 8189；环境独立安装，步骤见 [部署说明](DEPLOYMENT.md)。

应用配置为 `1.10.0`，生成 Prompt 为 `1.7.0`，生产 `checker.enabled=false`。每个新任务只生成一次；随后仅检查生成文件是否完整、可解码及尺寸是否有效，有效即保存 `output.png` 并由 `GENERATING` 进入 `COMPLETED`。不进入 `CHECKING`，不调用主题、绘制变化或风格模型，不因语义质量重试。

单轮 AI 阶段为 10 秒截止预算（`generation.time_budget_sec=10`），包含生成准备、上传、推理、结果读取、文件检查、下采样和保存，不包含采集、纸面校正或会话预热。到时仍无完整成品则进入 `FAILED / timeout`，取消本轮请求，不输出替代图。GUI 仅在界面提示，保留原图和校正图，不向图片添加标注，并重新激活 P；该预算用于结束等待，不保证每次都有成品。

记录必须为 `review_status="not_performed"`、`validation_scope="file_integrity_only"`；不伪造主题、风格、`passed` 或审查 JSON。文件损坏、解码或尺寸错误、生成服务不可用及超时仍直接失败，不使用云端替代。

生产应用与 `doctor` 不依赖 Ollama。审查训练和质量校准在流程外进行；旧检查器的 `enforced` / `deferred` 只保留为独立校准或历史任务兼容。`checker.enabled` 缺省为 `true`，已有任务沿用自己的配置快照；当前禁用审查不能改写旧失败，重跑旧审查任务须按提示显式启用检查器。显式启用时缺省策略为 `enforced`。

## 7. 文件、并发与恢复

采集会话保存在 `runs/phone/{session_id}/`，正式提交前不创建或随机分配生成任务。

| 内容 | 用途 |
|---|---|
| 原始 PNG | 默认保存本次触发后取得的视频解码帧；旧 ADB 路径保留实际 PNG/JPEG 来源 |
| `preview.png` | 采集来源对应的预览图 |
| `detection.png` | 原图坐标中的纸边识别结果 |
| `board.png` | 透视校正后的纸面，输出方向旋转前 |
| `mask.png` | 最终纸面的 alpha 掩膜 |
| `artwork.png` | 带 alpha 的完整纸面，纸内不透明 |
| `input.png` | 提供给生成模型的作品图 |
| `meta.json` | 来源、实际尺寸、角点、证据、分段耗时、状态与生成任务链接；正式任务记录未执行审查及文件完整性范围 |

仅保存已完成阶段的有效文件。失败时保留已有图像和原因，不伪造后续产物。

会话状态包括采集、提取、等待输入确认（`AWAITING_INPUT_CONFIRM`）、生成、完成和失败。通过 `generation_job_id` 关联 `runs/phone-{session_id}/` 的正式任务；生成图、参数与最终结果由正式任务保存。当前 `output.png` 在文件完整性检查后直接产生，不创建未执行的审查 JSON；旧审查文件保持原状。

手机入口全程持有跨进程文件锁，第二个实例被拒绝。进程退出后由操作系统释放，不通过删除锁文件或旧 PID 判断运行状态。

启动恢复只更新手机前置会话，并只读关联的正式任务：

| 已保存状态 | 恢复行为 |
|---|---|
| 采集、提取或等待确认时中断 | 标记前置会话失败，旧输入不自动重新提交 |
| 关联任务已完成且最终图存在 | 补齐前置完成状态及结果路径 |
| 关联任务已失败，或任务／应有输出缺失 | 同步或标记前置失败并保留原因 |
| 关联生成任务仍活动 | 保留原记录并阻止新一轮，不重复提交 |
| 前置会话已完成、失败、取消或被重取替代 | 保持原记录 |

恢复不调用模型、不更改正式任务、不自动处理未关联的一般任务。采集前也须确认正式任务入口空闲。活动任务按提示使用现有恢复入口处理：

```powershell
.\.venv\Scripts\python.exe -m app.main resume
```

入口错误日志位于 `runs/runtime/phone.log`。

## 8. 配置字段

相对路径均以项目根目录为基准。

| 字段 | 当前设置或要求 | 含义 |
|---|---|---|
| `mode` | `confirm` | `confirm` / `direct` 均使用统一交互键义；`auto` 自动执行一轮 |
| `capture_runs_dir` | `runs/phone` | 手机采集会话目录 |
| `generation_warmup` | `true` | 启动时依次完整预热八个画派，再开始串流；独立计时 |
| `preprocess.paper_format` | `a4` | 生成配置 `configs/app.json`：A4 画布，完整原图 contain 放置 |
| `preprocess.output_short_side` | `720` | 最终输出短边，横向 1018×720、竖向 720×1018 |
| `preprocess.max_input_side` | `1088` | 原生模型输入长边上限 |
| `phone.adb_path` | `tools/android/platform-tools/adb.exe` | ADB 可执行文件 |
| `phone.serial` | 当前手机序列号 | `null` 时仅在单一设备下自动选择 |
| `capture_backend` | `camera_stream` | 持续 Camera2 取流；`adb` 仅兼容配置 |
| `stream.server_path` / `version` | `tools/android/scrcpy-server-v5.0.1` / `5.0.1` | 官方相机取流 server |
| `stream.camera_id` | `0` | 当前后置相机 |
| `stream.size` / `fps` | `2048x1536` / `30` | 请求的取流尺寸与帧率 |
| `stream.zoom` | `1.5` | Camera2 zoom，独立于原生相机界面的倍率 |
| `stream.bit_rate` | `16000000` | H.264 码率 |
| `stream.startup_timeout_sec` | `12` | 启动取流超时 |
| `stream.fresh_frame_timeout_sec` | `1.0` | 触发后等待新解码帧的超时 |
| `stream.max_frame_age_ms` | `150` | 本地解码后的最大帧龄，含 PNG 编码结束检查 |
| `extraction.method` | `paper` | 完整矩形纸面检测；旧方法仅兼容 |
| `extraction.max_detection_side` | `1000` | 纸边检测长边上限 |
| `extraction.max_output_side` | `1536` | 透视校正输出长边上限 |
| `extraction.rotation_degrees` | `0` | 当前串流方向正确，不额外旋转 |

模型输入长边继续使用 `configs/app.json` 的 `preprocess.max_input_side`，不在手机配置中重复维护。
应用配置的 `checker.enabled=false` 控制当前直接输出路径；该字段必须是布尔值，缺省旧配置按 `true`，不得借此改写历史任务。

## 9. 验收条件

- 当前 Camera2 后置相机 zoom 1.5，画纸接近铺满取景且四边完整可见；深灰底只留清楚辨认纸边所需的小幅边距。
- 默认持续串流，触发后取得新解码帧并记录序号和本地帧龄；断流、过期或超时直接失败，不回退快门／相册。
- 纸面透视与方向正确，纸内原彩、黑白内容和留白保留，纸外底板移除。
- 所有交互 CLI 入口中 P 执行全流程，T/R 仅采集提取，Enter 生成当前待确认输入；`auto` 启动后执行一轮并停止。
- 连按或长按不重复提交；跨进程锁、中断输入和关联生成任务按恢复规则处理。
- 原始文件、采集来源、输出与会话／生成任务一一对应，不复用旧照片或覆盖旧记录。
- 分别报告采集、提取、生成与总耗时；复用已有输入时标明来源，不把纯算法时间当成端到端速度。
- 每件只生成一次，最终输出可解码且尺寸有效，并记录 `not_performed` / `file_integrity_only`；主题和风格训练移至流程外，本阶段不接入真实打印。

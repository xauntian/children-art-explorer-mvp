# Children’s Art Explorer — 当前技术规格

更新日期：2026-10-25（香港时间）。本文件是技术规格入口，完整的当前功能边界、参数、状态与保存规则以 [MVP_SPEC.md](MVP_SPEC.md) 为准。

- [README.md](README.md)：项目概览、启动与操作。
- [MVP_SPEC.md](MVP_SPEC.md)：当前数字 MVP 的完整技术规格。
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)：环境、模型与设备部署。
- [docs/VALIDATION.md](docs/VALIDATION.md)：实际验证、测量范围和未完成项。
- [MVP_V1.0.md](MVP_V1.0.md)：项目研究与设计说明。
- [agent.md](agent.md)：开发指引。

## 当前数字流程

**A4 白纸创作 → 工作人员在 GUI 按 P → 原生相机当前取景截图 → 完整纸边检测与透视校正 → 八派随机选择 → 本地 AI 生成 → 文件完整性检查 → A4 数字图片保存。**

GUI 为主要操作入口，使用 `start_gui.cmd`。点击「开始」后检查手机与模型，准备 ComfyUI，逐一预热全部八个启用画派，完成后接收 P。待机第一栏显示手机屏幕的实时视频；采集后在处理中保留本轮原图，校正图与 AI 图随进度显示。P 在一轮处理中失效，不排队；保存完成或 AI 超时结束后重新激活。

工作人员在同一部 Mate 40 Pro 上手动打开原生相机并设置取景。GUI 不读取倍率或 UI XML，不后台定时截图，不更改相机设置、不触发快门；正式原图由 P 独立采集，不使用视频预览帧代替。命令行保留独立 Camera2 串流入口，其设置不适用于 GUI 原生相机取景。

## 当前配置与边界

| 项目 | 当前规则 |
|---|---|
| 应用 / 生成 Prompt | `configs/app.json` 版本 1.10.0；Prompt 1.7.0 |
| 模型 | FLUX.2 Klein 4B FP8；独立 Qwen3 4B FP4 编码器与 flux2-vae |
| 采样 | 4 steps、CFG 1、Euler、Flux2Scheduler |
| 随机画派 | 野兽派、立体主义、印象派、后印象派（梵高式方向）、表现主义、超现实主义、新造型主义、波普艺术 |
| 画派实现 | 共用同一套基础权重，使用各派 Prompt；不对应八个独立模型或已训练 LoRA |
| 单轮生成 | 每任务等概率随机一次，只生成一次；原作真实进入图像条件链路 |
| AI 阶段截止 | 10 秒；拍照、纸面校正和会话预热另计 |
| 图片输出 | A4 比例；横向 1018×720、竖向 720×1018，由更高原生尺寸下采样 |
| 语义审查 | `checker.enabled=false`；不调用主题／风格审查，不依赖 Ollama，不按语义结果重试 |
| 打印 | GUI `printer.enabled=false`；跳过打印检查、提交与出纸等待，保留既有打印记录 |
| 部署 | 本地服务；技术错误报告失败，不静默切换云端 |

10 秒预算用于结束 AI 等待，不保证每次都有成品。到达截止时间仍无完整 AI 图片时，本轮失败，仅在 GUI 提示超时，保留原图和校正图；不输出替代图片、不向图片添加标注，随后恢复 P。生成准备、上传、推理、结果读取、文件检查、下采样和保存的预算范围见 [MVP_SPEC.md](MVP_SPEC.md)。

文件存在、可解码且尺寸有效才保存最终输出，任务记录明确标记 `review_status="not_performed"`、`validation_scope="file_integrity_only"`。文件完整不等于画派质量或主题关联已通过评估。

Windows 实体 A4 打印与作业监视代码已接入，但当前关闭、真实出纸尚未联调。重新启用后的锁定、驱动状态、人工出纸确认和重启恢复规则见完整规格；打印故障不得导致重新生成或自动重印。

## 证据与后续验证

公开开发样例保留在 [examples/gallery.html](examples/gallery.html)，其来源、参数与状态见 [examples/README.md](examples/README.md) 和 [examples/manifest.json](examples/manifest.json)。这些历史展示图不能证明当前 FLUX.2 的画质、速度或定制训练完成。

真实测量和测试结果统一见 [docs/VALIDATION.md](docs/VALIDATION.md)。实体打印、当前 GUI 连续多件和长期运行、断网完整流程、现场纸张与光照、独立画质以及儿童体验仍需分别验证。儿童触屏、传感器触发、实体反馈与展览结构尚未接入；后续方向见项目研究说明。

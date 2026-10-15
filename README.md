# Children’s Art Explorer · MVP V1.0

面向儿童艺术展览的本地生成原型：**粗马克笔创作 → 拍摄或导入 → 随机画派 → 本地重绘 → 自动粗检 → 保存数字作品**。

儿童使用实体材料，后台由工作人员触发。当前支持野兽派、立体主义和印象派，采用本地 ComfyUI 与 Ollama。打印仅保留接口，默认关闭。

**V1.0 是工程与文档基线。三画派的独立质量验收、摄像头现场测试和儿童体验研究尚未完成。**

## 文档入口

| 文档 | 用途 |
|---|---|
| [MVP_V1.0.md](MVP_V1.0.md) | 项目概念、儿童与艺术研究、当前方案、技术路径、验证和课程时间线 |
| [SPEC.md](SPEC.md) | 可执行需求、模块接口、状态、参数和验收条件 |
| [agent.md](agent.md) | 开发 Agent 的工作范围与规则；[AGENTS.md](AGENTS.md) 为自动发现入口 |
| [部署说明](docs/DEPLOYMENT.md) | Python、ComfyUI、模型、Ollama 的完整安装与故障处理 |
| [样例与配方](examples/README.md) | 三张输入、九张结果、认可状态、实际 Prompt 和复现命令 |
| [验证记录](docs/VALIDATION.md) | V1.0 发布检查及尚未完成的验证 |

## 当前结果

| 画派 | 状态 |
|---|---|
| 野兽派 | 三张开发候选，尚未完成独立验收 |
| 立体主义 | 三张展示图获用户认可；跨新输入的稳定性仍待验证 |
| 印象派 | 山坡与房屋为用户选定参照；花园和家庭的新候选仍待评价 |

最新印象派以环境光、空间与自然人物比例为重点。花园保留人物、房屋、树与花地；家庭保留四人牵手。两张改为背影，笔触比山坡参照细软。完整比较见 [样例页](examples/README.md)；克隆后可在浏览器打开 [本地画廊](examples/gallery.html)。GitHub 文件页仅显示 HTML 源码。

## 快速开始

以下命令在 Windows PowerShell 中执行。先安装 Git 和 Python 3.12：

```powershell
git clone https://github.com/xauntian/children-art-explorer-mvp.git
cd children-art-explorer-mvp
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.main doctor --offline
.\.venv\Scripts\python.exe -m pytest tests/unit
```

这一步可检查配置和程序规则，不会调用图像模型。生成前，按 [部署说明](docs/DEPLOYMENT.md) 安装并启动本地 ComfyUI（`127.0.0.1:8188`）与 Ollama（`127.0.0.1:11434`），下载所需权重。

```powershell
.\.venv\Scripts\python.exe -m app.main doctor
.\.venv\Scripts\python.exe -m app.main import --photo samples/children_originals/01_garden.png --artwork
```

`--artwork` 适用于已经完整裁好的画作，关闭前处理中的纸边检测和裁边。带画框或背景的真实照片使用普通导入。应用默认工作流仍有模型侧中心裁切；样例配方另外关闭了该裁切，两者不要混为一谈。

正常导入等概率随机选择一个画派。开发调试可加 `--style cubism`，不改变默认随机机制。通过后生成 `runs/<job_id>/output.png`；不通过则保留候选和原因，不产生最终图。

```powershell
.\.venv\Scripts\python.exe -m app.main doctor --camera
.\.venv\Scripts\python.exe -m app.main camera
.\.venv\Scripts\python.exe -m app.main list --all
.\.venv\Scripts\python.exe -m app.main resume
```

相机默认使用 Windows DirectShow，其他平台需调整 `capture.backend` 并实机验证。日常重做请新建任务；开发参数 `--force` 会重跑指定终态任务，可能重写其候选与记录，不用于保留历史结果。

## 生成与确认规则

- 画派特征优先，允许省略、合并、增补相关元素和重组构图；结果需保留主题关联或可辨认线索。
- 每画派跨原图使用共用摘要；原图直接作为图像条件，不插入逐图识别所得的场景描述。
- 自动确认必须同时满足 `style_match == "match"` 和 `content_preserved is True`。后者表示主题关联，不要求完整保留全部元素。
- 每次正常自动流程最多两次生成：质量失败重试一次，固定输入与画派；技术错误直接失败。
- 仅确认并可靠保存的结果进入 `COMPLETED`。无效、截断或类型错误的检查响应不视为成功。
- 无云端推理、人工审批、Arduino 或真实打印步骤。

## 技术配置

| 环节 | 当前配置 |
|---|---|
| 后台 | Python 3.12、OpenCV、Pillow、Requests |
| 图像生成 | ComfyUI + ComfyUI-GGUF；Qwen-Image-Edit-2509 Q3_K_M |
| 图像编码 | Qwen2.5-VL-7B-Instruct Q4_K_M + 配套 mmproj；Qwen Image VAE |
| 默认采样 | 4 steps、CFG 1、Lightning 4-step LoRA；Euler/simple、shift 3、denoise 1 |
| 本地粗检 | Ollama `qwen3-vl:8b-instruct`；`comparison-v3-style-first`；默认 CPU |
| 配置版本 | 应用 1.2.0；生成 Prompt 1.6.0；工程 1.0.0 |

展示配方与应用默认分开保存：野兽派示例为 4 步、CFG 1.5、Lightning；立体主义及印象派示例为 20 步、CFG 4、无 LoRA。样例的用户认可不等于自动确认，仓库没有训练新的画派权重。

## 项目结构

```text
app/                  采集、生成、检查、状态与保存
configs/              应用与三画派配置
prompts/              应用默认生成和检查提示
workflows/            ComfyUI API 工作流
tools/                生成实验、评估与检查器校准工具
tests/                规则测试、集成测试及固定校准夹具
samples/children_originals/  三张开发输入
examples/             当前展示、实际参数、共用提示与独立配方
docs/                 部署与验证说明
runs/                 本机任务与生成记录，Git 忽略
```

模型权重、运行环境、任务日志和本机配置不随仓库发布。示例输入已用于调参，不属于独立验收集，也不代表已完成儿童参与者测试。

## 验证与后续工作

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit
.\.venv\Scripts\python.exe -m ruff check app tools tests conftest.py
```

需要主动调用真实本地服务时才运行 `python -m pytest tests/integration -v -m integration`。不要用裸 `pytest` 作为离线检查，它会收集集成测试。

后续依次验证稳定的画派配方、独立输入质量、检查器误判、连续三件、断网及相机体验，再开展儿童现场研究。课程节点为 10 月 27 日进度汇报、11 月 24 日最终汇报、12 月 4 日 17:00 报告提交，均为 2026 年香港时间。

"""生成环境勘查报告（Markdown）。

用法：
    python -m tools.environment_report --out docs/environment-report.md
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd: list[str], timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (result.stdout or "").strip() or (result.stderr or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"<失败: {exc}>"


def collect() -> dict:
    info: dict = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    info["torch"] = {}
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "arch_list": torch.cuda.get_arch_list(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            info["torch"]["device_name"] = props.name
            info["torch"]["capability"] = f"{props.major}.{props.minor}"
            info["torch"]["vram_total_gb"] = round(total / 1024**3, 2)
            info["torch"]["vram_free_gb"] = round(free / 1024**3, 2)
    except Exception as exc:  # noqa: BLE001
        info["torch"] = {"error": str(exc)}

    info["nvidia_smi"] = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"])
    info["disk"] = {}
    for drive in ("C:\\", "D:\\"):
        try:
            usage = shutil.disk_usage(drive)
            info["disk"][drive] = {
                "total_gb": round(usage.total / 1024**3, 1),
                "free_gb": round(usage.free / 1024**3, 1),
            }
        except OSError:
            pass

    info["ollama_version"] = run(["ollama", "--version"])
    info["ollama_models"] = run(["ollama", "list"])

    comfy_dir = ROOT / "comfyui"
    info["comfyui_path"] = str(comfy_dir)
    version_file = comfy_dir / "comfyui_version.py"
    if version_file.is_file():
        info["comfyui_version"] = version_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    info["comfyui_commit"] = run(["git", "-C", str(comfy_dir), "log", "-1", "--format=%H %ad"])

    models_root = comfy_dir / "models"
    info["model_files"] = []
    for sub in ("checkpoints", "controlnet", "loras", "unet", "vae", "text_encoders"):
        folder = models_root / sub
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in (".safetensors", ".ckpt", ".gguf", ".pt", ".bin"):
                info["model_files"].append(
                    {
                        "folder": sub,
                        "name": path.name,
                        "size_gb": round(path.stat().st_size / 1024**3, 2),
                    }
                )

    return info


def collect_app_report() -> dict:
    try:
        from app.main import App

        app = App()
        return app.doctor(probe_camera=True, probe_models=True)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def render(info: dict, app_report: dict, unit_tests: str) -> str:
    lines = [
        "# 环境勘查报告",
        "",
        f"生成时间：{info['generated_at']}",
        "",
        "> 本报告由 `python -m tools.environment_report` 自动生成，记录实际检测结果。",
        "",
        "## 1 主机与系统",
        "",
        "| 项目 | 实测值 |",
        "|---|---|",
        f"| 操作系统 | {info['platform']} |",
        f"| 架构 | {info['machine']} |",
        f"| 处理器 | {info['processor']} |",
        f"| Python | {info['python']} (`{info['python_executable']}`) |",
        "",
        "## 2 GPU 与 PyTorch",
        "",
    ]

    torch_info = info.get("torch", {})
    if "error" in torch_info:
        lines.append(f"PyTorch 检测失败：`{torch_info['error']}`")
    else:
        lines += [
            "| 项目 | 实测值 |",
            "|---|---|",
            f"| PyTorch | {torch_info.get('version')} |",
            f"| CUDA 构建版本 | {torch_info.get('cuda_build')} |",
            f"| CUDA 可用 | {torch_info.get('cuda_available')} |",
            f"| 设备 | {torch_info.get('device_name')} |",
            f"| 计算能力 | sm_{str(torch_info.get('capability', '')).replace('.', '')} |",
            f"| 显存合计 | {torch_info.get('vram_total_gb')} GB |",
            f"| 显存可用（检测时） | {torch_info.get('vram_free_gb')} GB |",
            f"| 编译架构列表 | {', '.join(torch_info.get('arch_list', []))} |",
            "",
            "```text",
            info.get("nvidia_smi", ""),
            "```",
            "",
        ]

    lines += ["## 3 磁盘", "", "| 盘符 | 总容量 (GB) | 可用 (GB) |", "|---|---|---|"]
    for drive, usage in info.get("disk", {}).items():
        lines.append(f"| {drive} | {usage['total_gb']} | {usage['free_gb']} |")
    lines.append("")

    lines += [
        "## 4 本地模型服务",
        "",
        "### 4.1 视觉语言模型（Ollama）",
        "",
        "```text",
        info.get("ollama_version", ""),
        "```",
        "",
        "```text",
        info.get("ollama_models", ""),
        "```",
        "",
        "### 4.2 ComfyUI",
        "",
        f"- 路径：`{info.get('comfyui_path')}`",
        f"- 版本：{info.get('comfyui_version', '未检测到')}",
        f"- 提交：{info.get('comfyui_commit', '未检测到')}",
        "",
        "### 4.3 已下载的模型权重",
        "",
    ]
    model_files = info.get("model_files", [])
    if model_files:
        lines += ["| 目录 | 文件 | 大小 (GB) |", "|---|---|---|"]
        for item in model_files:
            lines.append(f"| {item['folder']} | `{item['name']}` | {item['size_gb']} |")
    else:
        lines.append("未检测到已下载的模型权重。")
    lines.append("")

    lines += ["## 5 应用自检（`python -m app.main doctor`）", "", "```json"]
    lines.append(json.dumps(app_report, ensure_ascii=False, indent=2))
    lines += ["```", ""]

    lines += [
        "## 6 规则测试结果",
        "",
        "```text",
        unit_tests.strip(),
        "```",
        "",
        "## 7 结论与未验证项",
        "",
        "- 生成与检查均在本机运行；外网仅用于一次性下载权重与依赖。",
        "- 生成模型与检查模型职责分开：ComfyUI 负责绘制，视觉语言模型负责粗略检查。",
        "- 具体耗时、显存峰值与画质通过率见 `docs/eval-report.md` 与测试记录。",
        "- 未验证项：真实儿童作品照片的采集质量、长时间连续运行、真实打印设备接入。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成环境勘查报告")
    parser.add_argument("--out", default="docs/environment-report.md")
    parser.add_argument("--skip-tests", action="store_true", help="不运行规则测试")
    args = parser.parse_args(argv)

    print("收集系统信息…")
    info = collect()
    print("运行应用自检…")
    app_report = collect_app_report()

    unit_tests = "<已跳过>"
    if not args.skip_tests:
        print("运行规则测试…")
        unit_tests = run(
            [sys.executable, "-m", "pytest", "tests/unit", "-q", "--no-header"], timeout=600
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(info, app_report, unit_tests), encoding="utf-8")
    print(f"报告已写入 {out}")

    json_out = out.with_suffix(".json")
    json_out.write_text(
        json.dumps({"system": info, "app": app_report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"原始数据已写入 {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

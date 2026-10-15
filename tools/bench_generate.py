"""生成配置实测脚本：记录一次生成的耗时、缓存状态与参数，不进入检查与确认流程。

用途：在调整量化、LoRA、步数、输入尺寸时快速对比耗时，
避免每次都走完整流程（含检查与保存）。

用法：
    python -m tools.bench_generate --photo samples/debug/debug_01_sun_house.jpg
    python -m tools.bench_generate --photo X.jpg --steps 4 --no-lora
    python -m tools.bench_generate --photo X.jpg --steps 20 --input-side 512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.capture import import_photo  # noqa: E402
from app.config import load_app_config  # noqa: E402
from app.generator import ComfyClient, ComfyGenerator, load_workflow  # noqa: E402
from app.preprocess import decode_image_bytes, prepare_model_input  # noqa: E402


def cache_evidence(history: dict, prompt_id: str, sampler_ids: list[str]) -> dict:
    """只把当前任务明确返回的 execution_cached 事件作为缓存证据。"""
    result = {"status": "unknown", "sampler_ids": sampler_ids, "cached_nodes": []}
    entry = history.get(prompt_id) if isinstance(history, dict) else None
    if not isinstance(entry, dict) or not sampler_ids:
        return result
    status = entry.get("status")
    messages = status.get("messages") if isinstance(status, dict) else None
    if not isinstance(messages, list):
        return result
    cached_nodes: set[str] = set()
    found = False
    for message in messages:
        if not isinstance(message, (list, tuple)) or len(message) != 2:
            continue
        event, payload = message
        if event != "execution_cached" or not isinstance(payload, dict):
            continue
        if payload.get("prompt_id") != prompt_id or not isinstance(payload.get("nodes"), list):
            continue
        if any(not isinstance(node, str) for node in payload["nodes"]):
            continue
        found = True
        cached_nodes.update(payload["nodes"])
    if found:
        hits = set(sampler_ids) & cached_nodes
        result["status"] = "hit" if hits == set(sampler_ids) else "partial" if hits else "miss"
        result["cached_nodes"] = sorted(cached_nodes)
    return result


def workflow_model_metadata(nodes: dict, save_node_id: str) -> dict:
    """读取解析后且实际连接到输出的模型依赖，忽略配置中的闲置模型与游离节点。"""
    active: set[str] = set()
    pending = [save_node_id]
    while pending:
        node_id = pending.pop()
        if node_id in active or node_id not in nodes:
            continue
        active.add(node_id)
        for value in nodes[node_id].get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                pending.append(value[0])

    def inputs_for(*classes: str) -> dict:
        return next(
            (node["inputs"] for node_id, node in nodes.items()
             if node_id in active and node["class_type"] in classes),
            {},
        )

    unet = inputs_for("UnetLoaderGGUF", "UNETLoader")
    checkpoint = inputs_for("CheckpointLoaderSimple")
    encoder = inputs_for("CLIPLoaderGGUF", "CLIPLoader")
    vae = inputs_for("VAELoader")
    lora = inputs_for("LoraLoaderModelOnly", "LoraLoader")
    sampling = inputs_for("ModelSamplingAuraFlow", "ModelSamplingSD3")
    return {
        "model_route": "unet" if unet else "checkpoint" if checkpoint else "unknown",
        "model": unet.get("unet_name") if unet else checkpoint.get("ckpt_name"),
        "text_encoder": encoder.get("clip_name"),
        "vae": vae.get("vae_name"),
        "lora_name": lora.get("lora_name"),
        "lora_strength": lora.get("strength_model"),
        "shift": sampling.get("shift"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ComfyUI 生成配置实测")
    parser.add_argument("--photo", required=True, help="输入照片路径")
    parser.add_argument("--config", type=Path, default=None, help="本次实验配置路径；省略时使用 configs/app.json")
    parser.add_argument(
        "--artwork", action="store_true", help="完整画作输入：关闭画框检测和四周裁切"
    )
    parser.add_argument("--style", default="fauvism", help="用于取生成参数的画派 id")
    parser.add_argument("--steps", type=int, default=None, help="覆盖采样步数")
    parser.add_argument("--cfg", type=float, default=None, help="覆盖 CFG")
    parser.add_argument("--input-side", type=int, default=None, help="覆盖 max_input_side")
    parser.add_argument("--no-lora", action="store_true", help="本次不使用 LoRA")
    parser.add_argument("--lora-strength", type=float, default=None, help="覆盖 LoRA 强度")
    parser.add_argument("--prompt", default=None, help="覆盖生成 Prompt（含 {style_*} 占位符）")
    parser.add_argument("--seed", type=int, default=None, help="固定实验种子；省略时随机生成并记录")
    parser.add_argument("--tag", default=None, help="输出文件名附加标记")
    parser.add_argument("--out", default="samples/_bench", help="输出目录")
    args = parser.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed < 2**64:
        parser.error("--seed 必须为 0 到 2**64-1 的整数")
    seed = args.seed if args.seed is not None else secrets.randbelow(2**63)

    config_path = (args.config or (ROOT / "configs" / "app.json")).resolve()
    config = (
        load_app_config(ROOT, config_path=config_path) if args.config is not None
        else load_app_config(ROOT)
    )
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if args.artwork:
        config = config.for_artwork()
    registry = config.load_styles()
    style = registry.resolve(args.style)
    generate_prompt, _ = config.load_prompts()

    input_side = args.input_side or config.pre_max_input_side
    photo = Path(args.photo)
    # 校验照片可读（与主流程一致），失败时立即报错
    import_photo(photo)
    result, input_bytes = prepare_model_input(
        photo,
        max_input_side=input_side,
        frame_detect=config.pre_frame_detect,
        border_trim_ratio=config.pre_border_trim_ratio,
        crop_ratios=config.pre_crop_ratios,
        min_area_ratio=config.pre_min_area_ratio,
        max_area_ratio=config.pre_max_area_ratio,
        approx_epsilon_ratio=config.pre_approx_epsilon_ratio,
    )
    width, height = result.output_size

    prompt = args.prompt if args.prompt is not None else generate_prompt
    for key, value in (
        ("style_name", style.name),
        ("style_features", style.style_features),
        ("preservation_rules", style.preservation_rules),
    ):
        prompt = prompt.replace("{" + key + "}", value)

    params = dict(config.sampler_for(style.id))
    if args.steps is not None:
        params["steps"] = args.steps
    if args.cfg is not None:
        params["cfg"] = args.cfg

    lora = None if args.no_lora else config.comfy_lora_name
    lora_strength = (
        args.lora_strength if args.lora_strength is not None else config.comfy_lora_strength
    )

    workflow_sha256 = hashlib.sha256(config.comfy_workflow_path.read_bytes()).hexdigest()
    client = ComfyClient(
        base_url=config.comfy_base_url,
        connect_timeout=config.comfy_connect_timeout,
        request_timeout=config.comfy_request_timeout,
        job_timeout=config.comfy_job_timeout,
        poll_interval=config.comfy_poll_interval,
    )
    generator = ComfyGenerator(
        client=client,
        workflow_nodes=load_workflow(config.comfy_workflow_path),
        checkpoint=config.comfy_checkpoint,
        negative_prompt=config.negative_prompt,
        controlnet=config.comfy_controlnet,
        controlnet_strength=config.controlnet_strength,
        unet_name=config.comfy_unet_name,
        text_encoder=config.comfy_text_encoder,
        vae_name=config.comfy_vae_name,
        shift=config.comfy_shift,
        lora_name=lora,
        lora_strength=lora_strength,
    )
    resolved_workflow = generator.build_workflow(
        input_image_name=f"bench-{photo.stem}.png",
        positive_prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        params=params,
        filename_prefix=f"bench/{photo.stem}",
    )
    models = workflow_model_metadata(resolved_workflow, generator.binding["save_image"])

    print(f"输入      : {photo.name} -> {width}x{height} (max_input_side={input_side})")
    print(f"画派      : {style.id} ({style.name})")
    print(f"生成模型  : {models['model']} ({models['model_route']})")
    print(f"编码器    : {models['text_encoder'] or '（未使用独立编码器，随 checkpoint 加载）'}")
    print(f"VAE       : {models['vae'] or '（未使用独立 VAE，随 checkpoint 加载）'}")
    print(f"LoRA      : {models['lora_name'] or '（不使用）'}")
    print(f"Shift     : {models['shift'] if models['shift'] is not None else '（未配置采样覆盖节点）'}")
    print(f"Seed      : {seed}")
    print(f"采样参数  : {json.dumps(params, ensure_ascii=False)}")
    print("")

    try:
        generator.verify_checkpoint()
    except Exception as exc:  # noqa: BLE001
        print(f"[模型校验失败] {exc}")
        return 2

    started = time.monotonic()
    try:
        image_bytes, prompt_id = generator.generate(
            input_bytes=input_bytes,
            input_filename=f"bench-{photo.stem}.png",
            positive_prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            params=params,
            filename_prefix=f"bench/{photo.stem}",
            job_timeout=config.comfy_job_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - started
        print(f"[生成失败] 耗时 {elapsed:.1f}s  {type(exc).__name__}: {exc}")
        return 1

    elapsed = time.monotonic() - started
    sampler_ids = list(generator.binding["ksampler"])
    try:
        history = client.history(prompt_id)
        cache = cache_evidence(history, prompt_id, sampler_ids)
    except Exception as exc:  # noqa: BLE001 - 生成成功但计时证据不足时明确标记未知
        history = {}
        cache = cache_evidence(history, prompt_id, sampler_ids)
        cache["error"] = f"{type(exc).__name__}: {exc}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.tag or f"s{params['steps']}{'_nolora' if args.no_lora else ''}"
    out_path = out_dir / f"{photo.stem}_{style.id}_{suffix}_seed{seed}_{prompt_id[:8]}.png"
    out_path.write_bytes(image_bytes)

    decoded = decode_image_bytes(image_bytes, str(out_path))
    metadata = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "photo": str(photo.resolve()),
        "style_id": style.id,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "input_size": [width, height],
        "preprocess": {
            "max_input_side": input_side,
            "frame_detect": config.pre_frame_detect,
            "border_trim_ratio": config.pre_border_trim_ratio,
            "crop_ratios": list(config.pre_crop_ratios),
            "min_area_ratio": config.pre_min_area_ratio,
            "max_area_ratio": config.pre_max_area_ratio,
            "approx_epsilon_ratio": config.pre_approx_epsilon_ratio,
        },
        "output_size": [decoded.shape[1], decoded.shape[0]],
        "positive_prompt": prompt,
        "negative_prompt": config.negative_prompt,
        "seed": seed,
        "sampler_params": params,
        **models,
        "workflow_path": str(config.comfy_workflow_path),
        "workflow_sha256": workflow_sha256,
        # 上传时 ComfyUI 可能重命名输入；实际提交节点（含最终输入名）保存在 comfy_history.prompt[2]。
        "resolved_workflow_before_upload": resolved_workflow,
        "prompt_id": prompt_id,
        "cache": cache,
        "elapsed_sec": round(elapsed, 3),
        "elapsed_scope": "upload, queue, wait and download; not sampler-only time",
        "image_path": str(out_path.resolve()),
        "comfy_history": history.get(prompt_id) if isinstance(history, dict) else None,
    }
    metadata_path = out_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== 结果 ===")
    print(f"耗时      : {elapsed:.1f}s")
    print(f"prompt_id : {prompt_id}")
    print(f"输出尺寸  : {decoded.shape[1]}x{decoded.shape[0]}")
    print(f"输出字节  : {len(image_bytes)}")
    print(f"保存到    : {out_path}")
    print(f"实验记录  : {metadata_path}")
    print(f"采样缓存  : {cache['status']}")
    if cache["status"] != "miss":
        print("[计时提示] 采样缓存命中或证据不足，本次耗时不能用作真实采样速度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

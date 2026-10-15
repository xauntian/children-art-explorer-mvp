"""Frozen development cases for checker calibration; not independent acceptance.

Prepare without model calls:
    python -m tools.calibrate_checker --out docs/calibration-prepared --prepare-only
Evaluate a fresh copy through the configured checker:
    python -m tools.calibrate_checker --out docs/calibration-run --model qwen3-vl:8b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.checker import (  # noqa: E402
    COMPARISON_VERSION,
    VisionChecker,
    build_comparison_board,
    validate_check_payload,
)
from app.config import load_app_config  # noqa: E402
from app.errors import CheckTechnicalError  # noqa: E402
from app.preprocess import decode_image_bytes, encode_png, prepare_model_input  # noqa: E402
from tools.checker_audit import tone_shift, unrelated_pattern  # noqa: E402

HOUSE = "debug_01_sun_house.jpg"
SHAPES = "debug_03_shapes.jpg"
FLOWER = "debug_04_flower.jpg"
FIXTURES = "tests/assets/checker_calibration"
GENERATED = f"{FIXTURES}/generated"
CONTENT_POLICY = {
    "version": "style-first-theme-association-v1",
    "field": "content_preserved",
    "definition": "A recognisable original theme or some visual cues remain; exact element preservation is not required. Omissions, merging, theme-related additions and layout changes are allowed. Wholly unrelated images are false; insufficient evidence is null.",
}
CASES = (
    ("house", HOUSE, "fauvism", True, "match", f"{GENERATED}/debug_01_sun_house_fauvism_blank_background_seed71402_5f9b1ef2.png"),
    ("identity", HOUSE, "fauvism", True, "mismatch", None),
    ("added-tree", HOUSE, "fauvism", True, None, f"{GENERATED}/debug_01_sun_house_fauvism_preserve_prompt_seed71402_1e5ec7d7.png"),
    ("unrelated", HOUSE, "fauvism", False, None, None),
    ("tone-only", HOUSE, "fauvism", True, "mismatch", None),
    ("flower", FLOWER, "cubism", True, None, f"{GENERATED}/debug_04_flower_cubism_blank_background_seed71403_920f426f.png"),
    ("shapes", SHAPES, "impressionism", True, None, f"{GENERATED}/debug_03_shapes_impressionism_blank_background_seed71401_5bd0169a.png"),
    ("remove-square", SHAPES, "impressionism", True, "mismatch", None),
    ("swap-square-circle", SHAPES, "impressionism", True, "mismatch", None),
    ("move-sun", HOUSE, "fauvism", True, "mismatch", None),
    ("house-textured", HOUSE, "fauvism", True, "match", f"{GENERATED}/house_textured.png"),
)
NOTES = {
    "house": "Engineering coarse-style positive: expressive saturated paint blocks; house, window, door and sun remain.",
    "identity": "Unchanged original: same content, no style transformation.",
    "added-tree": "The home-and-sun theme remains recognisable despite the added tree, ground and path. Content is positive; style remains unscored.",
    "unrelated": "A generic circle and rectangle replace the home-and-sun scene, without recognisable subject or thematic cues; generic primitive shapes alone do not establish a connection.",
    "tone-only": "Hue/saturation only; original marks, subjects and layout remain.",
    "flower": "Flower, six petals, centre, stem and two separate small ovals remain. Style is unscored: weak/ambiguous cubist evidence.",
    "shapes": "Five outer shapes and overlap remain. Style is unscored: swirl treatment is not a confirmed impressionist positive.",
    "remove-square": "The remaining four original geometric forms preserve recognisable visual cues after one square is omitted. Style mismatch: the surviving marker rendering is unchanged.",
    "swap-square-circle": "The same geometric forms remain recognisable after swapping two positions. Style mismatch: moving unchanged pixels is not a painting-style transformation.",
    "move-sun": "The home-and-sun theme remains recognisable when the sun moves. Style mismatch: both subjects retain the original marker rendering.",
    "house-textured": "Engineering coarse-style positive from a retained real run: the house and sun maintain a recognisable theme with expressive paint treatment.",
}


class RecordingChecker(VisionChecker):
    """Keep complete model responses even when downstream schema validation fails."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[dict] = []

    def _post(self, endpoint: str, body: dict) -> dict:
        payload = super()._post(endpoint, body)
        messages = body.get("messages", [])
        prompts = [{"role": item.get("role"), "content": item.get("content"),
                    "image_count": len(item.get("images", []))} for item in messages]
        self.calls.append({"endpoint": endpoint, "messages": prompts,
                           "prompt": body.get("prompt"),
                           "image_count": len(body.get("images", [])), "response": payload})
        return payload


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def colour_bbox(image: np.ndarray, colour: str) -> tuple[int, int, int, int]:
    """Find one fixture's entire isolated coloured object, with a paper margin."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    ranges = {"red": (hue < 9) | (hue > 171), "blue": (hue > 99) & (hue < 131),
              "sun": (hue > 9) & (hue < 41)}
    mask = (ranges[colour] & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 40)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise ValueError(f"Fixture does not contain the expected {colour} object")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # JPEG edge speckles can share another object's hue. Keep the complete main
    # outline; a sun additionally includes all substantial disconnected rays.
    ids = [largest]
    if colour == "sun":
        minimum = max(8, stats[largest, cv2.CC_STAT_AREA] * .005)
        ids = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= minimum]
    ys, xs = np.where(np.isin(labels, ids))
    if not len(xs):
        raise ValueError(f"Fixture does not contain the expected {colour} object")
    box = (int(xs.min()) - 4, int(ys.min()) - 4, int(xs.max()) + 5, int(ys.max()) + 5)
    x0, y0, x1, y1 = box
    h, w = image.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Object touches a frame edge; refusing to crop the fixture")
    return box


def relocate(image: np.ndarray, boxes: list[tuple], centres: list[tuple] | None) -> tuple[np.ndarray, dict]:
    """Erase isolated source rectangles and optionally paste whole unchanged crops."""
    result = image.copy()
    h, w = result.shape[:2]
    paper = np.median(image[:max(1, h // 20)], axis=(0, 1)).astype(np.uint8)
    patches = []
    for x0, y0, x1, y1 in boxes:
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("Source rectangle would crop a subject")
        patches.append(image[y0:y1, x0:x1].copy())
        result[y0:y1, x0:x1] = paper
    destinations = []
    if centres is not None:
        if len(centres) != len(boxes):
            raise ValueError("Each source must have one destination")
        occupied = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)[:, :, 1] > 60
        for patch, (cx, cy) in zip(patches, centres, strict=True):
            ph, pw = patch.shape[:2]
            x0, y0 = round(cx - pw / 2), round(cy - ph / 2)
            x1, y1 = x0 + pw, y0 + ph
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError("Destination would crop a subject")
            if occupied[y0:y1, x0:x1].any():
                raise ValueError("Destination would cover another subject")
            result[y0:y1, x0:x1] = patch
            occupied[y0:y1, x0:x1] = True
            destinations.append([x0, y0, x1, y1])
    return result, {"source_rectangles_xyxy": boxes, "destination_rectangles_xyxy": destinations,
                    "paper_bgr": paper.tolist(), "preserves_crop_pixels": True}


def perturb(image: np.ndarray, case_id: str) -> tuple[np.ndarray, dict]:
    if case_id == "remove-square":
        output, details = relocate(image, [colour_bbox(image, "red")], None)
    elif case_id == "swap-square-circle":
        boxes = [colour_bbox(image, "red"), colour_bbox(image, "blue")]
        centres = [((a + c) / 2, (b + d) / 2) for a, b, c, d in reversed(boxes)]
        output, details = relocate(image, boxes, centres)
    elif case_id == "move-sun":
        box = colour_bbox(image, "sun")
        output, details = relocate(image, [box], [((box[0] + box[2]) / 2, image.shape[0] * .78)])
    else:
        raise ValueError(f"Unknown perturbation {case_id}")
    return output, {"operation": case_id, **details}


def prepare_cases(config, out: Path, selected: list[str], comparison: bool = False) -> list[dict]:
    """Persist exact preprocessed inputs before any checker call; sources stay untouched."""
    cached = {}
    rows = []
    for case_id, source_name, sid, expected_content, expected_style, generated_name in CASES:
        if case_id not in selected:
            continue
        source = ROOT / FIXTURES / "originals" / source_name
        if source_name not in cached:
            result, data = prepare_model_input(
                source, max_input_side=config.pre_max_input_side,
                frame_detect=config.pre_frame_detect, border_trim_ratio=config.pre_border_trim_ratio,
                crop_ratios=config.pre_crop_ratios, min_area_ratio=config.pre_min_area_ratio,
                max_area_ratio=config.pre_max_area_ratio, approx_epsilon_ratio=config.pre_approx_epsilon_ratio,
            )
            cached[source_name] = (result, data)
        result, original = cached[source_name]
        transform = None
        candidate_source = None
        if generated_name:
            candidate_source = ROOT / generated_name
            candidate = candidate_source.read_bytes()
            decode_image_bytes(candidate, str(candidate_source))
        elif case_id == "identity":
            candidate = original
        elif case_id in ("tone-only", "unrelated"):
            fn = tone_shift if case_id == "tone-only" else unrelated_pattern
            candidate = encode_png(fn(result.image))
            transform = {"operation": case_id, "version": 1}
        else:
            transformed, transform = perturb(result.image, case_id)
            candidate = encode_png(transformed)
        folder = out / case_id
        folder.mkdir()
        (folder / "original.png").write_bytes(original)
        (folder / "candidate.png").write_bytes(candidate)
        rows.append({
            "case_id": case_id, "style_id": sid, "dataset_role": "development",
            "content_policy": dict(CONTENT_POLICY),
            "source_path": str(source), "source_sha256": sha256(source.read_bytes()),
            "candidate_source_path": str(candidate_source) if candidate_source else None,
            "candidate_source_sha256": sha256(candidate) if candidate_source else None,
            "original_path": f"{case_id}/original.png", "original_sha256": sha256(original),
            "candidate_path": f"{case_id}/candidate.png", "candidate_sha256": sha256(candidate),
            "preprocess": {"crop_used": result.crop_used, "corners": result.corners,
                           "source_size": result.source_size, "output_size": result.output_size},
            "expected_content": expected_content, "expected_style": expected_style,
            "known_reject": expected_content is False or expected_style == "mismatch",
            "label_note": NOTES[case_id], "transform": transform,
            "visual_review_required": case_id in ("remove-square", "swap-square-circle", "move-sun"),
        })
        if comparison:
            board, metadata = build_comparison_board(original, candidate)
            (folder / "board.png").write_bytes(board)
            rows[-1]["comparison_board"] = {**metadata, "path": f"{case_id}/board.png"}
    return rows


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("status") == "checked"]
    axes = {}
    for name, expected_key, actual_key in (
        ("content", "expected_content", "content_preserved"),
        ("style", "expected_style", "style_match"),
    ):
        labelled = [r for r in rows if r[expected_key] is not None]
        scored = [r for r in valid if r[expected_key] is not None]
        correct = sum(r[expected_key] == r["actual"][actual_key] for r in scored)
        axes[name] = {"labelled": len(labelled), "evaluated": len(scored), "correct": correct,
                      "accuracy": correct / len(scored) if scored else None,
                      "abstained": sum(r["actual"][actual_key] in (None, "uncertain") for r in scored),
                      "errors": sum(r.get("status") == "error" for r in labelled)}
    rejects = [r for r in rows if r["known_reject"]]
    checked_rejects = [r for r in valid if r["known_reject"]]
    false_accepts = sum(r["actual"]["passed"] is True for r in checked_rejects)
    return {"total": len(rows), "checked": len(valid),
            "errors": sum(r.get("status") == "error" for r in rows),
            **axes, "known_reject": {"total": len(rejects), "evaluated": len(checked_rejects),
                                     "false_accepts": false_accepts,
                                     "errors": sum(r.get("status") == "error" for r in rejects)}}


def markdown(report: dict) -> str:
    summary = report["summary"]
    lines = ["# Checker development calibration", "",
             "Development fixtures only; this is not independent quality acceptance.",
             f"Content policy: `{report['content_policy']['version']}`.",
             report["content_policy"]["definition"],
             "Current labels follow this policy; historical reports and model responses are not relabelled.",
             "House style match is an engineering coarse-check label. Null style labels are unscored.",
             "Technical errors are reported separately, never counted as quality rejections.", "",
             f"Model: `{report['checker']['model']}`; mode: `{report['checker']['mode']}`.",
             f"Comparison checker version: `{report['checker'].get('comparison_version', 'not applicable')}`.", "",
             "| Axis | Labelled | Evaluated | Correct | Accuracy | Abstained | Errors |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for axis in ("content", "style"):
        s = summary[axis]
        rate = f"{s['accuracy']:.1%}" if s["accuracy"] is not None else "N/A"
        lines.append(f"| {axis} | {s['labelled']} | {s['evaluated']} | {s['correct']} | {rate} | {s['abstained']} | {s['errors']} |")
    reject = summary["known_reject"]
    lines += ["", f"Known rejections: {reject['false_accepts']} false acceptances / {reject['evaluated']} evaluated; "
              f"{reject['errors']} technical errors; {reject['total']} labelled.", "",
              "| Case | Expected theme association | Expected style | Actual theme association | Actual style | Status |",
              "|---|---|---|---|---|---|"]
    for row in report["cases"]:
        actual = row.get("actual", {})
        lines.append(f"| {row['case_id']} | {row['expected_content']} | {row['expected_style']} | "
                     f"{actual.get('content_preserved', '—')} | {actual.get('style_match', '—')} | {row['status']} |")
    lines += ["", "Inspect each case's original.png, candidate.png and result.json for full evidence.",
              "New content perturbations require visual review before their results are used as calibration evidence.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New directory; existing paths are never overwritten")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", choices=("two_stage", "pairwise", "comparison"), default=None)
    parser.add_argument("--num-gpu", type=int, default=0)
    parser.add_argument("--num-predict", type=int, default=None, help="Optional positive response-token limit")
    parser.add_argument("--cases", nargs="+", help="Case ids, separated by spaces or commas")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.num_predict is not None and args.num_predict < 1:
        parser.error("--num-predict must be positive")
    known = [c[0] for c in CASES]
    selected = [part for item in args.cases or known for part in item.split(",")]
    if len(set(selected)) != len(selected) or set(selected) - set(known):
        parser.error(f"Cases must be unique and selected from: {', '.join(known)}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = load_app_config(ROOT)
    checker_config = {"model": args.model or config.checker_model,
                      "mode": args.mode or config.checker_mode,
                      "options": {**config.checker_options, "num_gpu": args.num_gpu},
                      "api": config.checker_api, "control": config.checker_control}
    if checker_config["mode"] == "comparison":
        checker_config["comparison_version"] = COMPARISON_VERSION
    if args.num_predict is not None:
        checker_config["options"]["num_predict"] = args.num_predict
    _, check_template = config.load_prompts()
    manifest = {"manifest_version": 3, "dataset_role": "development",
                "content_policy": dict(CONTENT_POLICY),
                "checker": checker_config, "check_prompt_sha256": sha256(check_template.encode()),
                "checker_code_sha256": sha256((ROOT / "app/checker.py").read_bytes()),
                "style_config_sha256": sha256(config.style_config_path.read_bytes()),
                "app_config_sha256": sha256((ROOT / "configs/app.json").read_bytes()),
                "cases": prepare_cases(config, out, selected, checker_config["mode"] == "comparison")}
    write_json(out / "manifest.json", manifest)
    checker = None if args.prepare_only else RecordingChecker(
        base_url=config.checker_base_url, api=config.checker_api, model=checker_config["model"],
        mode=checker_config["mode"], options=checker_config["options"], control=config.checker_control,
        connect_timeout=config.checker_connect_timeout, request_timeout=config.checker_request_timeout,
        keep_alive=config.checker_keep_alive,
    )
    registry = config.load_styles()
    rows = []
    for case in manifest["cases"]:
        row = {**case, "status": "prepared", "seconds": 0.0}
        if checker is not None:
            checker.calls.clear()
            style = registry.resolve(case["style_id"])
            prompt = check_template.replace("{style_name}", style.name).replace(
                "{style_features}", style.style_features).replace("{preservation_rules}", style.preservation_rules)
            row["prompt"] = prompt
            started = time.monotonic()
            try:
                result = checker.check_pair(
                    original_bytes=(out / case["original_path"]).read_bytes(),
                    candidate_bytes=(out / case["candidate_path"]).read_bytes(), prompt=prompt,
                    style_name=style.name, style_features=style.style_features,
                    preservation_rules=style.preservation_rules,
                )
                actual = result.to_dict()
                if checker_config["mode"] == "comparison":
                    actual_board = actual.get("comparison_board", {})
                    if actual_board.get("sha256") != case["comparison_board"]["sha256"]:
                        raise CheckTechnicalError("Saved comparison board differs from checker evidence", stage="CHECKING")
                validated = validate_check_payload(actual)
                actual["passed"] = validated.passed
                row.update(status="checked", actual=actual, raw_text=result.raw_text)
            except Exception as exc:  # Record protocol/service/unexpected errors separately from rejection.
                row.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            row["seconds"] = round(time.monotonic() - started, 3)
            row["model_responses"] = list(checker.calls)
        write_json(out / case["case_id"] / "result.json", row)
        rows.append(row)
        print(json.dumps({"case": case["case_id"], "status": row["status"],
                          "seconds": row["seconds"], "actual": row.get("actual"),
                          "error": row.get("error")}, ensure_ascii=True), flush=True)
    report = {**manifest, "cases": rows, "summary": summarize(rows)}
    write_json(out / "report.json", report)
    with (out / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown(report))
    return 1 if report["summary"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

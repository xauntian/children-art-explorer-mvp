"""One fresh frame or photo and its three requested delivery images, with wall time."""

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_app_config
from app.errors import ConfigError
from app.generator import ComfyClient
from app.model_warmup import warmup_generation
from app.phone import PhoneController, load_phone_config
from app.phone_capture import PhoneCapture
from app.phone_runtime import PhoneRunLock, reconcile_phone_sessions
from app.preprocess import decode_image_bytes
from app.storage import atomic_write_bytes, atomic_write_json, utc_now_iso


class NativePhotoValidation(PhoneCapture):
    def capture(self):
        return self.capture_photo(reason="operator_requested_original_photo")


def report(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def main():
    config = load_phone_config(project_root=ROOT)
    streaming = config.get("capture_backend") == "camera_stream"
    camera = None if streaming else NativePhotoValidation(ROOT, **config["phone"])
    controller = PhoneController(config, camera=camera)
    with PhoneRunLock(config["capture_runs_dir"]):
        recovered = reconcile_phone_sessions(config["capture_runs_dir"], controller.generation_store)
        if any(item["blocking"] for item in recovered):
            raise ConfigError("Existing active phone generation blocks this validation")
        active = controller.generation_store.find_active()
        if active is not None:
            raise ConfigError("An existing active generation blocks this validation")
        startup_started = time.perf_counter()
        warmup = warmup_generation(ROOT) if config.get("generation_warmup") else None
        try:
            if streaming:
                controller.camera.start()
            runtime = ComfyClient(load_app_config(ROOT).comfy_base_url).system_stats().get("system", {})
            startup = {
                "total_sec": round(time.perf_counter() - startup_started, 4),
                "model_warmup": warmup,
                "camera_stream_sec": getattr(controller.camera, "startup_sec", None),
                "inference_runtime": runtime,
                "included_in_trigger_total": False,
            }
            return deliver(controller, startup)
        finally:
            if streaming:
                controller.camera.close()


def deliver(controller, startup):
        started_at = utc_now_iso()
        started = time.perf_counter()
        session = controller.prepare("auto")
        destination = ROOT / "runs" / "deliveries" / session.directory.name
        destination.mkdir(parents=True, exist_ok=False)
        original_name = session.meta["capture"]["raw_file"]
        for source, target in (
            (original_name, original_name),
            ("artwork.png", "rectified.png"),
        ):
            atomic_write_bytes(destination / target, (session.directory / source).read_bytes())
        manifest = {
            "started_at": started_at,
            "timing_scope": "fresh_capture_through_three_delivery_images_saved",
            "model_preheated": startup["model_warmup"] is not None,
            "startup": startup,
            "state": "INPUT_READY",
            "session_id": session.directory.name,
            "phone_session": str(session.directory),
            "capture": session.meta["capture"],
            "extraction": session.meta["extraction"],
            "files": {"original": original_name, "rectified": "rectified.png"},
        }
        atomic_write_json(destination / "manifest.json", manifest)
        report({"stage": "INPUT_READY", "delivery": str(destination), "timings": session.meta["timings"]})
        result = controller.generate(session)
        manifest.update(
            state=result["state"],
            generation_job_id=result["generation_job_id"],
            generation=result.get("generation"),
            timings=result.get("timings"),
            error=result.get("error"),
        )
        if result["state"] == "COMPLETED":
            style = result["generation"]["style_id"]
            generated = Path(result["output_path"]).read_bytes()
            filename = "ai-" + style + ".png"
            atomic_write_bytes(destination / filename, generated)
            manifest["files"]["generated"] = filename
            job = controller.generation_store.load(result["generation_job_id"])
            if job.meta.get("review") is not None:
                manifest["review"] = job.meta["review"]
            attempts = job.meta.get("attempts") or []
            quality = attempts[-1].get("check_result") if attempts else None
            if quality is not None:
                manifest["quality"] = quality
        manifest["sha256"] = {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in manifest["files"].values()
        }
        manifest["image_sizes"] = {}
        for name in manifest["files"].values():
            image = decode_image_bytes((destination / name).read_bytes(), name)
            manifest["image_sizes"][name] = [int(image.shape[1]), int(image.shape[0])]
        manifest["completed_at"] = utc_now_iso()
        manifest["actual_total_sec"] = round(time.perf_counter() - started, 4)
        generation_sec = (result.get("timings") or {}).get("generation_sec")
        manifest["generation_budget"] = {
            "target_sec": 10.0,
            "actual_sec": generation_sec,
            "met": generation_sec is not None and generation_sec <= 10.0,
        }
        atomic_write_json(destination / "manifest.json", manifest)
        report({"stage": result["state"], "delivery": str(destination), "files": manifest["files"], "generation": result.get("generation"), "timings": manifest["timings"], "actual_total_sec": manifest["actual_total_sec"], "generation_budget": manifest["generation_budget"]})
        return 0 if result["state"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

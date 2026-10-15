"""ComfyUI 本地生成适配。

要点：
1. 调用前校验工作流确实接收原图（LoadImage → VAEEncode → KSampler.latent_image），
   不得把文字生图当作图生图。
2. 通过 /upload/image 上传预处理图，通过 /prompt 提交，轮询 /history/{prompt_id}
   并用 /view 取回候选图。
3. 技术错误（服务不可用、超时、输出缺失或损坏）直接抛出，不静默改走其他服务。
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from .errors import (
    ConfigError,
    GenerationError,
    ServiceUnavailableError,
    TimeoutError_,
)

PLACEHOLDER_PREFIX = "__"
PLACEHOLDER_SUFFIX__ = "__"


def _is_placeholder(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(PLACEHOLDER_PREFIX)
        and value.endswith(PLACEHOLDER_SUFFIX__)
        and len(value) > 4
    )


def load_workflow(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"工作流文件不存在: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"工作流不是合法 JSON: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"工作流顶层必须是 JSON 对象: {path}")
    nodes = {k: v for k, v in data.items() if not k.startswith("_")}
    if not nodes:
        raise ConfigError(f"工作流没有任何节点: {path}")
    for node_id, node in nodes.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise ConfigError(f"工作流节点 {node_id} 缺少 class_type")
        if not isinstance(node.get("inputs", {}), dict):
            raise ConfigError(f"工作流节点 {node_id} 的 inputs 必须是 JSON 对象")
    return nodes


def _outputs_of(node: dict) -> list[tuple[str, int]]:
    """收集节点输入中引用的 (源节点 id, 输出槽位)。"""
    refs: list[tuple[str, int]] = []
    for value in node.get("inputs", {}).values():
        if (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
        ):
            refs.append((value[0], value[1]))
    return refs


def validate_image_input_workflow(nodes: dict) -> dict:
    """确认工作流真正把原图送进采样器。

    检查链路：LoadImage 存在，且其 IMAGE 输出经 VAEEncode 进入 KSampler 的
    latent_image 输入。缺少任一环节都视为「不是图生图」，直接拒绝。
    """
    load_image_ids = [nid for nid, n in nodes.items() if n["class_type"] == "LoadImage"]
    if not load_image_ids:
        raise ConfigError("工作流缺少 LoadImage 节点，无法接收原图输入")

    vae_encode_ids = [nid for nid, n in nodes.items() if n["class_type"] == "VAEEncode"]
    if not vae_encode_ids:
        raise ConfigError("工作流缺少 VAEEncode 节点，原图未进入 latent，属于文字生图")

    ksampler_ids = [nid for nid, n in nodes.items() if n["class_type"] in ("KSampler", "KSamplerAdvanced")]
    if not ksampler_ids:
        raise ConfigError("工作流缺少 KSampler 节点")

    # 找出哪些节点消费了 LoadImage 的 IMAGE 输出
    image_consumers: set[str] = set()
    for nid, node in nodes.items():
        for ref_id, _slot in _outputs_of(node):
            if ref_id in load_image_ids:
                image_consumers.add(nid)
    if not image_consumers:
        raise ConfigError("LoadImage 的输出没有被任何节点使用，原图未参与生成")

    # 沿「输入来源」反向回溯：VAEEncode 的像素必须最终来自 LoadImage
    # （中间允许有 ImageScale / ImageInvert 等图像处理节点）
    def derives_from(start: str, origins: set[str]) -> bool:
        stack = [start]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in origins:
                return True
            node = nodes.get(current)
            if node is None:
                continue
            stack.extend(ref_id for ref_id, _slot in _outputs_of(node))
        return False

    if not any(derives_from(enc_id, set(load_image_ids)) for enc_id in vae_encode_ids):
        raise ConfigError("原图未连接到 VAEEncode，工作流不是图生图")

    latent_from_encode = False
    for kid in ksampler_ids:
        latent = nodes[kid]["inputs"].get("latent_image")
        if (
            isinstance(latent, list)
            and len(latent) == 2
            and isinstance(latent[0], str)
            and latent[0] in vae_encode_ids
        ):
            latent_from_encode = True
            break
    if not latent_from_encode:
        raise ConfigError("KSampler 的 latent_image 不来自 VAEEncode，原图未作为生成起点")

    save_ids = [nid for nid, n in nodes.items() if n["class_type"] == "SaveImage"]
    if not save_ids:
        raise ConfigError("工作流缺少 SaveImage 节点，无法取回候选图")

    return {
        "load_image": load_image_ids[0],
        "vae_encode": vae_encode_ids[0],
        "ksampler": [nid for nid in ksampler_ids],
        "save_image": save_ids[0],
    }


def apply_placeholders(nodes: dict, values: dict[str, Any]) -> dict:
    """把占位符替换为实际值；替换后仍存在未解析占位符时报错。"""
    resolved = json.loads(json.dumps(nodes))
    used: set[str] = set()

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        if _is_placeholder(obj):
            key = obj[2:-2]
            if key not in values:
                raise ConfigError(f"工作流占位符 {obj} 没有提供对应配置值")
            used.add(key)
            return values[key]
        return obj

    resolved = walk(resolved)

    def find_leftover(obj: Any) -> str | None:
        if isinstance(obj, dict):
            for v in obj.values():
                found = find_leftover(v)
                if found:
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = find_leftover(v)
                if found:
                    return found
        elif _is_placeholder(obj):
            return obj
        return None

    leftover = find_leftover(resolved)
    if leftover:
        raise ConfigError(f"工作流替换后仍存在未解析占位符: {leftover}")
    return resolved


class ComfyClient:
    """ComfyUI HTTP API 客户端。"""

    def __init__(
        self,
        base_url: str,
        connect_timeout: float = 5.0,
        request_timeout: float = 30.0,
        job_timeout: float = 600.0,
        poll_interval: float = 1.5,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.job_timeout = job_timeout
        self.poll_interval = poll_interval
        self.session = session or requests.Session()

    # ---- 健康检查 ----
    def system_stats(self) -> dict:
        url = f"{self.base_url}/system_stats"
        try:
            response = self.session.get(url, timeout=self.connect_timeout)
        except requests.RequestException as exc:
            raise ServiceUnavailableError(f"ComfyUI 不可用: {url} ({exc})", stage="GENERATING") from exc
        if response.status_code != 200:
            raise ServiceUnavailableError(
                f"ComfyUI /system_stats 返回 {response.status_code}", stage="GENERATING"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ServiceUnavailableError("ComfyUI /system_stats 响应不是 JSON", stage="GENERATING") from exc

    def available_checkpoints(self) -> list[str]:
        url = f"{self.base_url}/object_info/CheckpointLoaderSimple"
        try:
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ServiceUnavailableError(
                f"无法读取 ComfyUI 模型列表: {url} ({exc})", stage="GENERATING"
            ) from exc
        try:
            options = payload["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        except (KeyError, TypeError) as exc:
            raise ServiceUnavailableError("ComfyUI 模型列表结构异常", stage="GENERATING") from exc
        return list(options)

    def upload_image(self, data: bytes, filename: str, subfolder: str = "") -> dict:
        url = f"{self.base_url}/upload/image"
        files = {"image": (filename, data, "image/png")}
        form = {"overwrite": "true", "type": "input"}
        if subfolder:
            form["subfolder"] = subfolder
        try:
            response = self.session.post(
                url, files=files, data=form, timeout=self.request_timeout
            )
        except requests.RequestException as exc:
            raise ServiceUnavailableError(
                f"上传原图到 ComfyUI 失败: {url} ({exc})", stage="GENERATING"
            ) from exc
        if response.status_code != 200:
            raise GenerationError(
                f"ComfyUI /upload/image 返回 {response.status_code}: {response.text[:200]}",
                stage="GENERATING",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GenerationError("ComfyUI /upload/image 响应不是 JSON", stage="GENERATING") from exc
        if "name" not in payload:
            raise GenerationError("ComfyUI /upload/image 响应缺少 name 字段", stage="GENERATING")
        return payload

    def queue_prompt(self, workflow: dict, client_id: str) -> str:
        url = f"{self.base_url}/prompt"
        body = {"prompt": workflow, "client_id": client_id}
        try:
            response = self.session.post(url, json=body, timeout=self.request_timeout)
        except requests.RequestException as exc:
            raise ServiceUnavailableError(
                f"提交生成任务失败: {url} ({exc})", stage="GENERATING"
            ) from exc
        if response.status_code != 200:
            raise GenerationError(
                f"ComfyUI /prompt 返回 {response.status_code}: {response.text[:500]}",
                stage="GENERATING",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GenerationError("ComfyUI /prompt 响应不是 JSON", stage="GENERATING") from exc
        prompt_id = payload.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise GenerationError(
                f"ComfyUI /prompt 响应缺少 prompt_id: {str(payload)[:300]}", stage="GENERATING"
            )
        return prompt_id

    def history(self, prompt_id: str) -> dict:
        url = f"{self.base_url}/history/{prompt_id}"
        try:
            response = self.session.get(url, timeout=self.request_timeout)
        except requests.RequestException as exc:
            raise ServiceUnavailableError(
                f"查询生成进度失败: {url} ({exc})", stage="GENERATING"
            ) from exc
        if response.status_code != 200:
            raise ServiceUnavailableError(
                f"ComfyUI /history 返回 {response.status_code}", stage="GENERATING"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ServiceUnavailableError("ComfyUI /history 响应不是 JSON", stage="GENERATING") from exc

    def wait_for(self, prompt_id: str, job_timeout: float | None = None) -> dict:
        """轮询直到任务结束；超时抛出技术错误。"""
        deadline = time.monotonic() + (job_timeout if job_timeout is not None else self.job_timeout)
        while True:
            record = self.history(prompt_id)
            entry = record.get(prompt_id) if isinstance(record, dict) else None
            if entry:
                status = entry.get("status") or {}
                if status.get("completed") is True or status.get("status_str") in ("success", "error"):
                    return entry
                if status.get("status_str") == "error":
                    return entry
            if time.monotonic() > deadline:
                raise TimeoutError_(
                    f"ComfyUI 生成超时（{job_timeout or self.job_timeout}s，prompt_id={prompt_id}）",
                    stage="GENERATING",
                )
            time.sleep(self.poll_interval)

    def fetch_image(self, item: dict) -> bytes:
        query = urlencode(
            {
                "filename": item["filename"],
                "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "output"),
            }
        )
        url = f"{self.base_url}/view?{query}"
        try:
            response = self.session.get(url, timeout=self.request_timeout)
        except requests.RequestException as exc:
            raise ServiceUnavailableError(
                f"取回生成图像失败: {url} ({exc})", stage="GENERATING"
            ) from exc
        if response.status_code != 200:
            raise GenerationError(
                f"ComfyUI /view 返回 {response.status_code}（filename={item.get('filename')}）",
                stage="GENERATING",
            )
        if not response.content:
            raise GenerationError(
                f"ComfyUI /view 返回空内容（filename={item.get('filename')}）", stage="GENERATING"
            )
        return response.content

    def interrupt(self) -> None:
        try:
            self.session.post(f"{self.base_url}/interrupt", timeout=self.connect_timeout)
        except requests.RequestException:
            pass


class ComfyGenerator:
    """把工作流、Prompt 与参数组装成一次生成调用。"""

    PROMPT_VERSION_FIELDS = ("style_name", "style_features", "preservation_rules")

    def __init__(
        self,
        client: ComfyClient,
        workflow_nodes: dict,
        checkpoint: str,
        negative_prompt: str,
        controlnet: str | None = None,
        controlnet_strength: float = 0.55,
        filename_prefix: str = "amcc",
        unet_name: str | None = None,
        text_encoder: str | None = None,
        vae_name: str | None = None,
        shift: float = 3.0,
        lora_name: str | None = None,
        lora_strength: float = 1.0,
    ) -> None:
        self.client = client
        self.workflow_nodes = workflow_nodes
        self.checkpoint = checkpoint
        self.negative_prompt = negative_prompt
        self.controlnet = controlnet
        self.controlnet_strength = controlnet_strength
        self.filename_prefix = filename_prefix
        # 编辑模型（Qwen-Image-Edit 等）使用独立的 unet / 文本编码器 / VAE
        self.unet_name = unet_name
        self.text_encoder = text_encoder
        self.vae_name = vae_name
        self.shift = shift
        self.lora_name = lora_name
        self.lora_strength = lora_strength
        self.binding = validate_image_input_workflow(workflow_nodes)
        self.client_id = str(uuid.uuid4())

    # ---- 校验 ----
    def _model_options(self, class_type: str, field: str) -> list[str]:
        try:
            response = self.client.session.get(
                f"{self.client.base_url}/object_info/{class_type}",
                timeout=self.client.request_timeout,
            )
            response.raise_for_status()
            return list(response.json()[class_type]["input"]["required"][field][0])
        except Exception as exc:  # noqa: BLE001 - 仅用于给出可读错误
            raise ConfigError(
                f"无法读取 ComfyUI 的 {class_type} 模型列表（{field}）: {exc}"
            ) from exc

    def uses_unet_loader(self) -> bool:
        return any(
            n["class_type"] in ("UnetLoaderGGUF", "UNETLoader")
            for n in self.workflow_nodes.values()
        )

    def verify_checkpoint(self) -> None:
        """校验工作流需要的模型确实存在于 ComfyUI 中。"""
        if self.uses_unet_loader():
            self._verify_unet_model()
            return
        available = self.client.available_checkpoints()
        if self.checkpoint not in available:
            raise ConfigError(
                "ComfyUI 中没有找到配置的生成模型 "
                f"'{self.checkpoint}'。当前可用: {available or '（无）'}"
            )

    def _verify_unet_model(self) -> None:
        if not self.unet_name:
            raise ConfigError(
                "工作流使用 UnetLoaderGGUF，但 app.comfy.unet_name 未配置"
            )
        class_type = (
            "UnetLoaderGGUF"
            if any(n["class_type"] == "UnetLoaderGGUF" for n in self.workflow_nodes.values())
            else "UNETLoader"
        )
        field = "unet_name"
        options = self._model_options(class_type, field)
        if self.unet_name not in options:
            hint = ""
            if class_type == "UnetLoaderGGUF" and not options:
                hint = "（ComfyUI-GGUF 自定义节点未加载，请检查 custom_nodes/ComfyUI-GGUF）"
            raise ConfigError(
                f"ComfyUI 中没有找到配置的扩散模型 '{self.unet_name}'{hint}。"
                f"当前可用: {options or '（无）'}"
            )

        if self.text_encoder:
            options = self._model_options("CLIPLoaderGGUF", "clip_name")
            if self.text_encoder not in options:
                raise ConfigError(
                    f"ComfyUI 中没有找到配置的文本编码器 '{self.text_encoder}'。"
                    f"当前可用: {options or '（无）'}"
                )
        if self.vae_name:
            options = self._model_options("VAELoader", "vae_name")
            if self.vae_name not in options:
                raise ConfigError(
                    f"ComfyUI 中没有找到配置的 VAE '{self.vae_name}'。"
                    f"当前可用: {options or '（无）'}"
                )
        if self.lora_name:
            options = self._model_options("LoraLoaderModelOnly", "lora_name")
            if self.lora_name not in options:
                raise ConfigError(
                    f"ComfyUI 中没有找到配置的 LoRA '{self.lora_name}'。"
                    f"当前可用: {options or '（无）'}"
                )

    def verify_controlnet(self) -> None:
        if not self.controlnet:
            return
        options = self._model_options("ControlNetLoader", "control_net_name")
        if self.controlnet not in options:
            raise ConfigError(
                f"ComfyUI 中没有找到配置的 ControlNet '{self.controlnet}'。当前可用: {options or '（无）'}"
            )

    # ---- 生成 ----
    def build_workflow(
        self,
        input_image_name: str,
        positive_prompt: str,
        width: int,
        height: int,
        seed: int,
        params: dict,
        filename_prefix: str | None = None,
    ) -> dict:
        nodes = json.loads(json.dumps(self.workflow_nodes))
        needs_controlnet = any(n["class_type"] == "ControlNetLoader" for n in nodes.values())

        values: dict[str, Any] = {
            "checkpoint": self.checkpoint,
            "unet_name": self.unet_name,
            "text_encoder": self.text_encoder,
            "vae_name": self.vae_name,
            "shift": float(self.shift),
            "input_image": input_image_name,
            "positive_prompt": positive_prompt,
            "negative_prompt": self.negative_prompt,
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "steps": int(params["steps"]),
            "cfg": float(params["cfg"]),
            "sampler_name": params["sampler_name"],
            "scheduler": params["scheduler"],
            "denoise": float(params.get("denoise", 1.0)),
            "filename_prefix": filename_prefix or self.filename_prefix,
            "controlnet": self.controlnet,
            "controlnet_strength": float(self.controlnet_strength),
            "lora_name": self.lora_name,
            "lora_strength": float(self.lora_strength),
        }
        if needs_controlnet and not self.controlnet:
            raise ConfigError(
                "工作流包含 ControlNet 节点，但 app.comfy.controlnet 未配置"
            )

        resolved = apply_placeholders(nodes, values)

        # 工作流中的可旁路节点：未配置时把下游引用直接接到其上游，保持链路有效
        bypass_node = None
        if not self.lora_name:
            bypass_node = next(
                (nid for nid, n in resolved.items() if n["class_type"] == "LoraLoaderModelOnly"),
                None,
            )
        if bypass_node is not None:
            upstream = resolved[bypass_node]["inputs"].get("model")
            if isinstance(upstream, list) and len(upstream) == 2:
                for node in resolved.values():
                    for key, value in list(node["inputs"].items()):
                        if (
                            isinstance(value, list)
                            and len(value) == 2
                            and value[0] == bypass_node
                        ):
                            node["inputs"][key] = list(upstream)
                resolved.pop(bypass_node, None)

        validate_image_input_workflow(resolved)
        return resolved

    def generate(
        self,
        input_bytes: bytes,
        input_filename: str,
        positive_prompt: str,
        width: int,
        height: int,
        seed: int,
        params: dict,
        filename_prefix: str | None = None,
        job_timeout: float | None = None,
    ) -> tuple[bytes, str]:
        """上传原图并生成一次候选图，返回 (图像字节, prompt_id)。"""
        upload = self.client.upload_image(input_bytes, input_filename)
        stored_name = upload["name"]
        if upload.get("subfolder"):
            stored_name = f"{upload['subfolder']}/{stored_name}"

        workflow = self.build_workflow(
            input_image_name=stored_name,
            positive_prompt=positive_prompt,
            width=width,
            height=height,
            seed=seed,
            params=params,
            filename_prefix=filename_prefix,
        )
        prompt_id = self.client.queue_prompt(workflow, self.client_id)
        entry = self.client.wait_for(prompt_id, job_timeout=job_timeout)

        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            raise GenerationError(
                f"ComfyUI 执行失败: {json.dumps(status.get('messages', []), ensure_ascii=False)[:500]}",
                stage="GENERATING",
            )

        outputs = entry.get("outputs") or {}
        save_id = self.binding["save_image"]
        items = (outputs.get(save_id) or {}).get("images") or []
        if not items:
            # 保存节点 id 在注入后不变，但为稳妥起见回退到任意 SaveImage 输出
            for node_outputs in outputs.values():
                if isinstance(node_outputs, dict) and node_outputs.get("images"):
                    items = node_outputs["images"]
                    break
        if not items:
            raise GenerationError(
                f"ComfyUI 未返回生成图像（prompt_id={prompt_id}）", stage="GENERATING"
            )
        return self.client.fetch_image(items[0]), prompt_id

"""ComfyUI 本地生成适配。

要点：
1. 调用前校验工作流确实接收原图（原图编码进入采样 latent 或原生参考条件），
   不得把文字生图当作图生图。
2. 通过 /upload/image 上传预处理图，通过 /prompt 提交，轮询 /history/{prompt_id}
   并用 /view 取回候选图。
3. 技术错误（服务不可用、超时、输出缺失或损坏）直接抛出，不静默改走其他服务。
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from urllib3.util import Timeout as HTTPTimeout

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
    def collect(value):
        if (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
        ):
            refs.append((value[0], value[1]))
        elif isinstance(value, dict):
            for nested in value.values():
                collect(nested)

    for value in node.get("inputs", {}).values():
        collect(value)
    return refs


def validate_image_input_workflow(nodes: dict) -> dict:
    """确认工作流真正把原图送进采样器。

    KSampler/Advanced 要求原图编码作为 latent_image；SamplerCustomAdvanced
    另支持 FLUX.2 原生编辑：空 Flux2 latent + 实际 guider 正向条件消费原图的
    ReferenceLatent。游离的图像编码、reference 或采样节点不能证明输出使用原图。
    TextEncodeQwenImage21 内部编码参考图：正向 slot 0 与 latent slot 2 必须由
    同一节点提供；兼容键 vae_encode 在此路线指向该综合编码节点。
    """
    load_image_ids = [nid for nid, n in nodes.items() if n["class_type"] == "LoadImage"]
    if not load_image_ids:
        raise ConfigError("工作流缺少 LoadImage 节点，无法接收原图输入")

    vae_encode_ids = [nid for nid, n in nodes.items() if n["class_type"] == "VAEEncode"]
    qwen21_ids = [nid for nid, n in nodes.items() if n["class_type"] == "TextEncodeQwenImage21"]
    if not vae_encode_ids and not qwen21_ids:
        raise ConfigError("工作流缺少 VAEEncode 节点，原图未进入 latent，属于文字生图")

    sampler_classes = ("KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced")
    ksampler_ids = [nid for nid, n in nodes.items() if n["class_type"] in sampler_classes]
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
    def ancestors(start: str) -> set[str]:
        stack = [start]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            node = nodes.get(current)
            if node is None:
                continue
            stack.extend(ref_id for ref_id, _slot in _outputs_of(node))
        return visited

    def image_origins(reference: Any) -> set[str]:
        """Trace the pixel input, not the encoder's unrelated VAE dependency."""
        if not (
            isinstance(reference, list) and len(reference) == 2
            and isinstance(reference[0], str) and isinstance(reference[1], int)
        ):
            return set()
        pending = [tuple(reference)]
        visited = set()
        origins = set()
        while pending:
            node_id, slot = pending.pop()
            if (node_id, slot) in visited:
                continue
            visited.add((node_id, slot))
            if node_id in load_image_ids:
                if slot == 0:  # LoadImage output 1 is a mask, not the original IMAGE.
                    origins.add(node_id)
                continue
            if node_id in nodes:
                pending.extend(_outputs_of(nodes[node_id]))
        return origins

    encoded_origins = {
        enc_id: image_origins(nodes[enc_id].get("inputs", {}).get("pixels"))
        for enc_id in vae_encode_ids
    }
    qwen21_origins = {}
    for encoder_id in qwen21_ids:
        inputs = nodes[encoder_id].get("inputs", {})
        vae = inputs.get("vae")
        valid_vae = (
            isinstance(vae, list) and len(vae) == 2 and isinstance(vae[0], str)
            and (nodes.get(vae[0], {}).get("class_type"), vae[1])
            in {("VAELoader", 0), ("CheckpointLoaderSimple", 2)}
        )
        images = inputs.get("images", {})
        references = [
            value for key, value in inputs.items()
            if key.startswith("images.image_") and key.removeprefix("images.image_") in {str(i) for i in range(1, 17)}
        ]
        if isinstance(images, dict):
            references.extend(value for key, value in images.items() if key in {f"image_{i}" for i in range(1, 17)})
        origins = set()
        if valid_vae:
            for reference in references:
                origins.update(image_origins(reference))
        qwen21_origins[encoder_id] = origins
    encoded_origins.update(qwen21_origins)
    if not any(encoded_origins.values()):
        raise ConfigError("原图未连接到 VAEEncode 或有效的 TextEncodeQwenImage21 图像参考，工作流不是图生图")

    save_ids = [nid for nid, n in nodes.items() if n["class_type"] == "SaveImage"]
    if not save_ids:
        raise ConfigError("工作流缺少 SaveImage 节点，无法取回候选图")
    output_ancestors = ancestors(save_ids[0])
    active_samplers = [nid for nid in ksampler_ids if nid in output_ancestors]
    if not active_samplers:
        raise ConfigError("SaveImage 未连接到 KSampler 或 SamplerCustomAdvanced，采样结果未用于输出")

    def original_encoder(reference: Any) -> str | None:
        if (
            isinstance(reference, list) and len(reference) == 2
            and isinstance(reference[0], str) and reference[1] == 0
            and reference[0] in vae_encode_ids
            and encoded_origins.get(reference[0])
        ):
            return reference[0]
        return None

    def positive_conditioning(sampler: dict):
        if sampler["class_type"] != "SamplerCustomAdvanced":
            return sampler["inputs"].get("positive")
        guider_ref = sampler["inputs"].get("guider")
        if not (isinstance(guider_ref, list) and len(guider_ref) == 2 and isinstance(guider_ref[0], str) and guider_ref[1] == 0):
            return None
        guider = nodes.get(guider_ref[0], {})
        key = {"BasicGuider": "conditioning", "CFGGuider": "positive"}.get(guider.get("class_type"))
        return guider.get("inputs", {}).get(key) if key else None

    def qwen21_encoder(sampler: dict) -> str | None:
        latent = sampler["inputs"].get("latent_image")
        if not (
            isinstance(latent, list) and len(latent) == 2
            and isinstance(latent[0], str) and latent[1] == 2
            and qwen21_origins.get(latent[0])
        ):
            return None
        encoder_id = latent[0]
        conditioning = positive_conditioning(sampler)
        if not (isinstance(conditioning, list) and len(conditioning) == 2 and isinstance(conditioning[0], str)):
            return None
        pending = [tuple(conditioning)]
        visited = set()
        while pending:
            node_id, slot = pending.pop()
            if (node_id, slot) in visited:
                continue
            visited.add((node_id, slot))
            if node_id == encoder_id:
                if slot == 0:
                    return encoder_id
                continue
            # Another combined encoder's output does not prove joint origin.
            if node_id in qwen21_ids:
                continue
            if node_id in nodes:
                pending.extend(_outputs_of(nodes[node_id]))
        return None

    def custom_reference_encoder(sampler: dict) -> str | None:
        """Native FLUX.2 edits begin with noise but consume a real reference."""
        latent = sampler["inputs"].get("latent_image")
        if not (
            isinstance(latent, list) and len(latent) == 2 and latent[1] == 0
            and nodes.get(latent[0], {}).get("class_type") == "EmptyFlux2LatentImage"
        ):
            return None
        guider_ref = sampler["inputs"].get("guider")
        if not (isinstance(guider_ref, list) and len(guider_ref) == 2 and guider_ref[1] == 0):
            return None
        guider = nodes.get(guider_ref[0], {})
        conditioning_key = {"BasicGuider": "conditioning", "CFGGuider": "positive"}.get(guider.get("class_type"))
        if conditioning_key is None:
            return None
        conditioning = guider.get("inputs", {}).get(conditioning_key)
        if not (isinstance(conditioning, list) and len(conditioning) == 2 and isinstance(conditioning[0], str)):
            return None
        consumed = ancestors(conditioning[0])
        for node_id, node in nodes.items():
            if node_id in consumed and node["class_type"] == "ReferenceLatent":
                encoder = original_encoder(node.get("inputs", {}).get("latent"))
                if encoder:
                    return encoder
        return None

    encoded_inputs = []
    for kid in active_samplers:
        sampler = nodes[kid]
        encoder = original_encoder(sampler["inputs"].get("latent_image"))
        if encoder is None:
            encoder = qwen21_encoder(sampler)
        if encoder is None and sampler["class_type"] == "SamplerCustomAdvanced":
            encoder = custom_reference_encoder(sampler)
        if encoder is None:
            raise ConfigError(
                f"{nodes[kid]['class_type']} 的 latent_image 不来自 VAEEncode 的原图编码，原图未作为生成起点"
                "或有效的原生图像参考条件"
            )
        encoded_inputs.append(encoder)

    binding = {
        "load_image": next(nid for nid in load_image_ids if nid in encoded_origins[encoded_inputs[0]]),
        "vae_encode": encoded_inputs[0],
        # The historical key is also consumed by cache-evidence tooling.
        "ksampler": active_samplers,
        "save_image": save_ids[0],
    }
    if encoded_inputs[0] in qwen21_ids:
        binding["vae_encode_type"] = "TextEncodeQwenImage21"
    return binding


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


@dataclass
class _GenerationDeadline:
    expires_at: float
    seconds: float
    on_prompt_id: Callable[[str], None] | None = None
    prompt_id: str | None = None
    cancellation_started: bool = False


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
        self._deadline_local = threading.local()

    @contextmanager
    def deadline(
        self, seconds: float, *, on_prompt_id: Callable[[str], None] | None = None,
    ) -> Iterator[None]:
        """Limit the combined upload, queue, poll and download budget in this thread.

        Requests timeouts bound connection/read waits, not every possible socket or
        local computation stall. A caller requiring an absolute output deadline
        must also supervise the operation in a separate process.
        """
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("生成总时限必须是大于 0 的有限秒数")
        previous = self._active_deadline()
        state = _GenerationDeadline(
            expires_at=min(time.monotonic() + seconds, previous.expires_at)
            if previous else time.monotonic() + seconds,
            seconds=seconds,
            on_prompt_id=on_prompt_id or (previous.on_prompt_id if previous else None),
            prompt_id=previous.prompt_id if previous else None,
        )
        self._deadline_local.state = state
        try:
            self._check_deadline()
            yield
            self._check_deadline()
        except TimeoutError_ as exc:
            details = dict(exc.details or {})
            details.setdefault("prompt_id", state.prompt_id)
            details.setdefault("deadline_seconds", state.seconds)
            exc.details = details
            if state.prompt_id and not state.cancellation_started:
                state.cancellation_started = True
                self.cancel_prompt_async(state.prompt_id)
            raise
        finally:
            if previous and state.prompt_id:
                previous.prompt_id = state.prompt_id
                previous.cancellation_started |= state.cancellation_started
            self._deadline_local.state = previous

    def _active_deadline(self) -> _GenerationDeadline | None:
        return getattr(self._deadline_local, "state", None)

    def _check_deadline(self) -> None:
        state = self._active_deadline()
        if state is not None and time.monotonic() >= state.expires_at:
            raise TimeoutError_(
                f"ComfyUI 生成总时限已到（{state.seconds}s，prompt_id={state.prompt_id}）",
                stage="GENERATING",
                details={"prompt_id": state.prompt_id, "deadline_seconds": state.seconds},
            )

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self._check_deadline()
        state = self._active_deadline()
        if state is not None:
            remaining = state.expires_at - time.monotonic()
            if remaining <= 0:
                self._check_deadline()
            timeout = float(kwargs.get("timeout", self.request_timeout))
            kwargs["timeout"] = HTTPTimeout(
                total=remaining,
                connect=min(self.connect_timeout, timeout, remaining),
                read=min(timeout, remaining),
            )
        try:
            response = getattr(self.session, method)(url, **kwargs)
        except requests.RequestException:
            # Convert expiry here before endpoint-specific handlers can turn it
            # into an unrelated service error.
            self._check_deadline()
            raise
        self._check_deadline()
        return response

    def cancel_prompt(self, prompt_id: str) -> None:
        """Best-effort cancellation of this prompt, never a global interrupt.

        The bundled job API checks the live running id and interrupts under
        the queue mutex, so completion cannot redirect cancellation to the
        next prompt. Unknown or completed ids are intentional no-ops.
        A separate session keeps cleanup independent of an in-flight request.
        """
        if not prompt_id:
            return
        cleanup_session = requests.Session()
        timeout = min(self.connect_timeout, 2.0)
        try:
            cleanup_session.post(
                f"{self.base_url}/api/jobs/{prompt_id}/cancel", timeout=(timeout, timeout),
            )
        except requests.RequestException:
            pass
        finally:
            cleanup_session.close()

    def cancel_prompt_async(self, prompt_id: str) -> threading.Thread | None:
        """Start targeted cleanup without delaying the caller's timeout handling."""
        if not prompt_id:
            return None
        worker = threading.Thread(
            target=self.cancel_prompt, args=(prompt_id,),
            name="comfy-prompt-cleanup", daemon=True,
        )
        worker.start()
        return worker

    # ---- 健康检查 ----
    def system_stats(self) -> dict:
        url = f"{self.base_url}/system_stats"
        try:
            response = self._request("get", url, timeout=self.connect_timeout)
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
            response = self._request("get", url, timeout=self.request_timeout)
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
            response = self._request(
                "post", url, files=files, data=form, timeout=self.request_timeout
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
        state = self._active_deadline()
        if state is not None:
            self._check_deadline()
            # Register before submission so a supervising process can cancel a
            # request even if its response arrives after the hard deadline.
            state.prompt_id = str(uuid.uuid4())
            body["prompt_id"] = state.prompt_id
            if state.on_prompt_id:
                state.on_prompt_id(state.prompt_id)
        try:
            response = self._request("post", url, json=body, timeout=self.request_timeout)
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
        if state is not None and state.prompt_id != prompt_id:
            state.prompt_id = prompt_id
            if state.on_prompt_id:
                state.on_prompt_id(prompt_id)
        self._check_deadline()
        return prompt_id

    def history(self, prompt_id: str) -> dict:
        url = f"{self.base_url}/history/{prompt_id}"
        try:
            response = self._request("get", url, timeout=self.request_timeout)
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
        state = self._active_deadline()
        if state is not None:
            state.prompt_id = prompt_id
            deadline = min(deadline, state.expires_at)
        while True:
            self._check_deadline()
            record = self.history(prompt_id)
            self._check_deadline()
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
                    details={"prompt_id": prompt_id},
                )
            time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))

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
            response = self._request("get", url, timeout=self.request_timeout)
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

    def uses_lora_loader(self) -> bool:
        return any(
            node["class_type"] in ("LoraLoaderModelOnly", "LoraLoader")
            for node in self.workflow_nodes.values()
        )

    def verify_checkpoint(self) -> None:
        """校验工作流需要的模型确实存在于 ComfyUI 中。"""
        if self.uses_unet_loader():
            self._verify_unet_model()
        else:
            available = self.client.available_checkpoints()
            if self.checkpoint not in available:
                raise ConfigError(
                    "ComfyUI 中没有找到配置的生成模型 "
                    f"'{self.checkpoint}'。当前可用: {available or '（无）'}"
                )
        self._verify_lora()

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
            encoder_loaders = dict.fromkeys(
                node["class_type"] for node in self.workflow_nodes.values()
                if node["class_type"] in ("CLIPLoaderGGUF", "CLIPLoader")
            )
            for encoder_loader in encoder_loaders:
                options = self._model_options(encoder_loader, "clip_name")
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

    def _verify_lora(self) -> None:
        if self.lora_name and self.uses_lora_loader():
            class_type = next(
                node["class_type"]
                for node in self.workflow_nodes.values()
                if node["class_type"] in ("LoraLoaderModelOnly", "LoraLoader")
            )
            options = self._model_options(class_type, "lora_name")
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

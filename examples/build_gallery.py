r"""Render the eight-movement gallery from the public example manifest.

Run from the repository root with Python 3.11 or newer:
    python examples/build_gallery.py

Uses only the standard library. Validates the 3 × 8 selection, image hashes,
shared extended-style prompts and local links before replacing gallery.html.
Does not generate images, change recipes or run automatic quality checks.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TITLES = {"01_garden": "花园", "02_family": "一家四口", "03_hill_house": "山坡与房屋"}
STYLE_IDS = (
    "fauvism", "cubism", "impressionism", "post_impressionism",
    "expressionism", "surrealism", "neoplasticism", "pop_art",
)
ADDED_STYLES = frozenset(STYLE_IDS[3:])
STATUS_LABELS = {
    "retained": "用户认可 · 保留",
    "reference": "用户认可 · 风格参照",
    "candidate": "候选 · 待评价",
    "development": "开发候选 · 待评价",
}


def escaped(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def local_file(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a repository-relative file path: {value!r}")
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or Path(value).is_absolute():
        raise ValueError(f"Expected a repository-relative file path: {value}")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Missing file or path outside the repository: {value}")
    return path


def relative_url(value: str) -> str:
    return quote(Path(os.path.relpath(local_file(value), HERE)).as_posix(), safe="/")


def verify_hash(value: str, expected: str) -> None:
    actual = hashlib.sha256(local_file(value).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Image hash differs from manifest: {value}")


def load_selection() -> tuple[dict, dict[str, str]]:
    manifest = read_json(HERE / "manifest.json")
    registry = read_json(HERE / "styles.json")
    styles = {item["id"]: item["name_zh"] for item in registry["styles"]}
    if set(styles) != set(STYLE_IDS):
        raise ValueError("The experimental registry must contain exactly eight movements")
    originals = manifest.get("originals", [])
    if len(originals) != 3 or {item["id"] for item in originals} != set(TITLES):
        raise ValueError("Gallery requires the garden, family and hill originals")
    original_by_file = {item["file"]: item["id"] for item in originals}
    for original in originals:
        verify_hash(original["file"], original["sha256"])
    results = manifest.get("results", [])
    if len(results) != 24:
        raise ValueError("Gallery requires exactly 24 selected results (3 originals × 8 movements)")
    pairs = set()
    ids = set()
    counts = {"retained": 0, "reference": 0, "candidate": 0}
    for result in results:
        original_id = original_by_file.get(result["original"])
        style_id = result["style"]
        pair = (original_id, style_id)
        if original_id is None or style_id not in styles or pair in pairs or result["id"] in ids:
            raise ValueError(f"Unknown or duplicate original/style selection: {result['id']}")
        pairs.add(pair)
        ids.add(result["id"])
        verify_hash(result["image"], result["image_sha256"])
        for key in ("config", "prompt_file", "workflow"):
            local_file(result[key])
        actual_prompt = local_file(result["prompt_file"]).read_text(encoding="utf-8-sig").strip()
        if actual_prompt != result["positive_prompt"].strip():
            raise ValueError(f"Prompt file differs from manifest: {result['id']}")
        status = result["review_status"]
        if status not in STATUS_LABELS:
            raise ValueError(f"Unknown review status: {status}")
        counts["candidate" if status == "development" else status] += 1
        if status == "retained" and style_id != "cubism":
            raise ValueError("Only the three approved Cubism results are retained")
        if status == "reference" and pair != ("03_hill_house", "impressionism"):
            raise ValueError("Only hill Impressionism is the approved visual reference")
        if result.get("automatic_check") is not None or result.get("confirmed_output") is not False:
            raise ValueError("This selection has no automatic checks or confirmed system outputs")
    expected_pairs = {(original_id, style_id) for original_id in TITLES for style_id in STYLE_IDS}
    if pairs != expected_pairs or counts != {"retained": 3, "reference": 1, "candidate": 20}:
        raise ValueError("Expected 3 retained, 1 reference and 20 candidate results across all 8 movements")
    for style_id in ADDED_STYLES:
        selected = [result for result in results if result["style"] == style_id]
        for key in ("positive_prompt", "negative_prompt"):
            if len({result[key] for result in selected}) != 1:
                raise ValueError(f"Extended style must use one shared {key}: {style_id}")
    return manifest, styles


def result_details(result: dict) -> str:
    status = result["review_status"]
    css_status = "candidate" if status == "development" else status
    keys = (
        "model", "text_encoder", "vae", "lora_name", "lora_strength", "shift", "sampler_params",
        "seed", "input_size", "output_size", "elapsed_sec", "elapsed_scope", "image_sha256",
        "input_preprocessed_sha256", "config", "prompt_file", "workflow",
    )
    parameters = {key: result[key] for key in keys if key in result}
    links = "\n".join(
        f'<a href="{relative_url(result[key])}" target="_blank" rel="noopener">{label}</a>'
        for key, label in (
            ("image", "结果原尺寸图片"), ("config", "生成配方"),
            ("prompt_file", "实际 Prompt"), ("workflow", "API 工作流"),
        )
    )
    return f'''<div class="result-details">
      <span class="tag {css_status}">{STATUS_LABELS[status]}</span>
      <p class="note">{escaped(result.get('visual_note', ''))}</p>
      <p class="check-status">未自动检查 · 未生成系统已确认输出</p>
      <div class="links">{links}</div>
      <details><summary>Prompt 与参数</summary><h4>生成 Prompt</h4>
        <pre>{escaped(result['positive_prompt'])}</pre><h4>负向 Prompt</h4>
        <pre>{escaped(result['negative_prompt'])}</pre><h4>实际参数与来源</h4>
        <pre>{escaped(json.dumps(parameters, ensure_ascii=False, indent=2))}</pre></details>
    </div>'''


def comparison_html(original: dict, result: dict, style_name: str) -> str:
    title = TITLES[original["id"]]
    original_url = relative_url(original["file"])
    result_url = relative_url(result["image"])
    width, height = original["size"]
    out_width, out_height = result["output_size"]
    return f'''<article class="comparison-card" id="{escaped(result['id'])}">
      <h3>{title}</h3><div class="image-pair">
        <figure class="original"><h4>原画</h4>
          <a class="image-well" href="{original_url}" target="_blank" rel="noopener" aria-label="查看原画原尺寸图片">
            <img src="{original_url}" alt="{title}原画，完整画面" loading="lazy"></a>
          <figcaption><p class="dimensions">{width} × {height} px</p>
            <div class="links"><a href="{original_url}" target="_blank" rel="noopener">原画原尺寸图片</a>
              <a href="manifest.json" target="_blank" rel="noopener">原图与结果清单</a></div>
          </figcaption></figure>
        <figure class="result"><h4>画派结果</h4>
          <a class="image-well" href="{result_url}" target="_blank" rel="noopener" aria-label="查看{escaped(style_name)}原尺寸图片">
            <img src="{result_url}" alt="{title} · {escaped(style_name)}，完整画面" loading="lazy"></a>
          <figcaption><p class="dimensions">{out_width} × {out_height} px</p></figcaption>
        </figure></div>{result_details(result)}</article>'''


CSS = """
:root{color-scheme:light;--ink:#23332c;--muted:#657369;--line:#dce3d9;--accent:#365d45}
*{box-sizing:border-box}body{margin:0;background:#f4f5ef;color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
main{width:min(1840px,calc(100% - 48px));margin:auto;padding:42px 0 28px}header{max-width:1120px;margin-bottom:26px}
.eyebrow{font-size:12px;letter-spacing:.15em;color:var(--accent);font-weight:700;margin:0 0 10px}
h1{font-size:clamp(28px,3.2vw,42px);line-height:1.2;margin:0 0 16px}h2{font-size:23px;margin:0 0 18px}h3{font-size:14px;margin:0 0 12px}h4{font-size:12px;margin:14px 0 5px}
.intro{color:var(--muted);margin:10px 0}.counts,.links,nav{display:flex;gap:10px 18px;flex-wrap:wrap}.counts{margin:20px 0}.counts span{border:1px solid var(--line);border-radius:20px;font-size:13px;padding:4px 12px}
.movement{scroll-margin-top:100px;margin-bottom:36px}.comparisons{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}.comparison-card{min-width:0;background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px}.image-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0;min-width:0}.image-pair h4{margin:0 0 8px;color:var(--muted)}
.image-well{display:flex;height:clamp(170px,17vw,270px);padding:5px;background:#f8f9f5;border:1px solid #e6eae2;border-radius:8px;align-items:center;justify-content:center}.image-well img{display:block;width:100%;height:100%;object-fit:contain}.dimensions{font-size:11px;color:var(--muted);margin:0}.result-details{margin-top:16px}
figcaption{padding-top:14px}.tag{display:inline-block;padding:3px 8px;font-size:12px;border-radius:4px;background:#eef1e9;color:var(--accent)}.tag.reference{background:#e8eef6;color:#3b587a}.tag.candidate{background:#fbf0d9;color:#7c5d20}
.note{font-size:13px;margin:12px 0;overflow-wrap:anywhere}.check-status{font-size:12px;color:var(--muted);margin:10px 0}.links{font-size:12px;margin-top:12px}a{color:var(--accent);text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:4px}
details{border-top:1px solid var(--line);margin-top:15px;padding-top:10px}summary{cursor:pointer;font-size:12px;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.65 ui-monospace,Consolas,monospace;color:var(--muted);margin:6px 0 12px}footer{font-size:12px;color:var(--muted)}
.movement-nav{position:sticky;top:0;z-index:2;background:rgba(244,245,239,.97);padding:14px 0;margin:0 0 28px;gap:8px}.movement-nav a{padding:5px 12px;border:1px solid var(--line);border-radius:18px;font-size:13px;text-decoration:none}.movement-nav a:hover{background:#e7eddf}.data-links{margin-top:16px}
@media(max-width:1180px){.comparisons{grid-template-columns:1fr}.image-well{height:clamp(220px,35vw,420px)}.comparison-card{padding:22px}.image-pair{gap:20px}}
@media(max-width:620px){main{width:calc(100% - 24px);padding-top:26px}.comparison-card{padding:16px}.image-pair{grid-template-columns:1fr;gap:20px}.image-well{height:min(78vw,390px)}h2{font-size:21px}.movement-nav{position:static}.movement{scroll-margin-top:12px}}
"""


class LocalLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[Path] = []
        self.image_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            self.image_count += 1
        for key, value in attrs:
            if key not in ("src", "href") or not value:
                continue
            parsed = urlsplit(value)
            if parsed.scheme or parsed.netloc:
                raise ValueError(f"Gallery must use only local links: {value}")
            if not parsed.path:
                continue
            path = (HERE / unquote(parsed.path)).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError(f"Missing gallery file or path outside the repository: {value}")
            self.paths.append(path)


def build_page(manifest: dict, styles: dict[str, str]) -> tuple[str, int]:
    originals = {item["id"]: item for item in manifest["originals"]}
    selected = {(item["original"], item["style"]): item for item in manifest["results"]}
    sections = []
    for style_id in STYLE_IDS:
        cards = "\n".join(
            comparison_html(originals[original_id], selected[(originals[original_id]["file"], style_id)], styles[style_id])
            for original_id in TITLES
        )
        sections.append(f'''<section class="movement" id="{style_id}" aria-labelledby="title-{style_id}">
          <h2 id="title-{style_id}">{escaped(styles[style_id])}</h2><div class="comparisons">{cards}</div></section>''')
    navigation = " ".join(f'<a href="#{style_id}">{escaped(styles[style_id])}</a>' for style_id in STYLE_IDS)
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>八画派作品对照</title><style>{CSS}</style></head><body><main>
<header><p class="eyebrow">AMCC5000 · CHILDREN’S ART EXPLORER</p><h1>八画派作品对照</h1>
<p class="intro">每个画派收录花园、一家四口、山坡与房屋三组原画与选定结果。点击图片可查看完整尺寸，展开卡片可查看实际 Prompt 与参数。</p>
<p class="intro">立体主义三张为用户认可的保留结果，山坡印象派为指定视觉参照，其余二十张待评价。新增五种画派共十五张候选，未执行自动粗检或独立质量验收。</p>
<p class="intro">当前页面为开发样例对照。三张原画均参与过调参；正式流程仍从野兽派、立体主义、印象派中随机选择，八画派实验配方不覆盖应用默认配置。</p>
<div class="counts"><span>3 张原画</span><span>8 种画派</span><span>24 张结果</span><span>3 张认可保留</span><span>1 张风格参照</span><span>20 张待评价候选</span></div>
<div class="links data-links"><a href="README.md">样例与复现</a><a href="manifest.json" target="_blank" rel="noopener">参数与哈希清单</a><a href="styles.json" target="_blank" rel="noopener">实验画派配置</a><a href="../MVP_V1.0.md">MVP 文档</a></div></header>
<nav class="movement-nav" aria-label="八画派导航">{navigation}</nav>
{''.join(sections)}<footer>选图日期：{escaped(manifest.get('date', '2026-10-19'))}。图片直接引用公开样例原文件；本页不生成系统已确认输出。用户认可、自动粗检和独立质量验收分别记录。</footer>
</main></body></html>'''
    links = LocalLinks()
    links.feed(page)
    if links.image_count != 48:
        raise ValueError(f"Expected 48 images (original/result pairs), got {links.image_count}")
    return page, len(links.paths)


def main() -> None:
    manifest, styles = load_selection()
    page, link_count = build_page(manifest, styles)
    (HERE / "gallery.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"Gallery ready: 3 originals, 8 movements, 24 selected results, 48 image elements; {link_count} local links verified")
    print(HERE / "gallery.html")


if __name__ == "__main__":
    main()

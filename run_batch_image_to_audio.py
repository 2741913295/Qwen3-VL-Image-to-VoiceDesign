#!/usr/bin/env python3
"""Batch image-to-audio with resident vLLM and vLLM-Omni engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from run_image_to_audio import DEFAULT_TTS_PROJECT, PROJECT_ROOT, stable_image_seed
from run_voice_design import (
    DEFAULT_OUTPUT_DIR,
    generate_instruct_for_image,
)
from voice_design.inference import (
    DEFAULT_MODEL_PATH,
    DEFAULT_VLLM_KV_CACHE_MEMORY_BYTES,
    DEFAULT_VLLM_SOURCE_PATH,
    VoiceDesigner,
)
from voice_design.tts_inference import (
    DEFAULT_QWEN3_TTS_DEPLOY_CONFIG,
    DEFAULT_VLLM_OMNI_SOURCE_PATH,
    VLLMOmniVoiceDesignTTS,
)


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
SINGLE_IMAGE_ENTRY = PROJECT_ROOT / "run_image_to_audio.py"
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
)


@dataclass(frozen=True)
class BatchItem:
    image_path: Path
    instruct_path: Path
    audio_path: Path


def discover_images(input_dir: str, *, recursive: bool = False) -> list[Path]:
    """Find supported images in a stable order and reject output-name collisions."""
    directory = Path(input_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在：{directory}")

    candidates = directory.rglob("*") if recursive else directory.iterdir()
    images = sorted(
        (
            path.resolve()
            for path in candidates
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )
    if not images:
        extensions = "、".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise FileNotFoundError(
            f"输入文件夹中没有支持的图片：{directory}（支持：{extensions}）"
        )

    paths_by_stem: dict[str, Path] = {}
    for image_path in images:
        collision_key = image_path.stem.casefold()
        previous = paths_by_stem.get(collision_key)
        if previous is not None:
            raise ValueError(
                "多张图片会生成相同的输出文件名，请先重命名其中一张："
                f"{previous}；{image_path}"
            )
        paths_by_stem[collision_key] = image_path
    return images


def resolve_output_dir(input_dir: str, output_dir: str | None = None) -> Path:
    """Use output/<input folder name>, while preserving the original data default."""
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    resolved_input = Path(input_dir).expanduser().resolve()
    default_input = DEFAULT_INPUT_DIR.expanduser().resolve()
    if resolved_input == default_input:
        return DEFAULT_OUTPUT_DIR.expanduser().resolve()
    return (DEFAULT_OUTPUT_DIR / resolved_input.name).resolve()


def make_batch_items(images: list[Path], output_dir: Path) -> list[BatchItem]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        BatchItem(
            image_path=image_path,
            instruct_path=output_dir / f"{image_path.stem}.txt",
            audio_path=output_dir / f"{image_path.stem}.wav",
        )
        for image_path in images
    ]


def find_duplicate_images(images: list[Path]) -> list[list[Path]]:
    """Group byte-identical images so repeated controls are visible to the user."""
    images_by_digest: dict[str, list[Path]] = {}
    for image_path in images:
        digest = hashlib.sha256()
        with image_path.open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        images_by_digest.setdefault(digest.hexdigest(), []).append(image_path)
    return [group for group in images_by_digest.values() if len(group) > 1]


def build_single_image_command(
    args: argparse.Namespace,
    item: BatchItem,
    profile_registry: Path | None = None,
) -> list[str]:
    """Only pass user inputs so server-side defaults and hardcoding stay active."""
    command = [
        sys.executable,
        str(SINGLE_IMAGE_ENTRY),
        "--image",
        str(item.image_path),
        "--output-dir",
        str(item.instruct_path.parent),
    ]
    if args.text is not None:
        command.extend(["--text", args.text])
    else:
        command.extend(
            ["--text-file", str(Path(args.text_file).expanduser().resolve())]
        )
    if args.instruction.strip():
        command.extend(["--instruction", args.instruction.strip()])
    attn_implementation = getattr(args, "attn_implementation", None)
    if attn_implementation:
        command.extend(["--attn-implementation", attn_implementation])
    if profile_registry is not None:
        command.extend(["--contrast-profile-registry", str(profile_registry)])
    return command


def prepare_profile_registry(output_dir: Path, *, keep_existing: bool) -> Path:
    """Create the batch-level registry used for acoustic collision checks."""
    registry_path = output_dir / ".voice_profiles.json"
    if not keep_existing or not registry_path.is_file():
        registry_path.write_text(
            json.dumps([], ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return registry_path


def remove_stale_outputs(item: BatchItem) -> None:
    """Prevent failed reruns from leaving an old TXT/WAV that looks current."""
    for output_path in (item.instruct_path, item.audio_path):
        if output_path.is_file():
            output_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用常驻 vLLM + vLLM-Omni 批量生成同名提示词和 WAV 音频。"
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"图片文件夹（默认：{DEFAULT_INPUT_DIR}）",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="所有人物共同朗读的文本")
    text_group.add_argument("--text-file", help="UTF-8 朗读文本文件")
    parser.add_argument(
        "--instruction", default="", help="所有图片共同使用的可选声线或表达约束"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="仅 transformers 回退后端使用；vLLM 会自行选择注意力实现",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="vllm",
        help="Qwen3-VL 推理后端（默认：vllm）",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.15,
        help="Qwen3-VL vLLM 显存比例后备值（默认：0.15）",
    )
    parser.add_argument(
        "--vllm-kv-cache-memory-bytes",
        type=int,
        default=DEFAULT_VLLM_KV_CACHE_MEMORY_BYTES,
        help="Qwen3-VL KV Cache 固定字节数（默认：2 GiB）",
    )
    parser.add_argument(
        "--vllm-source-path",
        default=str(DEFAULT_VLLM_SOURCE_PATH),
        help=(
            "固定使用的 vLLM 0.28.x 源码/安装目录（服务器默认为项目下 vllm）"
            f"（默认：{DEFAULT_VLLM_SOURCE_PATH}）"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--tts-project", default=str(DEFAULT_TTS_PROJECT))
    parser.add_argument("--tts-model-path", default=None)
    parser.add_argument(
        "--vllm-omni-source-path",
        default=str(DEFAULT_VLLM_OMNI_SOURCE_PATH),
        help=(
            "固定使用的 vLLM-Omni 0.28.x 源码/安装目录"
            f"（默认：{DEFAULT_VLLM_OMNI_SOURCE_PATH}）"
        ),
    )
    parser.add_argument(
        "--tts-deploy-config",
        default=str(DEFAULT_QWEN3_TTS_DEPLOY_CONFIG),
        help="vLLM-Omni 的 Qwen3-TTS 部署 YAML",
    )
    parser.add_argument("--tts-language", default="Chinese")
    parser.add_argument("--tts-max-new-tokens", type=int, default=2048)
    parser.add_argument("--tts-seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认使用 output/输入文件夹名",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="递归处理输入文件夹中的子文件夹"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="同时存在同名 txt 和非空 WAV 时跳过，适合断点续跑",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.text_file:
            text_file = Path(args.text_file).expanduser().resolve()
            if not text_file.is_file():
                raise FileNotFoundError(f"朗读文本文件不存在：{text_file}")
        images = discover_images(args.input_dir, recursive=args.recursive)
        output_dir = resolve_output_dir(args.input_dir, args.output_dir)
        items = make_batch_items(images, output_dir)
        profile_registry = prepare_profile_registry(
            output_dir, keep_existing=args.skip_existing
        )
        duplicate_groups = find_duplicate_images(images)
        if args.text_file:
            target_text = text_file.read_text(encoding="utf-8").strip()
        else:
            target_text = (args.text or "").strip()
        if not target_text:
            raise ValueError("朗读文本不能为空。")
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    for group in duplicate_groups:
        names = "、".join(path.name for path in group)
        print(f"[重复图片提示] 以下文件内容完全相同：{names}", file=sys.stderr)

    pending_items = [
        item
        for item in items
        if not (
            args.skip_existing
            and item.instruct_path.is_file()
            and item.audio_path.is_file()
            and item.audio_path.stat().st_size > 0
        )
    ]
    if not pending_items:
        print(f"批处理完成：发现 {len(items)} 张图片，全部已有有效输出。")
        print(f"输出目录：{output_dir}")
        return 0

    tts_model_path = Path(
        args.tts_model_path or (Path(args.tts_project) / "model")
    ).expanduser().resolve()
    try:
        if args.backend == "vllm" and args.attn_implementation:
            print(
                "提示：vLLM 自行选择注意力后端，--attn-implementation 仅用于 transformers。",
                file=sys.stderr,
            )
        print("正在加载常驻 Qwen3-VL 模型……", file=sys.stderr)
        designer = VoiceDesigner(
            args.model_path,
            attn_implementation=(
                args.attn_implementation if args.backend == "transformers" else None
            ),
            max_new_tokens=args.max_new_tokens,
            backend=args.backend,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_kv_cache_memory_bytes=args.vllm_kv_cache_memory_bytes,
            vllm_source_path=args.vllm_source_path,
            allowed_local_media_path=args.input_dir,
        )
        if designer.vllm_version:
            print(
                "已加载 vLLM "
                f"{designer.vllm_version}（{designer.vllm_source_path}）",
                file=sys.stderr,
            )
        print("正在加载常驻 Qwen3-TTS vLLM-Omni 模型……", file=sys.stderr)
        tts = VLLMOmniVoiceDesignTTS(
            str(tts_model_path),
            omni_source_path=args.vllm_omni_source_path,
            deploy_config=args.tts_deploy_config,
            max_new_tokens=args.tts_max_new_tokens,
        )
        print(
            f"已加载 vLLM-Omni {tts.omni_version}；两个模型将常驻并连续推理。",
            file=sys.stderr,
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"模型加载失败：{exc}", file=sys.stderr)
        return 2

    failures: list[tuple[Path, str]] = []
    successful = 0
    for index, item in enumerate(items, start=1):
        if (
            args.skip_existing
            and item.instruct_path.is_file()
            and item.audio_path.is_file()
            and item.audio_path.stat().st_size > 0
        ):
            successful += 1
            print(f"[跳过 {index}/{len(items)}] {item.image_path.name}", file=sys.stderr)
            continue

        print(f"[处理 {index}/{len(items)}] {item.image_path.name}", file=sys.stderr)
        try:
            remove_stale_outputs(item)
            result = generate_instruct_for_image(
                designer,
                str(item.image_path),
                args.instruction,
                str(output_dir),
                str(profile_registry),
            )
            tts.generate(
                text=target_text,
                instruct=result.instruct,
                output_path=item.audio_path,
                seed=(
                    args.tts_seed
                    if args.tts_seed is not None
                    else stable_image_seed(item.image_path)
                ),
                language=args.tts_language,
            )
            if not item.instruct_path.is_file():
                raise RuntimeError(f"未生成提示词文件：{item.instruct_path}")
            if not item.audio_path.is_file() or item.audio_path.stat().st_size == 0:
                raise RuntimeError(f"未生成有效音频：{item.audio_path}")
            successful += 1
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append((item.image_path, str(exc)))
            print(f"  失败：{exc}", file=sys.stderr)

    print(
        f"批处理完成：发现 {len(items)} 张图片，成功 {successful} 张，失败 {len(failures)} 张。"
    )
    print(f"输出目录：{output_dir}")
    if failures:
        print("失败项目：", file=sys.stderr)
        for image_path, reason in failures:
            print(f"- {image_path.name}: {reason}", file=sys.stderr)
        tts.close()
        return 1
    tts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate a VoiceDesign instruct with vLLM and audio with vLLM-Omni."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from run_voice_design import DEFAULT_OUTPUT_DIR, generate_instruct_for_image
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


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TTS_PROJECT = (
    PROJECT_ROOT.parent / "voice-design-test" / "qwen3-tts-voice-design"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 vLLM + vLLM-Omni 完成人物图片到 VoiceDesign 音频。"
    )
    parser.add_argument("--image", required=True, help="本地人物图片路径")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="最终需要朗读的文本")
    text_group.add_argument("--text-file", help="UTF-8 朗读文本文件")
    parser.add_argument("--instruction", default="", help="可选声线或表达约束")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--backend",
        choices=("vllm", "transformers"),
        default="vllm",
        help="Qwen3-VL 推理后端（默认：vllm；transformers 仅作兼容回退）",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="仅 transformers 回退后端使用",
    )
    parser.add_argument(
        "--vllm-source-path",
        default=str(DEFAULT_VLLM_SOURCE_PATH),
        help=f"vLLM 0.28.x 源码目录（默认：{DEFAULT_VLLM_SOURCE_PATH}）",
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
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--tts-project", default=str(DEFAULT_TTS_PROJECT))
    parser.add_argument("--tts-model-path", default=None)
    parser.add_argument(
        "--vllm-omni-source-path",
        default=str(DEFAULT_VLLM_OMNI_SOURCE_PATH),
        help=f"vLLM-Omni 0.28.x 源码目录（默认：{DEFAULT_VLLM_OMNI_SOURCE_PATH}）",
    )
    parser.add_argument(
        "--tts-deploy-config",
        default=str(DEFAULT_QWEN3_TTS_DEPLOY_CONFIG),
        help="vLLM-Omni 的 Qwen3-TTS 部署 YAML",
    )
    parser.add_argument("--tts-language", default="Chinese")
    parser.add_argument("--tts-max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--tts-seed",
        type=int,
        default=None,
        help="VoiceDesign 随机种子；默认根据图片内容稳定生成",
    )
    parser.add_argument(
        "--contrast-profile-registry", default="", help=argparse.SUPPRESS
    )
    return parser.parse_args()


def stable_image_seed(image_path: Path) -> int:
    """Return a repeatable non-zero TTS seed derived from image bytes."""
    digest = hashlib.sha256()
    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return int.from_bytes(digest.digest()[:4], "big") % 2_147_483_646 + 1


def resolve_target_text(text: str | None, text_file: str | None) -> str:
    if text_file:
        path = Path(text_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"朗读文本文件不存在：{path}")
        result = path.read_text(encoding="utf-8").strip()
    else:
        result = (text or "").strip()
    if not result:
        raise ValueError("朗读文本不能为空。")
    return result


def resolve_output_paths(image_path: Path, output_dir: str | Path) -> tuple[Path, Path]:
    resolved_output = Path(output_dir).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    return (
        resolved_output / f"{image_path.stem}.txt",
        resolved_output / f"{image_path.stem}.wav",
    )


def main() -> int:
    args = parse_args()
    designer = None
    tts = None
    try:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"图片文件不存在：{image_path}")
        target_text = resolve_target_text(args.text, args.text_file)
        output_dir = Path(args.output_dir).expanduser().resolve()
        _, audio_path = resolve_output_paths(image_path, output_dir)
        registry_path = (
            Path(args.contrast_profile_registry).expanduser().resolve()
            if args.contrast_profile_registry
            else output_dir / ".voice_profiles.json"
        )
        tts_model_path = Path(
            args.tts_model_path or Path(args.tts_project) / "model"
        ).expanduser().resolve()

        print("正在加载 Qwen3-VL vLLM 常驻引擎……", file=sys.stderr)
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
            allowed_local_media_path=image_path.parent,
        )
        print("正在加载 Qwen3-TTS vLLM-Omni 常驻引擎……", file=sys.stderr)
        tts = VLLMOmniVoiceDesignTTS(
            str(tts_model_path),
            omni_source_path=args.vllm_omni_source_path,
            deploy_config=args.tts_deploy_config,
            max_new_tokens=args.tts_max_new_tokens,
        )
        result = generate_instruct_for_image(
            designer,
            str(image_path),
            args.instruction,
            str(output_dir),
            str(registry_path),
        )
        print(result.instruct)
        tts.generate(
            text=target_text,
            instruct=result.instruct,
            output_path=audio_path,
            seed=args.tts_seed or stable_image_seed(image_path),
            language=args.tts_language,
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    finally:
        if tts is not None:
            tts.close()

    print(audio_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

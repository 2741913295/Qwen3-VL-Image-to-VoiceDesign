#!/usr/bin/env python3
"""CLI: character image -> Qwen3-TTS VoiceDesign Chinese instruction."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from voice_design.inference import (
    DEFAULT_MODEL_PATH,
    DEFAULT_VLLM_KV_CACHE_MEMORY_BYTES,
    DEFAULT_VLLM_SOURCE_PATH,
    VoiceDesigner,
)
from voice_design.profiles import (
    BaseVoiceProfile,
    DeliveryProfile,
    PersonaProfile,
    SpeakingBehavior,
    VisualProfile,
    VoiceDesignReference,
    build_instruct,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class VoiceDesignResult:
    instruct: str
    output_path: Path
    visual_profile: VisualProfile
    persona_profile: PersonaProfile
    speaking_behavior: SpeakingBehavior
    voice_fingerprint: BaseVoiceProfile
    final_voice_fingerprint: BaseVoiceProfile
    delivery_profile: DeliveryProfile | None
    batch_contrast_applied: bool
    reference_design_count: int


def _read_profile_registry(registry_path: Path) -> list[dict[str, object]]:
    if not registry_path.is_file():
        return []
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"批量声纹注册文件无效：{registry_path}") from exc
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"批量声纹注册文件必须是 JSON 数组：{registry_path}")
    return data


def load_reference_designs(
    registry_file: str, current_image: str
) -> list[VoiceDesignReference]:
    """Load prior layered designs, including legacy voice-only entries."""
    registry_path = Path(registry_file).expanduser().resolve()
    current_path = str(Path(current_image).expanduser().resolve())
    designs: list[VoiceDesignReference] = []
    for entry in _read_profile_registry(registry_path):
        if entry.get("image") == current_path:
            continue
        profile_data = entry.get("profile")
        if isinstance(profile_data, dict):
            persona_data = entry.get("persona_profile")
            behavior_data = entry.get("speaking_behavior")
            designs.append(
                VoiceDesignReference(
                    voice_fingerprint=BaseVoiceProfile.from_mapping(profile_data),
                    persona_profile=(
                        PersonaProfile.from_mapping(persona_data)
                        if isinstance(persona_data, dict)
                        else None
                    ),
                    speaking_behavior=(
                        SpeakingBehavior.from_mapping(behavior_data)
                        if isinstance(behavior_data, dict)
                        else None
                    ),
                )
            )
    return designs


def load_reference_profiles(
    registry_file: str, current_image: str
) -> list[BaseVoiceProfile]:
    """Compatibility wrapper returning only saved voice fingerprints."""
    return [
        design.voice_fingerprint
        for design in load_reference_designs(registry_file, current_image)
    ]


def update_profile_registry(
    registry_file: str,
    image_path: str,
    profile: BaseVoiceProfile,
    persona_profile: PersonaProfile | None = None,
    speaking_behavior: SpeakingBehavior | None = None,
) -> None:
    """Atomically add or replace one image's layered character design."""
    registry_path = Path(registry_file).expanduser().resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_image = str(Path(image_path).expanduser().resolve())
    entries = [
        entry
        for entry in _read_profile_registry(registry_path)
        if entry.get("image") != resolved_image
    ]
    new_entry: dict[str, object] = {
        "image": resolved_image,
        "profile": profile.as_json_dict(),
    }
    if persona_profile is not None:
        new_entry["persona_profile"] = persona_profile.as_json_dict()
    if speaking_behavior is not None:
        new_entry["speaking_behavior"] = speaking_behavior.as_json_dict()
    entries.append(new_entry)
    temporary_path = registry_path.with_name(registry_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(registry_path)


def save_instruct(instruct: str, image_path: str, output_dir: str) -> Path:
    """Save one UTF-8 instruction using the input image stem as filename."""
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_output_dir / f"{Path(image_path).stem}.txt"
    output_path.write_text(instruct + "\n", encoding="utf-8")
    return output_path


def generate_instruct_for_image(
    designer: VoiceDesigner,
    image_path: str,
    instruction: str,
    output_dir: str,
    registry_file: str = "",
) -> VoiceDesignResult:
    """Generate one instruct while reusing an already loaded VL model."""
    reference_designs = (
        load_reference_designs(registry_file, image_path) if registry_file else []
    )
    visual_profile = designer.analyze_visual_profile(image_path)
    persona_profile = designer.analyze_persona_profile(visual_profile)
    speaking_behavior = designer.analyze_speaking_behavior(persona_profile)
    voice_fingerprint = designer.design_voice_fingerprint(
        visual_profile, persona_profile, speaking_behavior
    )
    base_profile = designer.apply_voice_instruction(instruction, voice_fingerprint)
    pre_contrast_design = (persona_profile, speaking_behavior, base_profile)
    persona_profile, speaking_behavior, base_profile = designer.ensure_batch_contrast(
        base_profile,
        reference_designs,
        visual_profile=visual_profile,
        persona_profile=persona_profile,
        speaking_behavior=speaking_behavior,
        instruction=instruction,
    )
    delivery_profile = designer.analyze_delivery(instruction, base_profile)
    instruct = build_instruct(base_profile, delivery_profile, persona_profile)
    output_path = save_instruct(instruct, image_path, output_dir)
    if registry_file:
        update_profile_registry(
            registry_file,
            image_path,
            base_profile,
            persona_profile,
            speaking_behavior,
        )
    return VoiceDesignResult(
        instruct=instruct,
        output_path=output_path,
        visual_profile=visual_profile,
        persona_profile=persona_profile,
        speaking_behavior=speaking_behavior,
        voice_fingerprint=voice_fingerprint,
        final_voice_fingerprint=base_profile,
        delivery_profile=delivery_profile,
        batch_contrast_applied=(
            pre_contrast_design != (persona_profile, speaking_behavior, base_profile)
        ),
        reference_design_count=len(reference_designs),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用本地 Qwen3-VL-8B-Instruct 将人物图片转换为 VoiceDesign instruct。"
    )
    parser.add_argument("--image", required=True, help="本地人物图片路径")
    parser.add_argument(
        "--instruction",
        default="",
        help="可选声线或表达约束，例如：45岁男性粗犷声线，或用愤怒语气说",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"本地 Qwen3-VL 权重目录（默认：{DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"保存同名 txt 的目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="可选注意力实现；安装 flash-attn 后可选 flash_attention_2",
    )
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
        help="vLLM 显存比例后备值（默认：0.15；显式 KV Cache 预算优先）",
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="每次结构化生成的最大新 token 数（默认：384）",
    )
    parser.add_argument(
        "--debug-profile",
        action="store_true",
        help="将中间 Voice Profile JSON 打印到 stderr，不影响 stdout 的最终 instruct",
    )
    parser.add_argument(
        "--contrast-profile-registry",
        default="auto",
        help=(
            "批量声线去重注册文件；默认 auto，自动使用 output-dir 下的 "
            ".voice_design_registry.json"
        ),
    )
    parser.add_argument(
        "--no-batch-contrast",
        action="store_true",
        help="关闭跨图片历史声线对比与自动去重。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 64:
        print("错误：--max-new-tokens 不能小于 64。", file=sys.stderr)
        return 2

    if args.no_batch_contrast:
        registry_file = ""
    elif args.contrast_profile_registry == "auto":
        registry_file = str(
            Path(args.output_dir).expanduser().resolve()
            / ".voice_design_registry.json"
        )
    else:
        registry_file = args.contrast_profile_registry

    try:
        designer = VoiceDesigner(
            args.model_path,
            attn_implementation=args.attn_implementation,
            max_new_tokens=args.max_new_tokens,
            backend=args.backend,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_kv_cache_memory_bytes=args.vllm_kv_cache_memory_bytes,
            vllm_source_path=args.vllm_source_path,
            allowed_local_media_path=str(
                Path(args.image).expanduser().resolve().parent
            ),
        )
        result = generate_instruct_for_image(
            designer,
            args.image,
            args.instruction,
            args.output_dir,
            registry_file,
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if args.debug_profile:
        debug_data = {
            "visual_profile": result.visual_profile.as_json_dict(),
            "persona_profile": result.persona_profile.as_json_dict(),
            "speaking_behavior": result.speaking_behavior.as_json_dict(),
            "voice_fingerprint_before_instruction": result.voice_fingerprint.as_json_dict(),
            "final_voice_fingerprint": result.final_voice_fingerprint.as_json_dict(),
            "batch_contrast_applied": result.batch_contrast_applied,
            "contrast_registry": registry_file or None,
            "reference_design_count": result.reference_design_count,
            "delivery": (
                result.delivery_profile.__dict__ if result.delivery_profile else None
            ),
            "output_path": str(result.output_path),
        }
        print(json.dumps(debug_data, ensure_ascii=False, indent=2), file=sys.stderr)

    print(result.instruct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

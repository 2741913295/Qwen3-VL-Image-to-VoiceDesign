"""Offline Qwen3-VL inference for structured voice profiles."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

# These must be set before importing Transformers/Hugging Face Hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from .profiles import (
    BaseVoiceProfile,
    DeliveryProfile,
    PersonaProfile,
    SpeakingBehavior,
    VisualProfile,
    VisualAnchorReview,
    VoiceDesignReference,
    MATURE_ADULT_AGE,
    acoustic_contrast,
    align_voice_to_visual_structure,
    allocate_contrasting_profile,
    apply_voice_overrides,
    infer_literal_voice_overrides,
    layered_design_is_too_similar,
    layered_design_similarity,
    profile_meets_contrast,
    primary_timbre_contrast,
    reconcile_visual_anchors,
    macro_voice_contrast,
    similar_demographic_core_contrast,
    timbre_identity_contrast,
    weighted_acoustic_distance,
)
from .prompts import (
    CONTRASTIVE_PERSONA_SYSTEM_PROMPT,
    DELIVERY_SYSTEM_PROMPT,
    JSON_REPAIR_SYSTEM_PROMPT,
    PERSONA_PROFILE_SYSTEM_PROMPT,
    SPEAKING_BEHAVIOR_SYSTEM_PROMPT,
    VISUAL_PROFILE_SYSTEM_PROMPT,
    VISUAL_PROFILE_USER_PROMPT,
    VISUAL_ANCHOR_REVIEW_SYSTEM_PROMPT,
    VOICE_FINGERPRINT_SYSTEM_PROMPT,
    VOICE_OVERRIDE_SYSTEM_PROMPT,
    build_contrastive_persona_user_prompt,
    build_delivery_user_prompt,
    build_persona_user_prompt,
    build_refinement_user_prompt,
    build_speaking_behavior_user_prompt,
    build_visual_anchor_review_prompt,
    build_voice_fingerprint_user_prompt,
    build_voice_override_user_prompt,
)


DEFAULT_MODEL_PATH = "/data/huggingface/hub/Qwen3-VL-8B-Instruct-FP8"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VLLM_SOURCE_PATH = (
    PROJECT_ROOT / "vllm"
    if (PROJECT_ROOT / "vllm").is_dir()
    else PROJECT_ROOT / "vllm-0.28.0"
)
REQUIRED_VLLM_VERSION_PREFIX = "0.28."
DEFAULT_VLLM_KV_CACHE_MEMORY_BYTES = 2 * 1024**3
MIN_IMAGE_PIXELS = 256 * 32 * 32
MAX_IMAGE_PIXELS = 1280 * 32 * 32


def text_content(text: str) -> list[dict[str, str]]:
    """Wrap text in the multimodal content-list format required by Qwen3-VL."""
    return [{"type": "text", "text": text}]


def extract_json_object(text: str) -> Mapping[str, Any]:
    """Extract the first valid JSON object, tolerating Markdown fences or leading text."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"模型未返回合法 JSON 对象：{text[:300]!r}")


def load_project_vllm(
    source_path: str | os.PathLike[str] | None = None,
) -> tuple[Any, Path, str]:
    """Load the installed vLLM runtime aligned with the bundled 0.28.x source."""
    resolved_source = Path(
        source_path or DEFAULT_VLLM_SOURCE_PATH
    ).expanduser().resolve()
    package_init = resolved_source / "vllm" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(
            "项目指定的 vLLM 0.28.x 源码目录无效："
            f"{resolved_source}（缺少 vllm/__init__.py）"
        )

    try:
        loaded_module = importlib.import_module("vllm")
    except ImportError as exc:
        raise RuntimeError(
            "没有安装可运行的 vLLM 0.28.x。源码目录不能代替安装包，"
            "请先安装与项目 vllm 源码同版本的 CUDA wheel。"
        ) from exc

    version = str(getattr(loaded_module, "__version__", "unknown"))
    if not version.startswith(REQUIRED_VLLM_VERSION_PREFIX):
        raise RuntimeError(
            "vLLM 版本不匹配："
            f"当前为 {version}，要求 0.28.x；项目源码为 {resolved_source}。"
            "请安装 vLLM 0.28.x 后重启进程。"
        )
    return loaded_module, resolved_source, version


class VoiceDesigner:
    """Load one local Qwen3-VL model and create deterministic voice instructions."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        attn_implementation: str | None = None,
        max_new_tokens: int = 384,
        backend: str = "transformers",
        vllm_gpu_memory_utilization: float = 0.15,
        vllm_kv_cache_memory_bytes: int | None = DEFAULT_VLLM_KV_CACHE_MEMORY_BYTES,
        allowed_local_media_path: str | None = None,
        vllm_source_path: str | os.PathLike[str] | None = None,
    ) -> None:
        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.is_dir():
            raise FileNotFoundError(f"本地模型目录不存在：{resolved_path}")

        if backend not in {"transformers", "vllm"}:
            raise ValueError(f"不支持的 Qwen3-VL 推理后端：{backend}")
        if not 0 < vllm_gpu_memory_utilization < 1:
            raise ValueError("vLLM GPU 显存占用比例必须在 0 到 1 之间。")
        if (
            vllm_kv_cache_memory_bytes is not None
            and vllm_kv_cache_memory_bytes <= 0
        ):
            raise ValueError("vLLM KV Cache 字节数必须大于 0。")

        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.vllm_source_path: Path | None = None
        self.vllm_version: str | None = None
        if backend == "vllm":
            vllm_module, self.vllm_source_path, self.vllm_version = (
                load_project_vllm(vllm_source_path)
            )
            LLM = vllm_module.LLM
            SamplingParams = vllm_module.SamplingParams
            try:
                from qwen_vl_utils import process_vision_info
                from transformers import AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "vLLM 图像推理需要 transformers 和 qwen-vl-utils，"
                    "用于复用 Qwen3-VL 官方聊天模板与图片预处理。"
                ) from exc

            media_root = Path(
                allowed_local_media_path or Path.cwd()
            ).expanduser().resolve()
            if not media_root.is_dir():
                raise FileNotFoundError(f"vLLM 本地图片目录不存在：{media_root}")
            self.processor = AutoProcessor.from_pretrained(
                str(resolved_path), local_files_only=True
            )
            self._process_vision_info = process_vision_info
            self.model = LLM(
                model=str(resolved_path),
                dtype="bfloat16",
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
                max_model_len=4096,
                limit_mm_per_prompt={"image": 1},
                allowed_local_media_path=str(media_root),
                mm_processor_kwargs={
                    "min_pixels": MIN_IMAGE_PIXELS,
                    "max_pixels": MAX_IMAGE_PIXELS,
                },
                enable_prefix_caching=True,
                seed=0,
            )
            self._vllm_sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_new_tokens,
                top_k=-1,
                stop_token_ids=[],
            )
            return

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "缺少运行依赖，请先安装 requirements_voice_design.txt（Transformers 需 >= 4.57.0）。"
            ) from exc

        model_kwargs: dict[str, Any] = {
            "dtype": "auto",
            "device_map": "auto",
            "local_files_only": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if attn_implementation == "flash_attention_2":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("FlashAttention 2 需要先安装 PyTorch。") from exc
            model_kwargs["dtype"] = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(
            str(resolved_path), local_files_only=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(resolved_path), **model_kwargs
        )
        self.model.eval()

    def _prepare_vllm_input(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Use Qwen's official Processor path before handing work to vLLM."""
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        multimodal_data: dict[str, Any] = {}
        if image_inputs is not None:
            multimodal_data["image"] = image_inputs
        if video_inputs is not None:
            multimodal_data["video"] = video_inputs
        processor_kwargs = dict(video_kwargs or {}) if multimodal_data else {}
        if multimodal_data:
            # qwen-vl-utils has already applied the per-image pixel budget.
            # Avoid a second resize inside vLLM's Transformers processor.
            processor_kwargs["do_resize"] = False
        return {
            "prompt": prompt,
            "multi_modal_data": multimodal_data,
            "mm_processor_kwargs": processor_kwargs,
        }

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        if self.backend == "vllm":
            outputs = self.model.generate(
                [self._prepare_vllm_input(messages)],
                sampling_params=self._vllm_sampling_params,
                use_tqdm=False,
            )
            return outputs[0].outputs[0].text.strip()

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
        )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _generate_json(self, messages: list[dict[str, Any]]) -> Mapping[str, Any]:
        raw_output = self._generate(messages)
        try:
            return extract_json_object(raw_output)
        except ValueError:
            schema_rules = "\n".join(
                item.get("text", "")
                for item in messages[0].get("content", [])
                if item.get("type") == "text"
            )
            repair_messages = [
                {
                    "role": "system",
                    "content": text_content(JSON_REPAIR_SYSTEM_PROMPT),
                },
                {
                    "role": "user",
                    "content": text_content(
                        "目标 schema 和候选值规则：\n"
                        + schema_rules
                        + "\n\n请修复下面的模型输出，只保留合法 JSON：\n"
                        + raw_output
                    ),
                },
            ]
            return extract_json_object(self._generate(repair_messages))

    def analyze_visual_profile(self, image_path: str) -> VisualProfile:
        """Stage 1: extract image-grounded evidence without designing a voice."""
        image = Path(image_path).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"图片文件不存在：{image}")

        image_content = {
            "type": "image",
            # Pass a plain local path. Some Transformers 4.57 builds
            # misclassify percent-encoded file:// URIs as base64.
            "image": str(image),
            "min_pixels": MIN_IMAGE_PIXELS,
            "max_pixels": MAX_IMAGE_PIXELS,
        }
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": text_content(VISUAL_PROFILE_SYSTEM_PROMPT),
            },
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": VISUAL_PROFILE_USER_PROMPT},
                ],
            },
        ]
        visual_profile = VisualProfile.from_mapping(self._generate_json(messages))
        visual_json = json.dumps(
            visual_profile.as_json_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        review_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": text_content(VISUAL_ANCHOR_REVIEW_SYSTEM_PROMPT),
            },
            {
                "role": "user",
                "content": [
                    image_content,
                    {
                        "type": "text",
                        "text": build_visual_anchor_review_prompt(visual_json),
                    },
                ],
            },
        ]
        try:
            review = VisualAnchorReview.from_mapping(
                self._generate_json(review_messages)
            )
        except ValueError:
            print(
                "警告：视觉锚点复核未返回合法结构，已使用初步视觉结果并执行一致性校正。",
                file=sys.stderr,
            )
            review = VisualAnchorReview.from_mapping(
                {
                    "age_style": visual_profile.visual_age_style,
                    "gender_presentation": visual_profile.gender_presentation,
                    "facial_maturity": visual_profile.facial_maturity,
                    "visual_mass": visual_profile.visual_mass,
                    "facial_hair": visual_profile.facial_hair,
                    "confidence": "low",
                }
            )
        return reconcile_visual_anchors(visual_profile, review)

    def analyze_persona_profile(
        self, visual_profile: VisualProfile
    ) -> PersonaProfile:
        """Stage 2: create a fictional persona from visual evidence."""
        visual_json = json.dumps(
            visual_profile.as_json_dict(), ensure_ascii=False, separators=(",", ":")
        )
        messages = [
            {
                "role": "system",
                "content": text_content(PERSONA_PROFILE_SYSTEM_PROMPT),
            },
            {
                "role": "user",
                "content": text_content(build_persona_user_prompt(visual_json)),
            },
        ]
        profile_data = dict(self._generate_json(messages))
        profile_data.update(
            {
                "gender": visual_profile.gender_presentation,
                "age_style": visual_profile.visual_age_style,
            }
        )
        profile = PersonaProfile.from_mapping(profile_data)
        if profile.quality_issue_count() == 0:
            return profile

        profile_json = json.dumps(
            profile.as_json_dict(), ensure_ascii=False, separators=(",", ":")
        )
        refinement_messages = [
            {
                "role": "system",
                "content": text_content(PERSONA_PROFILE_SYSTEM_PROMPT),
            },
            {
                "role": "user",
                "content": text_content(
                    build_persona_user_prompt(visual_json)
                    + "\n\n上一版人设存在泛化、诗化、视觉物件或残句问题：\n"
                    + profile_json
                    + "\n请重新设计具体身份、驱动力、社交方式与情绪模式，"
                    + "必须使用简洁完整的现实语言，不得使用比喻、视觉物件或半截句。"
                ),
            },
        ]
        refined_data = dict(self._generate_json(refinement_messages))
        refined_data.update(
            {
                "gender": visual_profile.gender_presentation,
                "age_style": visual_profile.visual_age_style,
            }
        )
        refined_profile = PersonaProfile.from_mapping(refined_data)
        if refined_profile.quality_issue_count() > 0:
            selected_profile = min(
                (profile, refined_profile),
                key=lambda candidate: candidate.quality_issue_count(),
            )
            print(
                "警告：Persona Profile 两次均存在文案质量问题，已保留两版中"
                "问题更少的一版，"
                "并继续交由批量三层相似度检查。",
                file=sys.stderr,
            )
            return selected_profile
        return refined_profile

    def analyze_speaking_behavior(
        self, persona_profile: PersonaProfile
    ) -> SpeakingBehavior:
        """Stage 3: derive stable sentence-level habits from the persona."""
        persona_json = json.dumps(
            persona_profile.as_json_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages = [
            {
                "role": "system",
                "content": text_content(SPEAKING_BEHAVIOR_SYSTEM_PROMPT),
            },
            {
                "role": "user",
                "content": text_content(
                    build_speaking_behavior_user_prompt(persona_json)
                ),
            },
        ]
        profile = SpeakingBehavior.from_mapping(self._generate_json(messages))
        if profile.neutral_dimension_count() <= 3:
            return profile

        profile_json = json.dumps(
            profile.as_json_dict(), ensure_ascii=False, separators=(",", ":")
        )
        retry_messages = [
            messages[0],
            {
                "role": "user",
                "content": text_content(
                    build_speaking_behavior_user_prompt(persona_json)
                    + "\n\n上一版说话习惯过于默认化：\n"
                    + profile_json
                    + "\n请让停顿、节奏、句子推进、句尾与交流距离更明确，"
                    + "默认项最多 3 个。"
                ),
            },
        ]
        refined_profile = SpeakingBehavior.from_mapping(
            self._generate_json(retry_messages)
        )
        if refined_profile.neutral_dimension_count() > 3:
            selected_profile = min(
                (profile, refined_profile),
                key=lambda candidate: candidate.neutral_dimension_count(),
            )
            print(
                "警告：Speaking Behavior 两次均偏默认化，已保留两版中"
                "区分度较高的一版，并继续交由批量三层相似度检查。",
                file=sys.stderr,
            )
            return selected_profile
        return refined_profile

    def design_voice_fingerprint(
        self,
        visual_profile: VisualProfile,
        persona_profile: PersonaProfile,
        speaking_behavior: SpeakingBehavior,
    ) -> BaseVoiceProfile:
        """Stage 4: convert visual, persona and behavior into audible attributes."""
        visual_json = json.dumps(
            visual_profile.as_json_dict(), ensure_ascii=False, separators=(",", ":")
        )
        persona_json = json.dumps(
            persona_profile.as_json_dict(), ensure_ascii=False, separators=(",", ":")
        )
        behavior_json = json.dumps(
            speaking_behavior.as_json_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages = [
            {
                "role": "system",
                "content": text_content(VOICE_FINGERPRINT_SYSTEM_PROMPT),
            },
            {
                "role": "user",
                "content": text_content(
                    build_voice_fingerprint_user_prompt(
                        visual_json, persona_json, behavior_json
                    )
                ),
            },
        ]
        raw_profile = dict(self._generate_json(messages))
        raw_profile["vocal_maturity"] = self._consistent_vocal_maturity(
            raw_profile.get("vocal_maturity"), persona_profile
        )
        raw_profile.update(
            {
                "visual_age_style": visual_profile.visual_age_style,
                "voice_presentation": visual_profile.gender_presentation,
                "maturity": self._voice_maturity(persona_profile),
                "character_identity": persona_profile.identity,
                "speech_rate": speaking_behavior.speech_rate,
                "pause_pattern": speaking_behavior.pause_pattern,
                "rhythm_style": speaking_behavior.rhythm_style,
                "ending_style": speaking_behavior.ending_style,
                "vocal_distance": speaking_behavior.vocal_distance,
            }
        )
        profile = align_voice_to_visual_structure(
            visual_profile, BaseVoiceProfile.from_mapping(raw_profile)
        )
        if profile.neutral_dimension_count() <= 3:
            return profile

        profile_json = json.dumps(
            profile.as_json_dict(), ensure_ascii=False, separators=(",", ":")
        )
        refinement_messages = [
            {
                "role": "system",
                "content": text_content(VOICE_FINGERPRINT_SYSTEM_PROMPT),
            },
            {
                "role": "user",
                "content": text_content(
                    build_refinement_user_prompt(
                        visual_json, persona_json, behavior_json, profile_json
                    )
                ),
            },
        ]
        refined_data = dict(self._generate_json(refinement_messages))
        refined_data["vocal_maturity"] = self._consistent_vocal_maturity(
            refined_data.get("vocal_maturity"), persona_profile
        )
        refined_data.update(
            {
                "visual_age_style": visual_profile.visual_age_style,
                "voice_presentation": visual_profile.gender_presentation,
                "maturity": self._voice_maturity(persona_profile),
                "character_identity": persona_profile.identity,
                "speech_rate": speaking_behavior.speech_rate,
                "pause_pattern": speaking_behavior.pause_pattern,
                "rhythm_style": speaking_behavior.rhythm_style,
                "ending_style": speaking_behavior.ending_style,
                "vocal_distance": speaking_behavior.vocal_distance,
            }
        )
        refined_profile = align_voice_to_visual_structure(
            visual_profile, BaseVoiceProfile.from_mapping(refined_data)
        )
        if refined_profile.neutral_dimension_count() > 3:
            selected_profile = min(
                (profile, refined_profile),
                key=lambda candidate: candidate.neutral_dimension_count(),
            )
            print(
                "警告：Voice Fingerprint 两次均偏中性，已保留两版中"
                "中性字段更少的一版，并继续执行批量三层去重。",
                file=sys.stderr,
            )
            return selected_profile
        return refined_profile

    @staticmethod
    def _voice_maturity(persona_profile: PersonaProfile) -> str:
        if persona_profile.age_style <= 18:
            return "youthful"
        if persona_profile.age_style >= MATURE_ADULT_AGE:
            return "mature"
        return "young"

    @staticmethod
    def _default_vocal_maturity(persona_profile: PersonaProfile) -> str:
        age = persona_profile.age_style
        if age <= 18:
            return "youthful"
        if age <= 25:
            return "young_adult"
        if age < MATURE_ADULT_AGE:
            return "mature_young"
        if age <= 55:
            return "mature"
        return "seasoned"

    @classmethod
    def _consistent_vocal_maturity(
        cls, requested: Any, persona_profile: PersonaProfile
    ) -> str:
        """Keep creative maturity choices inside a visually plausible band."""
        age = persona_profile.age_style
        allowed_by_age = (
            ("youthful", "young_adult")
            if age <= 18
            else ("youthful", "young_adult", "mature_young")
            if age <= 25
            else ("young_adult", "mature_young")
            if age < MATURE_ADULT_AGE
            else ("mature_young", "mature")
            if age <= 50
            else ("mature", "seasoned")
        )
        normalized = str(requested).strip() if requested is not None else ""
        if normalized in allowed_by_age:
            return normalized
        return cls._default_vocal_maturity(persona_profile)

    def analyze_base_voice(self, image_path: str) -> BaseVoiceProfile:
        """Compatibility wrapper executing all structured design stages."""
        visual_profile = self.analyze_visual_profile(image_path)
        persona_profile = self.analyze_persona_profile(visual_profile)
        speaking_behavior = self.analyze_speaking_behavior(persona_profile)
        return self.design_voice_fingerprint(
            visual_profile, persona_profile, speaking_behavior
        )

    def apply_voice_instruction(
        self, instruction: str, base_profile: BaseVoiceProfile
    ) -> BaseVoiceProfile:
        """Apply only stable voice properties explicitly requested by the user."""
        normalized_instruction = instruction.strip()
        if not normalized_instruction:
            return base_profile

        base_json = json.dumps(
            base_profile.as_json_dict(), ensure_ascii=False, separators=(",", ":")
        )
        messages = [
            {
                "role": "system",
                "content": text_content(VOICE_OVERRIDE_SYSTEM_PROMPT),
            },
            {
                "role": "user",
                "content": text_content(
                    build_voice_override_user_prompt(
                        base_json, normalized_instruction
                    )
                ),
            },
        ]
        model_overrides = dict(self._generate_json(messages))
        # Literal, unambiguous requests take precedence so age, gender and
        # nasality cannot disappear if the model under-extracts the sentence.
        model_overrides.update(
            infer_literal_voice_overrides(normalized_instruction)
        )
        return apply_voice_overrides(base_profile, model_overrides)

    def ensure_batch_contrast(
        self,
        base_profile: BaseVoiceProfile,
        reference_designs: list[VoiceDesignReference],
        *,
        visual_profile: VisualProfile,
        persona_profile: PersonaProfile,
        speaking_behavior: SpeakingBehavior,
        instruction: str = "",
        minimum_differences: int = 5,
        minimum_strong_differences: int = 3,
        max_redesign_attempts: int = 3,
    ) -> tuple[PersonaProfile, SpeakingBehavior, BaseVoiceProfile]:
        """Resolve batch collisions by redesigning persona before acoustics."""
        if not reference_designs:
            return persona_profile, speaking_behavior, base_profile

        candidate_persona = persona_profile
        candidate_behavior = speaking_behavior
        candidate_voice = base_profile
        for attempt in range(max_redesign_attempts + 1):
            failing_references = [
                reference
                for reference in reference_designs
                if (
                    not profile_meets_contrast(
                        candidate_voice,
                        [reference.voice_fingerprint],
                        minimum_differences=minimum_differences,
                        minimum_strong_differences=minimum_strong_differences,
                    )
                    or layered_design_is_too_similar(
                        candidate_persona,
                        candidate_behavior,
                        candidate_voice,
                        reference,
                    )
                )
            ]
            if not failing_references:
                return candidate_persona, candidate_behavior, candidate_voice
            if attempt == max_redesign_attempts:
                layered_collisions = [
                    reference
                    for reference in failing_references
                    if layered_design_is_too_similar(
                        candidate_persona,
                        candidate_behavior,
                        candidate_voice,
                        reference,
                    )
                ]
                if layered_collisions:
                    closest_reference = max(
                        layered_collisions,
                        key=lambda reference: layered_design_similarity(
                            candidate_persona,
                            candidate_behavior,
                            candidate_voice,
                            reference,
                        ),
                    )
                    similarity = layered_design_similarity(
                        candidate_persona,
                        candidate_behavior,
                        candidate_voice,
                        closest_reference,
                    )
                    print(
                        "警告：批量角色三层设计多次重构后仍相似"
                        f"（综合相似度 {similarity:.2f}），"
                        "将使用受约束的声学分配器完成最终去重。",
                        file=sys.stderr,
                    )

                reference_profiles = [
                    reference.voice_fingerprint for reference in reference_designs
                ]
                locked_fields = infer_literal_voice_overrides(instruction).keys()
                allocated = allocate_contrasting_profile(
                    candidate_voice,
                    reference_profiles,
                    minimum_differences=minimum_differences,
                    minimum_strong_differences=minimum_strong_differences,
                    locked_fields=locked_fields,
                    visual_profile=(visual_profile if not instruction.strip() else None),
                )
                if allocated is not None:
                    return candidate_persona, candidate_behavior, allocated
                closest_reference = min(
                    failing_references,
                    key=lambda reference: weighted_acoustic_distance(
                        candidate_voice, reference.voice_fingerprint
                    ),
                )
                differences, strong_differences = acoustic_contrast(
                    candidate_voice, closest_reference.voice_fingerprint
                )
                weighted_distance = weighted_acoustic_distance(
                    candidate_voice, closest_reference.voice_fingerprint
                )
                demographic_core_differences = similar_demographic_core_contrast(
                    candidate_voice, closest_reference.voice_fingerprint
                )
                macro_axes, macro_distance = macro_voice_contrast(
                    candidate_voice, closest_reference.voice_fingerprint
                )
                timbre_differences = timbre_identity_contrast(
                    candidate_voice, closest_reference.voice_fingerprint
                )
                (
                    primary_timbre_differences,
                    primary_timbre_strong_differences,
                    primary_timbre_groups,
                ) = primary_timbre_contrast(
                    candidate_voice, closest_reference.voice_fingerprint
                )
                raise ValueError(
                    "批量声纹碰撞在重新设计后仍未解决："
                    f"最近参考仅有 {differences} 个核心差异、"
                    f"{strong_differences} 个明显差异，加权距离为 "
                    f"{weighted_distance:.2f}，相近年龄/性别核心差异为 "
                    f"{demographic_core_differences}，宏观声线差异轴为 "
                    f"{macro_axes}（宏观距离 {macro_distance:.2f}），核心音色差异为 "
                    f"{timbre_differences}；直接音色指标差异为 "
                    f"{primary_timbre_differences}，其中明显差异 "
                    f"{primary_timbre_strong_differences} 项、覆盖 "
                    f"{primary_timbre_groups} 组。"
                )

            closest_references = sorted(
                failing_references,
                key=lambda reference: layered_design_similarity(
                    candidate_persona,
                    candidate_behavior,
                    candidate_voice,
                    reference,
                ),
                reverse=True,
            )[:3]
            voice_json = json.dumps(
                candidate_voice.as_json_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            references_json = json.dumps(
                [
                    {
                        "persona_profile": (
                            reference.persona_profile.as_json_dict()
                            if reference.persona_profile
                            else None
                        ),
                        "speaking_behavior": (
                            reference.speaking_behavior.as_json_dict()
                            if reference.speaking_behavior
                            else None
                        ),
                        "voice_fingerprint": (
                            reference.voice_fingerprint.as_json_dict()
                        ),
                    }
                    for reference in closest_references
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            visual_json = json.dumps(
                visual_profile.as_json_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            persona_json = json.dumps(
                candidate_persona.as_json_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            behavior_json = json.dumps(
                candidate_behavior.as_json_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            messages = [
                {
                    "role": "system",
                    "content": text_content(CONTRASTIVE_PERSONA_SYSTEM_PROMPT),
                },
                {
                    "role": "user",
                    "content": text_content(
                        build_contrastive_persona_user_prompt(
                            visual_json,
                            persona_json,
                            behavior_json,
                            voice_json,
                            references_json,
                        )
                    ),
                },
            ]
            redesigned_persona_data = dict(self._generate_json(messages))
            redesigned_persona_data.update(
                {
                    "gender": visual_profile.gender_presentation,
                    "age_style": visual_profile.visual_age_style,
                }
            )
            redesigned_persona = PersonaProfile.from_mapping(
                redesigned_persona_data
            )
            redesigned_behavior = self.analyze_speaking_behavior(
                redesigned_persona
            )
            redesigned_voice = self.design_voice_fingerprint(
                visual_profile,
                redesigned_persona,
                redesigned_behavior,
            )
            if instruction.strip():
                redesigned_voice = self.apply_voice_instruction(
                    instruction, redesigned_voice
                )
            candidate_persona = redesigned_persona
            candidate_behavior = redesigned_behavior
            candidate_voice = redesigned_voice

        raise RuntimeError("批量声线去重流程未正常结束。")

    def analyze_delivery(
        self, instruction: str, base_profile: BaseVoiceProfile
    ) -> DeliveryProfile | None:
        normalized_instruction = instruction.strip()
        if not normalized_instruction:
            return None

        base_json = json.dumps(
            base_profile.as_json_dict(), ensure_ascii=False, separators=(",", ":")
        )
        messages = [
            {
                "role": "system",
                "content": text_content(DELIVERY_SYSTEM_PROMPT),
            },
            {
                "role": "user",
                "content": text_content(
                    build_delivery_user_prompt(base_json, normalized_instruction)
                ),
            },
        ]
        return DeliveryProfile.from_mapping(self._generate_json(messages))

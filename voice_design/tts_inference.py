"""Resident Qwen3-TTS VoiceDesign inference powered by vLLM-Omni 0.28.x."""

from __future__ import annotations

import copy
import importlib
import os
import random
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VLLM_OMNI_SOURCE_PATH = (
    PROJECT_ROOT / "vllm-omni"
    if (PROJECT_ROOT / "vllm-omni").is_dir()
    else PROJECT_ROOT / "vllm-omni-0.28.0"
)
DEFAULT_QWEN3_TTS_DEPLOY_CONFIG = (
    PROJECT_ROOT / "voice_design" / "qwen3_tts_low_memory.yaml"
)
REQUIRED_VLLM_OMNI_VERSION_PREFIX = "0.28."


def load_project_vllm_omni(
    source_path: str | os.PathLike[str] | None = None,
) -> tuple[Any, Path, str]:
    """Load installed vLLM-Omni 0.28 and verify the selected source tree."""
    resolved_source = Path(
        source_path or DEFAULT_VLLM_OMNI_SOURCE_PATH
    ).expanduser().resolve()
    package_init = resolved_source / "vllm_omni" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(
            "vLLM-Omni 源码目录无效："
            f"{resolved_source}（缺少 vllm_omni/__init__.py）"
        )
    try:
        module = importlib.import_module("vllm_omni")
    except ImportError as exc:
        raise RuntimeError(
            "当前环境尚未安装 vLLM-Omni。请先从项目内 vllm-omni 目录执行"
            "可编辑安装；仅上传源码目录不能直接运行 CUDA 推理。"
        ) from exc

    version = str(getattr(module, "__version__", "unknown"))
    if not version.startswith(REQUIRED_VLLM_OMNI_VERSION_PREFIX):
        raise RuntimeError(
            f"当前 vLLM-Omni 版本为 {version}，本项目要求 0.28.x。"
        )
    try:
        vllm_module = importlib.import_module("vllm")
    except ImportError as exc:
        raise RuntimeError("vLLM-Omni 需要先安装 vLLM 0.28.x。") from exc
    vllm_version = str(getattr(vllm_module, "__version__", "unknown"))
    if not vllm_version.startswith(REQUIRED_VLLM_OMNI_VERSION_PREFIX):
        raise RuntimeError(
            f"当前 vLLM 版本为 {vllm_version}，必须与 vLLM-Omni 同为 0.28.x。"
        )
    return module, resolved_source, version


class VLLMOmniVoiceDesignTTS:
    """Load Qwen3-TTS once through vLLM-Omni and reuse it for requests."""

    def __init__(
        self,
        model_path: str,
        *,
        omni_source_path: str | os.PathLike[str] | None = None,
        deploy_config: str | os.PathLike[str] | None = None,
        max_new_tokens: int = 2048,
    ) -> None:
        resolved_model = Path(model_path).expanduser().resolve()
        self._validate_model(resolved_model)
        if max_new_tokens <= 0:
            raise ValueError("TTS max_new_tokens 必须大于 0。")

        module, self.omni_source_path, self.omni_version = (
            load_project_vllm_omni(omni_source_path)
        )
        resolved_deploy_config = Path(
            deploy_config or DEFAULT_QWEN3_TTS_DEPLOY_CONFIG
        ).expanduser().resolve()
        if not resolved_deploy_config.is_file():
            raise FileNotFoundError(
                f"Qwen3-TTS vLLM-Omni 部署配置不存在：{resolved_deploy_config}"
            )

        try:
            import torch
            from transformers import AutoTokenizer
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
                Qwen3TTSConfig,
            )
            from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
                Qwen3TTSPromptEmbedsBuilder,
            )
        except ImportError as exc:
            raise RuntimeError("vLLM-Omni 的 Qwen3-TTS 运行依赖不完整。") from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(resolved_model),
            trust_remote_code=True,
            padding_side="left",
            local_files_only=True,
        )
        tts_config = Qwen3TTSConfig.from_pretrained(
            str(resolved_model), trust_remote_code=True, local_files_only=True
        )
        talker_config = getattr(tts_config, "talker_config", None)
        self._codec_language_id = getattr(talker_config, "codec_language_id", None)
        self._spk_is_dialect = getattr(talker_config, "spk_is_dialect", None)
        self._prompt_builder = Qwen3TTSPromptEmbedsBuilder
        self.model = module.Omni(
            model=str(resolved_model),
            deploy_config=str(resolved_deploy_config),
            log_stats=False,
        )

    @staticmethod
    def _validate_model(model_path: Path) -> None:
        required_files = (
            model_path / "config.json",
            model_path / "speech_tokenizer" / "config.json",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "本地 VoiceDesign 模型目录不完整：\n" + "\n".join(missing)
            )

    def _set_seed(self, seed: int) -> None:
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass
        random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    def _estimate_prompt_len(self, additional_information: dict[str, Any]) -> int:
        return int(
            self._prompt_builder.estimate_prompt_len_from_additional_information(
                additional_information,
                task_type="VoiceDesign",
                tokenize_prompt=lambda text: self.tokenizer(
                    text, padding=False
                )["input_ids"],
                codec_language_id=self._codec_language_id,
                spk_is_dialect=self._spk_is_dialect,
            )
        )

    @staticmethod
    def _extract_audio(output: Any) -> tuple[Any, int]:
        request_output = output[-1] if isinstance(output, list) else output
        stage_outputs = getattr(request_output, "outputs", None)
        if not stage_outputs:
            raise RuntimeError("vLLM-Omni 没有返回 TTS stage 输出。")
        multimodal_output = getattr(stage_outputs[0], "multimodal_output", None)
        if not multimodal_output or "audio" not in multimodal_output:
            raise RuntimeError("vLLM-Omni 没有返回音频数据。")
        sample_rate_raw = multimodal_output.get("sr", 0)
        if isinstance(sample_rate_raw, list):
            sample_rate_raw = sample_rate_raw[-1] if sample_rate_raw else 0
        sample_rate = (
            int(sample_rate_raw.item())
            if hasattr(sample_rate_raw, "item")
            else int(sample_rate_raw)
        )
        if sample_rate <= 0:
            raise RuntimeError("vLLM-Omni 返回了无效采样率。")
        return multimodal_output["audio"], sample_rate

    def generate(
        self,
        *,
        text: str,
        instruct: str,
        output_path: str | Path,
        seed: int,
        language: str = "Chinese",
    ) -> Path:
        target_text = text.strip()
        voice_instruct = instruct.strip()
        if not target_text:
            raise ValueError("朗读文本不能为空。")
        if not voice_instruct:
            raise ValueError("声线 instruct 不能为空。")

        self._set_seed(seed)
        additional_information = {
            "task_type": ["VoiceDesign"],
            "text": [target_text],
            "language": [language],
            "instruct": [voice_instruct],
            "max_new_tokens": [self.max_new_tokens],
            "non_streaming_mode": [True],
        }
        inputs = {
            "prompt_token_ids": [0]
            * self._estimate_prompt_len(additional_information),
            "additional_information": additional_information,
        }
        sampling_params_list = copy.deepcopy(self.model.default_sampling_params_list)
        if sampling_params_list:
            stage_zero = sampling_params_list[0]
            stage_zero.seed = seed
            if getattr(stage_zero, "extra_args", None) is None:
                stage_zero.extra_args = {}
            stage_zero.extra_args["tts_local_seed"] = seed
        outputs = self.model.generate(
            inputs,
            sampling_params_list=sampling_params_list,
            use_tqdm=False,
        )
        audio_data, sample_rate = self._extract_audio(outputs)
        if isinstance(audio_data, list):
            if not audio_data:
                raise RuntimeError("vLLM-Omni 返回了空音频。")
            audio_tensor = self.torch.cat(audio_data, dim=-1)
        else:
            audio_tensor = audio_data

        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("缺少 soundfile，无法保存 WAV。") from exc
        resolved_output = Path(output_path).expanduser().resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        waveform = audio_tensor.float().cpu().numpy().flatten()
        sf.write(str(resolved_output), waveform, sample_rate, format="WAV")
        if not resolved_output.is_file() or resolved_output.stat().st_size == 0:
            raise RuntimeError(f"生成的 WAV 无效：{resolved_output}")
        return resolved_output

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()


# Keep the old import name working for existing project code.
ResidentVoiceDesignTTS = VLLMOmniVoiceDesignTTS

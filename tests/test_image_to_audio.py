import tempfile
import types
import unittest
from pathlib import Path

from run_image_to_audio import (
    resolve_output_paths,
    resolve_target_text,
    stable_image_seed,
)
from voice_design.tts_inference import VLLMOmniVoiceDesignTTS


class PipelineCommandTests(unittest.TestCase):
    def test_same_image_stem_is_used_for_txt_and_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "周玉琼.png"
            image.write_bytes(b"mock image")
            instruct_path, audio_path = resolve_output_paths(image, root / "output")

        self.assertEqual(instruct_path.name, "周玉琼.txt")
        self.assertEqual(audio_path.name, "周玉琼.wav")

    def test_text_file_is_read_and_trimmed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            text_file = Path(temp_dir) / "朗读.txt"
            text_file.write_text("  需要朗读的文本。\n", encoding="utf-8")
            self.assertEqual(resolve_target_text(None, str(text_file)), "需要朗读的文本。")

    def test_tts_seed_is_stable_per_image_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.png"
            renamed_copy = root / "renamed.png"
            different = root / "different.png"
            first.write_bytes(b"same image bytes")
            renamed_copy.write_bytes(b"same image bytes")
            different.write_bytes(b"different image bytes")

            self.assertEqual(stable_image_seed(first), stable_image_seed(renamed_copy))
            self.assertNotEqual(stable_image_seed(first), stable_image_seed(different))

    def test_vllm_omni_audio_output_accepts_streamed_sample_rate(self):
        multimodal = {"audio": ["chunk"], "sr": [24_000]}
        output = types.SimpleNamespace(
            outputs=[types.SimpleNamespace(multimodal_output=multimodal)]
        )
        audio, sample_rate = VLLMOmniVoiceDesignTTS._extract_audio([output])
        self.assertEqual(audio, ["chunk"])
        self.assertEqual(sample_rate, 24_000)


if __name__ == "__main__":
    unittest.main()

import argparse
import tempfile
import unittest
from pathlib import Path

from run_batch_image_to_audio import (
    SINGLE_IMAGE_ENTRY,
    build_single_image_command,
    discover_images,
    find_duplicate_images,
    make_batch_items,
    prepare_profile_registry,
    remove_stale_outputs,
    resolve_output_dir,
)
from run_voice_design import DEFAULT_OUTPUT_DIR


class BatchPipelineTests(unittest.TestCase):
    def test_discovers_multiple_formats_and_unicode_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("徐知瑶.PNG", "Alice.jpeg", "角色.webp", "忽略.txt"):
                (root / name).write_bytes(b"test")
            nested = root / "nested"
            nested.mkdir()
            (nested / "子目录人物.jpg").write_bytes(b"test")

            top_level = discover_images(str(root))
            recursive = discover_images(str(root), recursive=True)

        self.assertEqual(
            {path.name for path in top_level}, {"徐知瑶.PNG", "Alice.jpeg", "角色.webp"}
        )
        self.assertEqual(
            {path.name for path in recursive},
            {"徐知瑶.PNG", "Alice.jpeg", "角色.webp", "子目录人物.jpg"},
        )

    def test_rejects_duplicate_output_stems(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "同名.jpg").write_bytes(b"test")
            (root / "同名.png").write_bytes(b"test")

            with self.assertRaisesRegex(ValueError, "相同的输出文件名"):
                discover_images(str(root))

    def test_only_user_inputs_are_forwarded_to_single_image_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "Chinese中文 portrait人像.tiff"
            image.write_bytes(b"test")
            output_dir = Path(temp_dir) / "output" / "人物甲"
            item = make_batch_items([image.resolve()], output_dir)[0]
            args = argparse.Namespace(
                text="统一朗读文本",
                text_file=None,
                instruction="使用非常愤怒的语气说话。",
                attn_implementation="flash_attention_2",
            )

            command = build_single_image_command(args, item)

        self.assertEqual(command[1], str(SINGLE_IMAGE_ENTRY))
        self.assertEqual(
            command[2:],
            [
                "--image",
                str(item.image_path),
                "--output-dir",
                str(output_dir),
                "--text",
                "统一朗读文本",
                "--instruction",
                "使用非常愤怒的语气说话。",
                "--attn-implementation",
                "flash_attention_2",
            ],
        )
        self.assertEqual(item.instruct_path.name, "Chinese中文 portrait人像.txt")
        self.assertEqual(item.audio_path.name, "Chinese中文 portrait人像.wav")

    def test_batch_registry_is_forwarded_for_cross_image_contrast(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "人物.png"
            image.write_bytes(b"test")
            output_dir = root / "output"
            item = make_batch_items([image.resolve()], output_dir)[0]
            registry = prepare_profile_registry(output_dir, keep_existing=False)
            args = argparse.Namespace(
                text="统一文本", text_file=None, instruction=""
            )

            command = build_single_image_command(args, item, registry)

        registry_index = command.index("--contrast-profile-registry")
        self.assertEqual(command[registry_index + 1], str(registry))

    def test_overwrite_mode_removes_stale_pair_before_rerun(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "人物.png"
            image.write_bytes(b"image")
            item = make_batch_items([image.resolve()], root / "output")[0]
            item.instruct_path.write_text("old", encoding="utf-8")
            item.audio_path.write_bytes(b"old wav")

            remove_stale_outputs(item)

            self.assertFalse(item.instruct_path.exists())
            self.assertFalse(item.audio_path.exists())

    def test_person_subfolder_gets_matching_output_subfolder(self):
        person_input = Path("/data/user/Qwen3-VL/data/徐若溪")
        output_dir = resolve_output_dir(str(person_input))
        self.assertEqual(output_dir.name, "徐若溪")
        self.assertEqual(output_dir.parent, DEFAULT_OUTPUT_DIR.resolve())

    def test_identical_images_are_reported_as_duplicate_controls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "徐若溪_09.png"
            second = root / "徐若溪_23.png"
            different = root / "徐若溪_01.png"
            first.write_bytes(b"same image")
            second.write_bytes(b"same image")
            different.write_bytes(b"different image")

            groups = find_duplicate_images([first, second, different])

        self.assertEqual(len(groups), 1)
        self.assertEqual({path.name for path in groups[0]}, {first.name, second.name})


if __name__ == "__main__":
    unittest.main()

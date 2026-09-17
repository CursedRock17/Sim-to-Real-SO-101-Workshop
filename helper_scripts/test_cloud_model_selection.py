"""Regression checks for bounded GR00T checkpoint downloads.

Run with the GR00T environment: python -m unittest discover -s helper_scripts
"""

import tempfile
import unittest
from fnmatch import fnmatch
from pathlib import Path
from unittest import mock

import run_gr00t_server as server


class ModelSelectionTest(unittest.TestCase):
    """Ensure model selection never fetches unrelated training checkpoints."""

    def test_explicit_checkpoint_filters_training_state_and_other_models(self) -> None:
        """Only the requested checkpoint's inference assets pass the filters."""
        files = [
            "config.json",
            "model.safetensors",
            "checkpoint-1000/config.json",
            "checkpoint-1000/model.safetensors",
            "checkpoint-30000/config.json",
            "checkpoint-30000/processor_config.json",
            "checkpoint-30000/statistics.json",
            "checkpoint-30000/model-00001-of-00002.safetensors",
            "checkpoint-30000/model-00002-of-00002.safetensors",
            "checkpoint-30000/model.safetensors.index.json",
            "checkpoint-30000/optimizer.pt",
            "checkpoint-30000/scheduler.pt",
            "checkpoint-30000/rng_state.pth",
            "checkpoint-30000/training_args.bin",
        ]
        selected = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def download(repo_id: str, **kwargs: object) -> str:
                del repo_id
                for name in files:
                    if any(
                        fnmatch(name, pattern) for pattern in kwargs["allow_patterns"]
                    ) and not any(
                        fnmatch(name, pattern) for pattern in kwargs["ignore_patterns"]
                    ):
                        selected.append(name)
                        path = root / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
                return str(root)

            with mock.patch.object(server, "snapshot_download", side_effect=download):
                resolved = server._resolve_model_path(
                    "org/model", checkpoint="checkpoint-30000"
                )
            self.assertEqual(resolved, str(root / "checkpoint-30000"))
        self.assertEqual(selected, files[4:10])

    def test_missing_checkpoint_fails_without_falling_back(self) -> None:
        """A root model must not silently replace the requested checkpoint."""
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").touch()
            with self.assertRaises(FileNotFoundError):
                server._resolve_model_path(directory, checkpoint="checkpoint-30000")

    def test_offline_selection_does_not_list_remote_repository(self) -> None:
        """An explicit cached checkpoint remains usable without network access."""
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-30000"
            checkpoint.mkdir()
            (checkpoint / "config.json").touch()
            with (
                mock.patch.object(server, "list_repo_files") as listing,
                mock.patch.object(
                    server, "snapshot_download", return_value=directory
                ) as download,
            ):
                resolved = server._resolve_model_path(
                    "org/model", checkpoint="checkpoint-30000", offline=True
                )
            listing.assert_not_called()
            self.assertTrue(download.call_args.kwargs["local_files_only"])
            self.assertEqual(resolved, str(checkpoint))

    def test_rejects_ambiguous_or_invalid_checkpoint(self) -> None:
        """Bad selections fail before invoking the downloader."""
        with mock.patch.object(server, "snapshot_download") as download:
            with self.assertRaises(ValueError):
                server._resolve_model_path(
                    "org/model", checkpoint="../checkpoint-30000"
                )
            with self.assertRaises(ValueError):
                server._resolve_model_path(
                    "org/model", auto_checkpoint=True, checkpoint="checkpoint-30000"
                )
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()

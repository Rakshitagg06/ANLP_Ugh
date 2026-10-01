"""Optional Weights & Biases integration with a no-op fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List


LOGGER = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, config: Dict[str, Any], run_name: str) -> None:
        self.run = None
        self._wandb = None
        tracking = config["logging"].get("wandb", {})
        if not tracking.get("enabled", False):
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B is enabled but not installed; install with pip install -e '.[wandb]'") from exc

        mode = tracking.get("mode") or os.environ.get("WANDB_MODE", "online")
        self._wandb = wandb
        self.run = wandb.init(
            project=tracking.get("project", "polar-hierarchy"),
            entity=tracking.get("entity") or None,
            name=run_name,
            config={key: value for key, value in config.items() if not key.startswith("_")},
            mode=mode,
            tags=tracking.get("tags", []),
        )

    def log(self, values: Dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            try:
                self.run.log(values, step=step)
            except Exception as exc:  # W&B/network failures must not invalidate training outputs.
                LOGGER.warning("W&B metric logging failed; continuing with local outputs: %s", exc)

    def log_images(self, values: Dict[str, str | Path]) -> None:
        if self.run is None or self._wandb is None:
            return
        images = {
            key: self._wandb.Image(str(path))
            for key, path in values.items()
            if Path(path).is_file()
        }
        if images:
            try:
                self.run.log(images)
            except Exception as exc:  # W&B/network failures must not invalidate training outputs.
                LOGGER.warning("W&B image logging failed; plots remain available locally: %s", exc)

    def set_summary(self, values: Dict[str, Any]) -> None:
        if self.run is not None:
            try:
                self.run.summary.update(values)
            except Exception as exc:  # W&B/network failures must not invalidate training outputs.
                LOGGER.warning("W&B summary update failed: %s", exc)

    def log_files_artifact(self, name: str, artifact_type: str, paths: List[str | Path]) -> None:
        """Upload small run outputs (predictions, metrics) so W&B holds a full copy."""
        if self.run is None or self._wandb is None:
            return
        try:
            artifact = self._wandb.Artifact(name, type=artifact_type)
            for path in paths:
                if Path(path).is_file():
                    artifact.add_file(str(path))
            self.run.log_artifact(artifact)
        except Exception as exc:  # W&B/network failures must not invalidate training outputs.
            LOGGER.warning("W&B artifact upload failed; files remain available locally: %s", exc)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as exc:  # W&B/network failures must not invalidate training outputs.
                LOGGER.warning("W&B finalization failed; continuing with local outputs: %s", exc)

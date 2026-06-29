from pathlib import Path
import logging

from pytest_iree.artifact import Artifact

logger = logging.getLogger(__name__)


class HuggingFaceArtifact(Artifact):
    """Represents an artifact that can be downloaded from the Hugging Face Hub."""

    def __init__(
        self,
        repo_id: str,
        filename: str,
        revision: str,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.revision = revision
        self._path: Path | None = None

    @property
    def path(self) -> Path:
        assert (
            self._path is not None
        ), "HuggingFaceArtifact.join() must be called before accessing 'path'."
        return self._path

    def join(self) -> None:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            revision=self.revision,
            repo_type="model",
        )
        self._path = Path(local_path)
        logger.info(
            f"  Using HuggingFace artifact '{self.repo_id}/{self.filename}'"
            f"@{self.revision} at '{self._path}'"
        )

    def __str__(self):
        return (
            f"https://huggingface.co/{self.repo_id}"
            f"/resolve/{self.revision}/{self.filename}"
        )

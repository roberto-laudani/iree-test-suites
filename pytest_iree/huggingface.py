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
        revision: str | None = None,
        repo_type: str = "model",
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.revision = revision
        self.repo_type = repo_type
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
            repo_type=self.repo_type,
        )
        self._path = Path(local_path)
        logger.info(
            f"  Using HuggingFace artifact '{self.repo_id}/{self.filename}'"
            f"{f'@{self.revision}' if self.revision else ''} at '{self._path}'"
        )

    def __str__(self):
        return f"hf://{self.repo_id}/{self.filename}"

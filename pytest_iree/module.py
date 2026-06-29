from pathlib import Path
import logging
import json
from pytest_iree.artifact import Artifact
from pytest_iree.azure import AzureArtifact
from pytest_iree.utils import iree_compile
import shutil

logger = logging.getLogger(__name__)


class ModuleArtifact(Artifact):
    """Represents a vmfb module artifact that can be loaded by IREE. Each
    artifact's uniqueness is defined by it's module name. The module definition
    can be found at `<module_base_dir>/Path(<module>).json`.

    The artifact creates a directory structure under the module_artifact_base_dir as:
    module_artifact_base_dir/
        modules/
            <module1>/
                <module1>.vmfb
            <module2>
                <module2>.vmfb
                ...
    """

    def __init__(
        self,
        artifact_base_dir: Path,
        module_artifact_base_dir: Path,
        module_base_dir: Path,
        module: str,
        external_file_dir: Path,
    ):
        module_artifact_dir = module_artifact_base_dir / "modules" / Path(module)
        name = "module.vmfb"
        module_json = (module_base_dir / Path(module)).with_suffix(".json")
        assert (
            module_json.exists()
        ), f"Module definition '{module_json}' does not exist."
        super().__init__(module_artifact_dir, name)

        self.artifact_base_dir = artifact_base_dir
        self.module_name = module
        self.module_json = module_json
        self.module_artifact_dir = module_artifact_dir
        self.external_file_dir = external_file_dir
        self.compstat_path = module_artifact_dir / "compilation_info.json"

    def _needs_recompile(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return True
        if not self.compstat_path.exists() or self.compstat_path.stat().st_size == 0:
            return True
        return False

    def join(self):
        super().join()
        # Check if the module file already exists.
        if not self._needs_recompile():
            logger.info(f"  Skipping '{self.path}' download - file exists")
            return
        # Compile the MLIR file to a VMFB file.
        mlir_path = self.get_mlir_path()
        module_data = json.loads(self.module_json.read_text())
        compiler_flags = module_data.get("compiler_flags", [])
        # Add compilation stats path and format.
        compiler_flags += [
            f"--iree-scheduling-dump-statistics-file={self.compstat_path.resolve()}",
            "--iree-scheduling-dump-statistics-format=json",
        ]
        iree_compile(
            source=mlir_path,
            output=self.path,
            cwd=self.external_file_dir,
            args=compiler_flags,
        )

    def get_compstats(self) -> dict:
        """Get the compilation statistics from the compilation_info.json file."""
        self.join()
        assert (
            self.compstat_path.exists()
        ), f"Compilation stats file '{self.compstat_path}' does not exist."
        compstats = json.loads(self.compstat_path.read_text())
        return compstats

    def get_mlir_path(self) -> Path:
        """Get the path to the MLIR file used to generate this module.

        The "mlir" field may be either an Azure Blob Storage URL (downloaded and
        cached) or a local filesystem path. A relative local path is resolved
        against the external file directory.
        """
        module_data = json.loads(self.module_json.read_text())
        mlir_ref = module_data["mlir"]
        if mlir_ref.startswith("https://"):
            assert (
                "blob.core.windows.net" in mlir_ref
            ), f"Only Azure Blob Storage URLs are supported, got '{mlir_ref}'"
            self.mlir_artifact = AzureArtifact(self.artifact_base_dir, mlir_ref)
            self.mlir_artifact.join()
            return self.mlir_artifact.path

        mlir_path = Path(mlir_ref)
        if not mlir_path.is_absolute():
            mlir_path = self.external_file_dir / mlir_path
        mlir_path = mlir_path.resolve()
        assert mlir_path.exists(), f"Local MLIR file '{mlir_path}' does not exist."
        return mlir_path

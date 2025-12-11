"""Client wrapper for delegating manifest generation to Blueprint Sim."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class BlueprintSimResult:
    """Normalized result returned from the shared simulation pipeline."""

    scene_usd_path: str
    manifest_path: str
    recipe_path: str
    layer_paths: dict[str, str] = field(default_factory=dict)
    replicator_bundle: Optional[str] = None
    isaac_lab_bundle: Optional[str] = None
    qa_report: dict[str, Any] | None = None


class BlueprintSimClient:
    """Thin wrapper around the shared Blueprint Sim pipeline.

    The client defers manifest-driven scene generation to the shared
    implementation while keeping NVIDIA-specific preparation inside
    BlueprintRecipe. A custom pipeline implementation may be injected for
    testing.
    """

    def __init__(self, pipeline: Any | None = None):
        self._pipeline = pipeline or self._load_pipeline()

    def generate_from_manifest(
        self,
        manifest: dict[str, Any],
        output_dir: str | Path,
        policies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BlueprintSimResult:
        """Invoke the shared pipeline with a canonical manifest."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        call_args = {
            "manifest": manifest,
            "output_dir": str(output_path),
            "policies": policies or [],
            "metadata": metadata or {},
        }

        if hasattr(self._pipeline, "generate_from_manifest"):
            raw_result = self._pipeline.generate_from_manifest(**call_args)
        elif hasattr(self._pipeline, "run_with_manifest"):
            # Legacy name compatibility
            raw_result = self._pipeline.run_with_manifest(**call_args)
        else:  # pragma: no cover - defensive branch
            raise RuntimeError("Blueprint Sim pipeline missing manifest entrypoint")

        return self._normalize_result(raw_result, output_path)

    def _normalize_result(self, result: Any, output_dir: Path) -> BlueprintSimResult:
        """Coerce arbitrary pipeline responses into a BlueprintSimResult."""

        if isinstance(result, BlueprintSimResult):
            return result

        if isinstance(result, dict):
            return BlueprintSimResult(
                scene_usd_path=str(
                    result.get("scene_usd")
                    or result.get("scene_path")
                    or result.get("scene_usd_path")
                    or result.get("scene")
                    or (output_dir / "scene.usda")
                ),
                manifest_path=str(
                    result.get("manifest_path")
                    or result.get("manifest")
                    or (output_dir / "scene_manifest.json")
                ),
                recipe_path=str(
                    result.get("recipe_path")
                    or result.get("recipe")
                    or (output_dir / "recipe.json")
                ),
                layer_paths=result.get("layer_paths", {}),
                replicator_bundle=result.get("replicator_bundle"),
                isaac_lab_bundle=result.get("isaac_lab_bundle"),
                qa_report=result.get("qa_report"),
            )

        raise RuntimeError("Unsupported Blueprint Sim result type")

    def _load_pipeline(self) -> Any:
        """Lazily import the shared pipeline implementation."""

        try:
            from blueprint_sim import pipeline as sim_pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "blueprint_sim package is required for simulation delegation"
            ) from exc

        for candidate in (
            "BlueprintSimPipeline",
            "SharedSimPipeline",
            "BlueprintPipeline",
        ):
            pipeline_cls: Optional[Callable[..., Any]] = getattr(
                sim_pipeline, candidate, None
            )
            if pipeline_cls:
                return pipeline_cls()

        # Fallback to module-level factory if available
        if hasattr(sim_pipeline, "get_pipeline"):
            return sim_pipeline.get_pipeline()

        raise RuntimeError("Unable to locate Blueprint Sim pipeline implementation")

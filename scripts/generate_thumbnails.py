"""Generate thumbnails for assets using Isaac Sim headless rendering.

This script opens USD assets listed in an asset catalog, renders a preview
image for each, and writes the thumbnail to an output directory. Thumbnail
paths can optionally be merged back into the catalog or written to a separate
manifest mapping asset IDs to thumbnail files.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def _isaac_available() -> bool:
    """Return True if Isaac Sim's Python modules can be imported."""

    try:
        from omni.isaac.kit import SimulationApp  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class ViewportConfig:
    width: int = 512
    height: int = 512
    fov: float = 60.0
    distance: float = 2.5
    elevation: float = 25.0


@dataclass
class AssetRecord:
    asset_id: str
    usd_path: Path


class IsaacThumbnailRenderer:
    """Render thumbnails for USD assets using Isaac Sim headless."""

    def __init__(self, viewport: ViewportConfig):
        if not _isaac_available():
            raise RuntimeError("Isaac Sim modules not available in this environment")

        from omni.isaac.kit import SimulationApp

        self._app = SimulationApp(
            {
                "headless": True,
                "width": viewport.width,
                "height": viewport.height,
                "renderer": "RayTracedLighting",
            }
        )

        import omni.kit.viewport.utility
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        self._viewport_utility = omni.kit.viewport.utility
        self._usd_context = omni.usd
        self._Gf = Gf
        self._UsdGeom = UsdGeom
        self._UsdLux = UsdLux
        self._viewport = viewport

    def render(self, asset: AssetRecord, output_path: Path) -> None:
        """Render the given asset to the requested path."""

        self._usd_context.get_context().open_stage(str(asset.usd_path))
        stage = self._usd_context.get_context().get_stage()

        if not stage:
            raise RuntimeError(f"Failed to open stage for asset {asset.asset_id}: {asset.usd_path}")

        self._setup_environment(stage)
        self._position_camera(stage)
        self._app.update()

        viewport_handle = self._viewport_utility.get_active_viewport()
        viewport_handle.take_screenshot(str(output_path))

    def close(self) -> None:
        self._app.close()

    def _setup_environment(self, stage) -> None:
        dome = self._UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(300)

        ground = self._UsdGeom.Mesh.Define(stage, "/World/Ground")
        ground.CreatePointsAttr(
            [
                (-5.0, -5.0, 0.0),
                (5.0, -5.0, 0.0),
                (5.0, 5.0, 0.0),
                (-5.0, 5.0, 0.0),
            ]
        )
        ground.CreateFaceVertexCountsAttr([4])
        ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        ground.CreateNormalsAttr([(0.0, 0.0, 1.0)])

    def _position_camera(self, stage) -> None:
        camera_path = "/World/Camera"
        camera = self._UsdGeom.Camera.Define(stage, camera_path)
        horizontal_aperture = 20.955
        vertical_aperture = 15.2908
        focal_length = (horizontal_aperture / 2) / math.tan(math.radians(self._viewport.fov / 2))

        camera.CreateHorizontalApertureAttr(horizontal_aperture)
        camera.CreateVerticalApertureAttr(vertical_aperture)
        camera.CreateFocalLengthAttr(focal_length)

        # Position camera relative to asset origin
        distance = self._viewport.distance
        elevation = self._viewport.elevation
        x = 0.0
        y = -distance
        z = distance * (elevation / 90.0)

        xform = self._UsdGeom.XformCommonAPI(camera)
        xform.SetTranslate(self._Gf.Vec3d(x, y, z))
        xform.SetRotate(self._Gf.Vec3f(elevation, 0.0, 0.0))

        self._usd_context.get_context().set_default_camera(camera_path)


def load_catalog(asset_index: Path) -> dict:
    with open(asset_index, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_assets(catalog: dict, asset_root: Path, asset_subset: Optional[set[str]]) -> Iterable[AssetRecord]:
    assets = catalog.get("assets", [])
    for asset in assets:
        asset_id = asset.get("asset_id")
        if asset_subset and asset_id not in asset_subset:
            continue
        rel_path = asset.get("relative_path")
        if not rel_path:
            continue
        usd_path = asset_root / rel_path
        yield AssetRecord(asset_id=asset_id, usd_path=usd_path)


def update_catalog_with_thumbnails(catalog_path: Path, thumbnail_map: dict[str, str], output_path: Optional[Path]) -> None:
    catalog = load_catalog(catalog_path)
    for asset in catalog.get("assets", []):
        asset_id = asset.get("asset_id")
        if asset_id in thumbnail_map:
            asset["thumbnail_path"] = thumbnail_map[asset_id]

    target = output_path or catalog_path
    with open(target, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)


def write_manifest(manifest_path: Path, thumbnail_map: dict[str, str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(thumbnail_map, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thumbnails for USD assets using Isaac Sim")
    parser.add_argument("asset_index", type=Path, help="Path to the asset_index.json catalog")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Root directory containing USD assets; defaults to the catalog directory",
    )
    parser.add_argument(
        "--assets",
        type=str,
        nargs="*",
        help="Optional list of asset IDs to render; renders all if omitted",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("thumbnails"), help="Directory to write thumbnails")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to write thumbnail manifest JSON")
    parser.add_argument(
        "--update-catalog",
        action="store_true",
        help="Update the asset_index.json with thumbnail paths instead of writing a manifest",
    )
    parser.add_argument(
        "--viewport-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[512, 512],
        help="Viewport width and height for rendered images",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Camera field of view in degrees",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=2.5,
        help="Camera distance from the asset origin",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=25.0,
        help="Camera elevation angle in degrees",
    )
    return parser.parse_args()


def main() -> int:
    if not _isaac_available():
        print("Isaac Sim is not available in this environment; exiting without rendering", file=sys.stderr)
        return 1

    args = parse_args()

    asset_root = args.asset_root or args.asset_index.parent
    asset_ids = set(args.assets) if args.assets else None

    viewport = ViewportConfig(
        width=args.viewport_size[0],
        height=args.viewport_size[1],
        fov=args.fov,
        distance=args.distance,
        elevation=args.elevation,
    )

    catalog = load_catalog(args.asset_index)
    renderer = IsaacThumbnailRenderer(viewport)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    thumbnail_map: dict[str, str] = {}
    try:
        for asset in iter_assets(catalog, asset_root, asset_ids):
            output_path = output_dir / f"{asset.asset_id}.png"
            try:
                renderer.render(asset, output_path)
                thumbnail_map[asset.asset_id] = str(output_path.relative_to(output_dir))
                print(f"Rendered {asset.asset_id} -> {output_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to render {asset.asset_id}: {exc}", file=sys.stderr)
    finally:
        renderer.close()

    manifest_path = args.manifest or (output_dir / "manifest.json")

    if args.update_catalog:
        update_catalog_with_thumbnails(args.asset_index, thumbnail_map, None)
        print(f"Updated catalog with {len(thumbnail_map)} thumbnails: {args.asset_index}")
    else:
        write_manifest(manifest_path, thumbnail_map)
        print(f"Wrote thumbnail manifest to {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

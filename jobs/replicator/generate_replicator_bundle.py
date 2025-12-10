#!/usr/bin/env python3
"""
Replicator Bundle Generator

This script generates Replicator YAML configurations from a recipe.
It produces a portable bundle that can be run with Isaac Sim/Replicator.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_recipe(recipe_path: str) -> dict[str, Any]:
    """Load recipe from GCS or local path."""
    if recipe_path.startswith("gs://"):
        from google.cloud import storage
        client = storage.Client()

        parts = recipe_path[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_text()
        return json.loads(content)
    else:
        with open(recipe_path) as f:
            return json.load(f)


def load_policy_config() -> dict[str, Any]:
    """Load policy configuration."""
    # In production, this would be loaded from a config file
    # For now, use embedded defaults
    return {
        "policies": {
            "dexterous_pick_place": {
                "display_name": "Dexterous Pick & Place",
                "randomizers": [
                    {"name": "object_scatter", "enabled": True, "frequency": "per_frame"},
                    {"name": "material_variation", "enabled": True, "frequency": "per_frame"},
                    {"name": "lighting_variation", "enabled": True, "frequency": "per_episode"}
                ],
                "capture_config": {
                    "annotations": ["rgb", "depth", "semantic_segmentation", "instance_segmentation", "bounding_box_2d"],
                    "resolution": [1280, 720]
                }
            },
            "articulated_access": {
                "display_name": "Articulated Access",
                "randomizers": [
                    {"name": "articulation_state", "enabled": True, "frequency": "per_episode"},
                    {"name": "lighting_variation", "enabled": True, "frequency": "per_episode"}
                ],
                "capture_config": {
                    "annotations": ["rgb", "depth", "semantic_segmentation", "joint_states"],
                    "resolution": [1280, 720]
                }
            },
            "drawer_manipulation": {
                "display_name": "Drawer Manipulation",
                "randomizers": [
                    {"name": "drawer_state", "enabled": True, "frequency": "per_episode"},
                    {"name": "drawer_contents", "enabled": True, "frequency": "per_episode"},
                    {"name": "lighting_variation", "enabled": True, "frequency": "per_episode"}
                ],
                "capture_config": {
                    "annotations": ["rgb", "depth", "semantic_segmentation", "joint_states"],
                    "resolution": [1280, 720]
                }
            }
        }
    }


def generate_dataset_yaml(
    recipe: dict[str, Any],
    policy: dict[str, Any],
    num_frames: int
) -> dict[str, Any]:
    """Generate dataset.yaml configuration."""
    return {
        "version": "1.0",
        "metadata": {
            "name": f"{recipe['metadata'].get('environment_type', 'scene')}_{datetime.utcnow().strftime('%Y%m%d')}",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "generator": "BlueprintRecipe",
            "policy": policy.get("display_name", "unknown")
        },
        "scene": {
            "path": recipe["metadata"].get("scene_path", "scene.usda"),
            "up_axis": "Y",
            "meters_per_unit": 1.0
        },
        "generation": {
            "num_frames": num_frames,
            "seed": recipe["metadata"].get("deterministic_seed"),
            "headless": True,
            "use_gpu": True
        },
        "output": {
            "format": "replicator_default"
        }
    }


def generate_cameras_yaml(recipe: dict[str, Any]) -> dict[str, Any]:
    """Generate cameras.yaml configuration."""
    room = recipe.get("room", {})
    dims = room.get("dimensions", {})

    width = dims.get("width", 5.0)
    depth = dims.get("depth", 5.0)
    height = dims.get("height", 3.0)

    cameras = [
        {
            "name": "overview",
            "type": "perspective",
            "position": [width * 0.8, height * 0.7, depth * 0.8],
            "look_at": [0, height * 0.3, 0],
            "focal_length": 24.0,
            "resolution": [1280, 720]
        },
        {
            "name": "robot_eye",
            "type": "perspective",
            "position": [0, 1.2, -depth * 0.3],
            "look_at": [0, 0.9, depth * 0.2],
            "focal_length": 50.0,
            "resolution": [1280, 720]
        },
        {
            "name": "corner_1",
            "type": "perspective",
            "position": [-width * 0.4, height * 0.5, -depth * 0.4],
            "look_at": [0, height * 0.3, 0],
            "focal_length": 35.0,
            "resolution": [1280, 720]
        },
        {
            "name": "corner_2",
            "type": "perspective",
            "position": [width * 0.4, height * 0.5, depth * 0.4],
            "look_at": [0, height * 0.3, 0],
            "focal_length": 35.0,
            "resolution": [1280, 720]
        }
    ]

    return {
        "version": "1.0",
        "cameras": cameras,
        "render_products": [
            {"camera": cam["name"], "resolution": cam["resolution"]}
            for cam in cameras
        ]
    }


def generate_randomizers_yaml(policy: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    """Generate randomizations.yaml configuration."""
    randomizers = []

    for rand in policy.get("randomizers", []):
        if not rand.get("enabled", True):
            continue

        randomizers.append({
            "name": rand["name"],
            "type": rand["name"],
            "enabled": True,
            "frequency": rand.get("frequency", "per_frame"),
            "parameters": rand.get("parameters", {})
        })

    return {
        "version": "1.0",
        "randomizers": randomizers
    }


def generate_writers_yaml(policy: dict[str, Any]) -> dict[str, Any]:
    """Generate writers.yaml configuration."""
    capture_config = policy.get("capture_config", {})
    annotations = capture_config.get("annotations", ["rgb"])

    writers = []

    annotation_configs = {
        "rgb": {"type": "rgb", "file_format": "png"},
        "depth": {"type": "distance_to_camera", "file_format": "npy"},
        "semantic_segmentation": {"type": "semantic_segmentation", "file_format": "png"},
        "instance_segmentation": {"type": "instance_segmentation", "file_format": "png"},
        "bounding_box_2d": {"type": "bounding_box_2d_tight", "file_format": "json"},
        "bounding_box_3d": {"type": "bounding_box_3d", "file_format": "json"},
        "joint_states": {"type": "joint_states", "file_format": "json"},
        "object_pose": {"type": "object_pose", "file_format": "json"}
    }

    for ann in annotations:
        if ann in annotation_configs:
            config = annotation_configs[ann]
            writers.append({
                "name": ann,
                "type": config["type"],
                "enabled": True,
                "file_format": config["file_format"]
            })

    return {
        "version": "1.0",
        "writers": writers
    }


def generate_run_script(policy_name: str, num_frames: int) -> str:
    """Generate run_generation.py script."""
    return f'''#!/usr/bin/env python3
"""
Replicator Generation Script
Generated by BlueprintRecipe

Run with Isaac Sim:
    ./python.sh run_generation.py
"""

import omni.replicator.core as rep
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent

def load_config(name):
    with open(CONFIG_DIR / f"{{name}}.yaml") as f:
        return yaml.safe_load(f)

def main():
    # Load configurations
    dataset_cfg = load_config("dataset")
    cameras_cfg = load_config("cameras")
    randomizers_cfg = load_config("randomizations")
    writers_cfg = load_config("writers")

    # Open scene
    rep.open_stage(dataset_cfg["scene"]["path"])

    # Setup cameras
    cameras = []
    for cam_cfg in cameras_cfg["cameras"]:
        camera = rep.create.camera(
            position=cam_cfg["position"],
            look_at=cam_cfg["look_at"],
            focal_length=cam_cfg["focal_length"]
        )
        cameras.append(camera)

    # Setup writers
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir="./output",
        rgb=True,
        semantic_segmentation=True,
        instance_segmentation=True,
        distance_to_camera=True,
        bounding_box_2d_tight=True
    )

    # Attach to render products
    for camera in cameras:
        render_product = rep.create.render_product(camera, resolution=(1280, 720))
        writer.attach([render_product])

    # Run generation
    rep.orchestrator.run_until_complete(num_frames={num_frames})

    print("Generation complete: {num_frames} frames generated")

if __name__ == "__main__":
    main()
'''


def upload_bundle(bucket: str, output_prefix: str, files: dict[str, str]):
    """Upload bundle files to GCS."""
    from google.cloud import storage
    client = storage.Client()
    gcs_bucket = client.bucket(bucket)

    for filename, content in files.items():
        blob = gcs_bucket.blob(f"{output_prefix}/{filename}")
        blob.upload_from_string(content, content_type="text/plain")
        print(f"[REPLICATOR] Uploaded {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate Replicator Bundle")
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--bucket", required=True, help="GCS bucket")
    parser.add_argument("--recipe-path", required=True, help="Path to recipe.json")
    parser.add_argument("--policy-id", default="dexterous_pick_place", help="Policy ID")
    parser.add_argument("--num-frames", type=int, default=1000, help="Number of frames")
    parser.add_argument("--output-prefix", required=True, help="Output prefix")
    args = parser.parse_args()

    print(f"[REPLICATOR] Processing job {args.job_id}")

    # Load recipe
    print(f"[REPLICATOR] Loading recipe from {args.recipe_path}")
    recipe = load_recipe(args.recipe_path)

    # Load policy config
    policy_config = load_policy_config()
    policy = policy_config["policies"].get(args.policy_id, {})

    if not policy:
        print(f"[REPLICATOR] Warning: Unknown policy {args.policy_id}, using defaults")
        policy = policy_config["policies"]["dexterous_pick_place"]

    # Generate configurations
    print("[REPLICATOR] Generating configurations...")

    files = {
        "dataset.yaml": yaml.dump(
            generate_dataset_yaml(recipe, policy, args.num_frames),
            default_flow_style=False
        ),
        "cameras.yaml": yaml.dump(
            generate_cameras_yaml(recipe),
            default_flow_style=False
        ),
        "randomizations.yaml": yaml.dump(
            generate_randomizers_yaml(policy, recipe),
            default_flow_style=False
        ),
        "writers.yaml": yaml.dump(
            generate_writers_yaml(policy),
            default_flow_style=False
        ),
        "run_generation.py": generate_run_script(args.policy_id, args.num_frames)
    }

    # Upload bundle
    print("[REPLICATOR] Uploading bundle...")
    upload_bundle(args.bucket, args.output_prefix, files)

    print("[REPLICATOR] Bundle generation complete")


if __name__ == "__main__":
    main()

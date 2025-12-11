"""Cloud Run job that delegates Replicator bundle creation to Blueprint Sim."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from src.sim_integration.blueprint_sim import BlueprintSimClient


def load_manifest(path: str) -> dict[str, Any]:
    """Load a canonical scene manifest from GCS or local storage."""

    if path.startswith("gs://"):
        from google.cloud import storage

        client = storage.Client()
        bucket_name, blob_name = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return json.loads(blob.download_as_text())

    with open(path, "r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def upload_file(local_path: Path, bucket: str, dest_prefix: str) -> str:
    """Upload a file to GCS and return the destination URI."""

    from google.cloud import storage

    client = storage.Client()
    blob_path = f"{dest_prefix}/{local_path.name}"
    bucket_ref = client.bucket(bucket)
    blob = bucket_ref.blob(blob_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket}/{blob_path}"


def main():
    parser = argparse.ArgumentParser(description="Delegate Replicator bundle generation")
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--bucket", required=True, help="GCS bucket")
    parser.add_argument("--manifest-path", required=True, help="Path to scene manifest")
    parser.add_argument("--policy-id", default="dexterous_pick_place", help="Policy ID")
    parser.add_argument("--output-prefix", required=True, help="Output prefix")
    args = parser.parse_args()

    print(f"[REPLICATOR] Processing job {args.job_id}")

    manifest = load_manifest(args.manifest_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        client = BlueprintSimClient()
        result = client.generate_from_manifest(
            manifest,
            output_dir=tmp_path,
            policies=[args.policy_id],
            metadata={"job_id": args.job_id},
        )

        bundle_path = result.replicator_bundle
        if not bundle_path:
            raise RuntimeError("Blueprint Sim did not return a Replicator bundle path")

        resolved_path = Path(bundle_path)
        if not resolved_path.is_absolute():
            resolved_path = tmp_path / resolved_path

        if not resolved_path.exists():
            raise FileNotFoundError(f"Replicator bundle missing at {resolved_path}")

        uploaded_uri = upload_file(resolved_path, args.bucket, args.output_prefix)
        print(f"[REPLICATOR] Uploaded bundle to {uploaded_uri}")


if __name__ == "__main__":
    main()

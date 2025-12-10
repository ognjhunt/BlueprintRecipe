# Replicator Job

Generates NVIDIA Isaac Sim Replicator bundles for synthetic data generation and domain randomization.

## Overview

The `replicator-job` analyzes completed scenes and generates ready-to-run Replicator configurations for training robotics policies. It uses Gemini to intelligently create:

- **Placement Regions**: USD layers defining where objects can be spawned (counters, shelves, drawers, etc.)
- **Variation Assets**: Manifests of additional assets needed for domain randomization
- **Policy-Specific Scripts**: Python scripts for different training scenarios
- **Randomizer Configurations**: Parameters for object scattering, material variation, lighting, etc.

## Pipeline Position

```
Image Upload → seg-job → multiview-job → sam3d-job → hunyuan-job
                                                          ↓
         usd-assembly-job (CONVERT) → simready-job → usd-assembly-job (FULL)
                                                          ↓
                                               replicator-job ← YOU ARE HERE
                                                          ↓
                                               scenes/<id>/replicator/
```

## Supported Environments

| Environment | Description | Default Policies |
|------------|-------------|------------------|
| **Kitchen** | Commercial prep lines, dish pits, quick-serve stations | dish_loading, table_clearing, articulated_access |
| **Grocery** | Planogrammed aisles, refrigeration units | grocery_stocking, mixed_sku_logistics |
| **Warehouse** | Racked corridors, pallets, totes | mixed_sku_logistics, dexterous_pick_place |
| **Loading Dock** | Bays, liftgates, staging areas | mixed_sku_logistics |
| **Lab** | Wet benches, fume hoods, equipment | precision_insertion, dexterous_pick_place |
| **Office** | Desks, filing cabinets, service nooks | drawer_manipulation, panel_interaction |
| **Utility Room** | Panels, valves, service equipment | panel_interaction, knob_manipulation |
| **Home Laundry** | Washer/dryer, hampers, folding tables | laundry_sorting, door_manipulation |
| **Bedroom** | Dressers, closets, nightstands | drawer_manipulation, laundry_sorting |

## Supported Policies

| Policy | Description |
|--------|-------------|
| `dexterous_pick_place` | General pick & place with various objects |
| `articulated_access` | Opening doors, drawers, mechanisms |
| `panel_interaction` | Switches, buttons, control panels |
| `mixed_sku_logistics` | Package handling in logistics |
| `precision_insertion` | Fine manipulation tasks |
| `laundry_sorting` | Cloth handling and sorting |
| `dish_loading` | Loading dishes into dishwashers |
| `grocery_stocking` | Restocking shelves |
| `table_clearing` | Clearing table surfaces |
| `drawer_manipulation` | Opening/closing drawers |
| `door_manipulation` | Opening/closing doors |
| `knob_manipulation` | Rotating knobs and dials |
| `general_manipulation` | Generic object manipulation |

## Output Structure

```
scenes/<scene_id>/replicator/
├── replicator_master.py          # Main entry point
├── placement_regions.usda        # USD layer with placement surfaces
├── bundle_metadata.json          # Scene and policy information
├── README.md                     # Usage instructions
├── policies/                     # Policy-specific scripts
│   ├── dish_loading.py
│   ├── table_clearing.py
│   └── ...
├── configs/                      # Policy configurations
│   ├── dish_loading.json
│   └── ...
└── variation_assets/             # Assets for domain randomization
    ├── manifest.json             # Asset requirements
    └── *.usdz                    # Asset files (to be added)
```

## Usage in Isaac Sim

### Basic Usage

1. Open `scene.usda` in Isaac Sim
2. Load the placement regions layer
3. In the Script Editor:

```python
from replicator_master import ReplicatorManager

manager = ReplicatorManager()
manager.list_policies()  # See available policies
manager.run_policy("dish_loading", num_frames=500)
```

### Running Specific Policies

```python
# Run dish loading policy
manager.run_policy("dish_loading", num_frames=1000)

# Run all policies sequentially
manager.run_all_policies(num_frames_each=200)
```

### Custom Configuration

Edit `configs/<policy_id>.json` to customize:
- Number of objects to spawn
- Randomization parameters
- Capture resolution
- Annotations to generate

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BUCKET` | Yes | GCS bucket name |
| `SCENE_ID` | Yes | Scene identifier |
| `SEG_PREFIX` | No | Path to segmentation data (default: `scenes/<id>/seg`) |
| `ASSETS_PREFIX` | No | Path to assets (default: `scenes/<id>/assets`) |
| `USD_PREFIX` | No | Path to USD files (default: `scenes/<id>/usd`) |
| `REPLICATOR_PREFIX` | No | Output path (default: `scenes/<id>/replicator`) |
| `GEMINI_API_KEY` | Yes | Gemini API key for scene analysis |
| `REQUESTED_POLICIES` | No | Comma-separated list of specific policies to generate |

## Deployment

### Build Docker Image

```bash
cd replicator-job

docker build -t gcr.io/${PROJECT_ID}/replicator-job:latest .
```

### Create Cloud Run Job

```bash
gcloud run jobs create replicator-job \
  --image=gcr.io/${PROJECT_ID}/replicator-job:latest \
  --region=us-central1 \
  --memory=2Gi \
  --cpu=2 \
  --max-retries=1 \
  --task-timeout=15m \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY}" \
  --execution-environment=gen2
```

### Deploy Updated Workflow

```bash
gcloud workflows deploy usd-assembly-pipeline \
  --source=workflows/usd-assembly-pipeline.yaml \
  --location=us-central1
```

## Performance Tips

1. **Batch frames**: Generate 100-500 frames per episode
2. **Parallel rendering**: Use multiple render products if GPU allows
3. **Selective annotations**: Only enable needed annotations
4. **Asset LOD**: Use appropriate level of detail for variation assets

## License

Part of BlueprintRecipe. See main repository for license.

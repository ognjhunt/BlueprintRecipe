# BlueprintRecipe

Scene recipe generation system for Isaac Sim/Lab training using NVIDIA asset packs.

## Overview

BlueprintRecipe transforms images into sim-ready USD scenes with physics, semantics, and training configurations. Given a photo or sketch, the system:

1. **Analyzes** the image using Gemini to generate a structured scene plan
2. **Matches** objects to NVIDIA asset packs (Residential, Warehouse, etc.)
3. **Compiles** a layered USD scene with physics and semantic annotations
4. **Generates** Replicator configs for synthetic data generation
5. **Creates** Isaac Lab task packages for policy training

## Architecture

```
BlueprintRecipe/
├── schemas/                    # JSON schemas for recipes, plans, assets
├── policy_configs/             # Environment and policy configurations
├── src/
│   ├── recipe_compiler/        # USD scene compilation
│   ├── asset_catalog/          # Asset indexing and matching
│   ├── replicator_generator/   # Replicator YAML generation
│   ├── isaac_lab_tasks/        # Isaac Lab task generation
│   └── planning/               # Gemini-based scene planning
├── api/                        # FastAPI service
├── jobs/                       # Cloud Run job definitions
│   ├── simready/               # Asset preparation job
│   ├── replicator/             # SDG bundle generation
│   └── qa_validation/          # Scene validation
└── examples/                   # Example recipe packs
```

## Recipe Pack Output

A recipe pack is a portable folder that can be "applied" to a customer's asset library:

```
kitchen_recipe_pack/
├── recipe.json                 # Machine-readable recipe
├── scene.usda                  # Top-level USD stage
├── layers/
│   ├── room_shell.usda         # Room geometry
│   ├── layout.usda             # Object placements
│   ├── semantics.usda          # Semantic labels
│   └── physics_overrides.usda  # Physics properties
├── replicator/
│   ├── dataset.yaml            # Generation config
│   ├── cameras.yaml            # Camera setups
│   ├── randomizations.yaml     # Domain randomization
│   └── writers.yaml            # Annotation writers
├── isaac_lab/
│   ├── env_cfg.py              # Environment config
│   ├── task_*.py               # Task implementations
│   └── train_cfg.yaml          # Training config
└── qa/
    └── compilation_report.json # Validation results
```

## Supported Environments

| Environment | Asset Pack | Policies |
|-------------|-----------|----------|
| Kitchen | Residential | pick_place, drawer_manipulation, articulated_access |
| Living Room | Residential | pick_place, general_manipulation |
| Bedroom | Residential | drawer_manipulation, laundry_sorting |
| Warehouse | Warehouse, SimReady | mixed_sku_logistics, pallet_handling |
| Grocery | Commercial | grocery_stocking, pick_place |
| Lab | Commercial | precision_insertion, articulated_access |

## Installation

```bash
# Clone the repository
git clone https://github.com/ognjhunt/BlueprintRecipe.git
cd BlueprintRecipe

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GOOGLE_API_KEY="your-gemini-api-key"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

## Quick Start

### 1. Build Asset Catalog

First, index your NVIDIA asset pack:

```python
from src.asset_catalog import AssetCatalogBuilder

builder = AssetCatalogBuilder(
    pack_path="/path/to/ResidentialAssetsPack",
    pack_name="ResidentialAssetsPack"
)
catalog = builder.build()
builder.save(catalog, "asset_index.json")
```

### 2. Generate Scene Plan from Image

```python
from src.planning import ScenePlanner

planner = ScenePlanner()
result = planner.plan_from_image(
    image_path="kitchen_photo.jpg",
    task_intent="robot picking mugs from counter",
    target_policies=["dexterous_pick_place"]
)

if result.success:
    scene_plan = result.scene_plan
```

### 3. Match Assets

```python
from src.asset_catalog import AssetCatalogBuilder, AssetMatcher

catalog = AssetCatalogBuilder.load("asset_index.json")
matcher = AssetMatcher(catalog)

results = matcher.match_batch(scene_plan["object_inventory"])
matched_assets = matcher.to_matched_assets(results)
```

### 4. Compile Recipe

```python
from src.recipe_compiler import RecipeCompiler, CompilerConfig

config = CompilerConfig(
    output_dir="./output/kitchen_recipe",
    asset_root="/path/to/assets"
)

compiler = RecipeCompiler(config)
result = compiler.compile(scene_plan, matched_assets)

print(f"Recipe saved to: {result.recipe_path}")
print(f"Scene saved to: {result.scene_path}")
```

### 5. Generate Replicator Config

```python
from src.replicator_generator import ReplicatorGenerator
import json

with open("policy_configs/environment_policies.json") as f:
    policy_config = json.load(f)

generator = ReplicatorGenerator(policy_config)

with open(result.recipe_path) as f:
    recipe = json.load(f)

rep_config = generator.generate(
    recipe=recipe,
    policy_id="dexterous_pick_place",
    num_frames=1000
)

generator.save(rep_config, "./output/kitchen_recipe/replicator")
```

### 6. Generate Isaac Lab Task

```python
from src.isaac_lab_tasks import IsaacLabTaskGenerator

task_generator = IsaacLabTaskGenerator(policy_config)

task = task_generator.generate(
    recipe=recipe,
    policy_id="dexterous_pick_place",
    robot_type="franka"
)

task_generator.save(task, "./output/kitchen_recipe/isaac_lab")
```

## API Service

Run the API locally:

```bash
cd api
uvicorn main:app --reload --port 8080
```

### Endpoints

- `POST /jobs` - Create a new recipe generation job
- `GET /jobs/{job_id}` - Get job status
- `POST /jobs/{job_id}/upload-image` - Upload source image
- `POST /jobs/{job_id}/approve` - Approve asset selections
- `GET /jobs/{job_id}/recipe` - Get generated recipe
- `POST /assets/search` - Search asset catalog

## Cloud Run Jobs

### SimReady Job

Prepares assets and validates the recipe:

```bash
gcloud run jobs create simready-job \
    --image gcr.io/PROJECT/simready-job \
    --set-env-vars JOB_ID=xxx,BUCKET=xxx,RECIPE_PATH=xxx
```

### Replicator Job

Generates Replicator YAML bundle:

```bash
gcloud run jobs create replicator-job \
    --image gcr.io/PROJECT/replicator-job \
    --set-env-vars JOB_ID=xxx,BUCKET=xxx,RECIPE_PATH=xxx,POLICY_ID=xxx
```

### QA Validation Job

Validates scene for simulation:

```bash
gcloud run jobs create qa-job \
    --image gcr.io/PROJECT/qa-job \
    --set-env-vars JOB_ID=xxx,BUCKET=xxx,SCENE_PATH=xxx
```

## Licensing Notes

NVIDIA asset packs have specific licensing requirements:

1. **Do not redistribute** NVIDIA Content on a standalone basis
2. **Reference customer's copy** of the pack (local path or Nucleus)
3. **Hosted services** may require separate NVIDIA agreement

This system outputs *recipes* that reference assets, not the assets themselves.

## Storage

The system uses Firebase Storage for scene data:

```
gs://blueprint-8c1ca.appspot.com/scenes/
├── {scene_id}/
│   ├── input/
│   │   └── source_image.jpg
│   ├── recipe/
│   │   └── recipe.json
│   ├── replicator/
│   │   └── *.yaml
│   └── isaac_lab/
│       └── *.py
```

## Development

### Running Tests

```bash
pytest tests/
```

### Building Docker Images

```bash
# API
docker build -t blueprint-recipe-api -f api/Dockerfile .

# Jobs
docker build -t simready-job -f jobs/simready/Dockerfile jobs/simready/
docker build -t replicator-job -f jobs/replicator/Dockerfile jobs/replicator/
docker build -t qa-job -f jobs/qa_validation/Dockerfile jobs/qa_validation/
```

## Related Projects

- [BlueprintPipeline](https://github.com/ognjhunt/BlueprintPipeline) - Original pipeline with simready-job and replicator-job
- [NVIDIA Omniverse USD](https://docs.omniverse.nvidia.com/usd/) - USD documentation
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) - Robot learning framework

## License

MIT License - See LICENSE file for details.

Note: NVIDIA asset packs are subject to NVIDIA's licensing terms.

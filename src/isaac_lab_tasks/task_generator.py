"""
Isaac Lab Task Generator - Creates Isaac Lab task packages from recipes.

This module generates complete Isaac Lab task packages including:
- Environment configuration
- Task definitions (observations, actions, rewards)
- Domain randomization hooks (EventManager compatible)
- Training configuration

Note: Generated code is designed to work with Isaac Lab's ManagerBasedEnv
architecture and its event-driven randomization system.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .env_config import EnvConfigGenerator
from .reward_functions import RewardFunctionGenerator


@dataclass
class TaskConfig:
    """Configuration for a generated Isaac Lab task."""
    task_name: str
    policy_id: str
    scene_path: str
    robot_type: str = "franka"  # franka, ur10, fetch, etc.
    num_envs: int = 1024
    episode_length: int = 500
    observation_space: dict[str, Any] = field(default_factory=dict)
    action_space: dict[str, Any] = field(default_factory=dict)
    reward_weights: dict[str, float] = field(default_factory=dict)
    randomization_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedTask:
    """Result of task generation."""
    task_name: str
    files: dict[str, str]  # filename -> content
    config: TaskConfig


class IsaacLabTaskGenerator:
    """
    Generates Isaac Lab task packages from scene recipes.

    Usage:
        generator = IsaacLabTaskGenerator(policy_config)
        task = generator.generate(recipe, policy_id="drawer_manipulation")
        generator.save(task, "isaac_lab/")
    """

    ROBOT_CONFIGS = {
        "franka": {
            "num_dofs": 7,
            "gripper_dofs": 2,
            "ee_frame": "panda_hand",
            "default_joint_pos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        },
        "ur10": {
            "num_dofs": 6,
            "gripper_dofs": 0,
            "ee_frame": "tool0",
            "default_joint_pos": [0.0, -1.571, 1.571, -1.571, -1.571, 0.0]
        },
        "fetch": {
            "num_dofs": 7,
            "gripper_dofs": 2,
            "ee_frame": "gripper_link",
            "default_joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }

    def __init__(self, policy_config: dict[str, Any]):
        self.policy_config = policy_config
        self.policies = policy_config.get("policies", {})
        self.environments = policy_config.get("environments", {})
        self.env_generator = EnvConfigGenerator()
        self.reward_generator = RewardFunctionGenerator()

    def generate(
        self,
        recipe: dict[str, Any],
        policy_id: str,
        robot_type: str = "franka",
        num_envs: int = 1024
    ) -> GeneratedTask:
        """
        Generate an Isaac Lab task package.

        Args:
            recipe: Scene recipe
            policy_id: ID of the training policy
            robot_type: Type of robot to use
            num_envs: Number of parallel environments

        Returns:
            GeneratedTask with all generated files
        """
        env_type = recipe.get("metadata", {}).get("environment_type", "unknown")
        scene_path = recipe.get("metadata", {}).get("scene_path", "scene.usda")
        policy = self.policies.get(policy_id, {})

        # Build task configuration
        task_config = self._build_task_config(
            recipe, policy, policy_id, robot_type, num_envs, scene_path
        )

        # Generate files
        files = {}

        # Environment config
        files["env_cfg.py"] = self._generate_env_config(task_config, recipe, policy)

        # Task implementation
        files[f"task_{policy_id}.py"] = self._generate_task_file(
            task_config, recipe, policy
        )

        # Training config
        files["train_cfg.yaml"] = self._generate_train_config(task_config, policy)

        # Randomization hooks
        files["randomizations.py"] = self._generate_randomization_hooks(
            task_config, recipe, policy
        )

        # Package init
        files["__init__.py"] = self._generate_init_file(task_config, policy_id)

        return GeneratedTask(
            task_name=f"{env_type}_{policy_id}",
            files=files,
            config=task_config
        )

    def _build_task_config(
        self,
        recipe: dict[str, Any],
        policy: dict[str, Any],
        policy_id: str,
        robot_type: str,
        num_envs: int,
        scene_path: str
    ) -> TaskConfig:
        """Build task configuration from recipe and policy."""
        robot_config = self.ROBOT_CONFIGS.get(robot_type, self.ROBOT_CONFIGS["franka"])

        # Build observation space based on policy
        obs_space = self._build_observation_space(policy, robot_config, recipe)

        # Build action space based on robot
        action_space = self._build_action_space(robot_config)

        # Get reward weights
        reward_components = policy.get("reward_components", [])
        reward_weights = {comp: 1.0 for comp in reward_components}

        # Get randomization config
        randomization_config = self._build_randomization_config(policy, recipe)

        return TaskConfig(
            task_name=f"{recipe['metadata']['environment_type']}_{policy_id}",
            policy_id=policy_id,
            scene_path=scene_path,
            robot_type=robot_type,
            num_envs=num_envs,
            episode_length=500,
            observation_space=obs_space,
            action_space=action_space,
            reward_weights=reward_weights,
            randomization_config=randomization_config
        )

    def _build_observation_space(
        self,
        policy: dict[str, Any],
        robot_config: dict[str, Any],
        recipe: dict[str, Any]
    ) -> dict[str, Any]:
        """Build observation space configuration."""
        obs_space = {
            "robot": {
                "joint_pos": robot_config["num_dofs"],
                "joint_vel": robot_config["num_dofs"],
                "ee_pos": 3,
                "ee_quat": 4,
                "gripper_pos": robot_config["gripper_dofs"]
            },
            "task": {}
        }

        # Add task-specific observations based on policy
        policy_id = policy.get("isaac_lab_task", "")

        if "pick_place" in policy_id or "manipulation" in policy_id:
            obs_space["task"]["target_pos"] = 3
            obs_space["task"]["object_pos"] = 3
            obs_space["task"]["object_quat"] = 4

        if "drawer" in policy_id or "door" in policy_id:
            obs_space["task"]["joint_pos"] = 1
            obs_space["task"]["joint_vel"] = 1
            obs_space["task"]["handle_pos"] = 3

        if "articulated" in policy_id:
            obs_space["task"]["articulation_pos"] = 1
            obs_space["task"]["articulation_vel"] = 1

        return obs_space

    def _build_action_space(self, robot_config: dict[str, Any]) -> dict[str, Any]:
        """Build action space configuration."""
        return {
            "type": "continuous",
            "arm_dofs": robot_config["num_dofs"],
            "gripper_dofs": robot_config["gripper_dofs"],
            "control_type": "joint_velocity",  # or "joint_position", "ee_pose"
            "action_scale": 0.1
        }

    def _build_randomization_config(
        self,
        policy: dict[str, Any],
        recipe: dict[str, Any]
    ) -> dict[str, Any]:
        """Build domain randomization configuration."""
        config = {
            "on_reset": [],
            "on_step": []
        }

        for rand in policy.get("randomizers", []):
            if not rand.get("enabled", True):
                continue

            freq = rand.get("frequency", "per_episode")
            event = {
                "name": rand["name"],
                "params": rand.get("parameters", {})
            }

            if freq in ["per_episode", "once"]:
                config["on_reset"].append(event)
            else:
                config["on_step"].append(event)

        return config

    def _generate_env_config(
        self,
        config: TaskConfig,
        recipe: dict[str, Any],
        policy: dict[str, Any]
    ) -> str:
        """Generate environment configuration file."""
        return f'''"""
Environment Configuration - {config.task_name}
Generated by BlueprintRecipe

This file defines the environment configuration for Isaac Lab.
"""

from dataclasses import MISSING
from typing import Literal

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from omni.isaac.lab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
from omni.isaac.lab.utils import configclass


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""

    # Ground plane (from recipe room)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Scene USD (from recipe)
    scene = AssetBaseCfg(
        prim_path="/World/Scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path="{config.scene_path}",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    # Robot configuration
    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="{{ROBOT_USD_PATH}}",  # To be resolved at runtime
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={self._format_joint_pos(config.robot_type)},
        ),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,
            color=(1.0, 1.0, 1.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation configuration."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for the policy."""

        # Robot state observations
        joint_pos = ObservationTermCfg(
            func="omni.isaac.lab.envs.mdp.joint_pos",
            params={{"asset_cfg": SceneEntityCfg("robot")}},
        )
        joint_vel = ObservationTermCfg(
            func="omni.isaac.lab.envs.mdp.joint_vel",
            params={{"asset_cfg": SceneEntityCfg("robot")}},
        )
        ee_pos = ObservationTermCfg(
            func="omni.isaac.lab.envs.mdp.body_pos_w",
            params={{
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "{self.ROBOT_CONFIGS[config.robot_type]['ee_frame']}"
            }},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action configuration."""

    # Joint velocity control
    joint_vel = {{
        "class_type": "omni.isaac.lab.envs.mdp.JointVelocityActionCfg",
        "asset_name": "robot",
        "joint_names": [".*"],
        "scale": {config.action_space.get("action_scale", 0.1)},
    }}


@configclass
class RewardsCfg:
    """Reward configuration."""

{self._generate_reward_terms(config, policy)}


@configclass
class TerminationsCfg:
    """Termination configuration."""

    time_out = TerminationTermCfg(
        func="omni.isaac.lab.envs.mdp.time_out",
        time_out=True,
    )

    # Task-specific terminations can be added here


@configclass
class EventsCfg:
    """Event configuration for domain randomization."""

{self._generate_event_terms(config, recipe, policy)}


@configclass
class {self._to_class_name(config.task_name)}EnvCfg(ManagerBasedEnvCfg):
    """Configuration for {config.task_name} environment."""

    # Scene
    scene: SceneCfg = SceneCfg(num_envs={config.num_envs}, env_spacing=2.0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Episode settings
    episode_length_s = {config.episode_length / 60.0:.1f}  # seconds

    def __post_init__(self):
        """Post initialization."""
        # Simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = 1

        # Update scene settings
        self.scene.num_envs = {config.num_envs}
'''

    def _generate_task_file(
        self,
        config: TaskConfig,
        recipe: dict[str, Any],
        policy: dict[str, Any]
    ) -> str:
        """Generate task implementation file."""
        class_name = self._to_class_name(config.task_name)
        return f'''"""
Task Implementation - {config.task_name}
Generated by BlueprintRecipe

This file implements the task logic for {policy.get("display_name", config.policy_id)}.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from .env_cfg import {class_name}EnvCfg


class {class_name}Task:
    """
    Task implementation for {config.task_name}.

    Policy: {policy.get("display_name", config.policy_id)}
    Description: {policy.get("description", "")}
    """

    def __init__(self, env: ManagerBasedEnv, cfg: {class_name}EnvCfg):
        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.num_envs = env.num_envs

        # Task state
        self._setup_task_state()

    def _setup_task_state(self):
        """Initialize task-specific state tensors."""
        # Target positions (for pick-place tasks)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Task progress tracking
        self.task_progress = torch.zeros(self.num_envs, device=self.device)

        # Success flags
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        """Reset task state for specified environments."""
        num_resets = len(env_ids)

        # Reset target positions (example: random positions in workspace)
        self.target_pos[env_ids] = torch.rand(num_resets, 3, device=self.device) * 0.4 - 0.2
        self.target_pos[env_ids, 2] += 0.1  # Above table

        # Reset progress
        self.task_progress[env_ids] = 0.0
        self.success[env_ids] = False

    def compute_observations(self) -> dict[str, torch.Tensor]:
        """Compute task-specific observations."""
        obs = {{}}

        # Target position relative to end-effector
        ee_pos = self._get_ee_pos()
        obs["target_rel_pos"] = self.target_pos - ee_pos

        return obs

    def compute_rewards(self) -> dict[str, torch.Tensor]:
        """Compute reward components."""
        rewards = {{}}

        ee_pos = self._get_ee_pos()
        dist_to_target = torch.norm(self.target_pos - ee_pos, dim=-1)

        # Distance reward (shaped)
        rewards["distance"] = 1.0 - torch.tanh(5.0 * dist_to_target)

        # Success bonus
        success_threshold = 0.05
        reached = dist_to_target < success_threshold
        rewards["success"] = reached.float() * 10.0

        # Update success tracking
        self.success = self.success | reached

        return rewards

    def compute_terminations(self) -> dict[str, torch.Tensor]:
        """Compute termination conditions."""
        terminations = {{}}

        # Success termination
        terminations["success"] = self.success

        return terminations

    def _get_ee_pos(self) -> torch.Tensor:
        """Get end-effector position."""
        robot = self.env.scene["robot"]
        ee_frame = "{self.ROBOT_CONFIGS[config.robot_type]['ee_frame']}"
        return robot.data.body_pos_w[:, robot.find_bodies(ee_frame)[0]]


# Reward function implementations for manager-based architecture
def reward_distance_to_target(
    env: ManagerBasedEnv,
    target_pos: torch.Tensor,
) -> torch.Tensor:
    """Reward for distance to target."""
    robot = env.scene["robot"]
    ee_pos = robot.data.body_pos_w[:, robot.find_bodies("{self.ROBOT_CONFIGS[config.robot_type]['ee_frame']}")[0]]
    dist = torch.norm(target_pos - ee_pos, dim=-1)
    return 1.0 - torch.tanh(5.0 * dist)


def reward_task_success(
    env: ManagerBasedEnv,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Reward for task success."""
    # Implementation depends on specific task
    return torch.zeros(env.num_envs, device=env.device)


def reward_collision_penalty(
    env: ManagerBasedEnv,
    penalty: float = -1.0,
) -> torch.Tensor:
    """Penalty for collisions."""
    # Check contact forces
    robot = env.scene["robot"]
    contact_forces = robot.data.net_contact_forces_w
    has_collision = torch.any(torch.abs(contact_forces) > 50.0, dim=-1)
    return has_collision.float() * penalty


def reward_smooth_motion(
    env: ManagerBasedEnv,
    penalty_scale: float = 0.01,
) -> torch.Tensor:
    """Reward for smooth motion (penalize large accelerations)."""
    robot = env.scene["robot"]
    joint_acc = robot.data.joint_acc
    return -penalty_scale * torch.sum(joint_acc ** 2, dim=-1)
'''

    def _generate_train_config(
        self,
        config: TaskConfig,
        policy: dict[str, Any]
    ) -> str:
        """Generate training configuration YAML."""
        return f'''# Training Configuration - {config.task_name}
# Generated by BlueprintRecipe

# Task settings
task_name: "{config.task_name}"
experiment_name: "{config.task_name}_training"

# Environment settings
env:
  num_envs: {config.num_envs}
  episode_length: {config.episode_length}

# PPO Algorithm settings
algo:
  name: "PPO"
  policy:
    class_name: "ActorCritic"
    init_noise_std: 1.0
    actor_hidden_dims: [256, 256, 128]
    critic_hidden_dims: [256, 256, 128]
    activation: "elu"

  # PPO specific
  clip_param: 0.2
  entropy_coef: 0.01
  value_loss_coef: 1.0
  max_grad_norm: 1.0

  # Learning rate
  learning_rate: 3.0e-4
  lr_schedule: "adaptive"

  # Batch settings
  num_learning_epochs: 5
  num_mini_batches: 4

  # Discount
  gamma: 0.99
  lam: 0.95

# Training settings
runner:
  max_iterations: 1500
  save_interval: 100
  log_interval: 10

  # Checkpointing
  checkpoint_path: null
  resume: false

# Logging
logging:
  wandb:
    enabled: false
    project: "blueprint_recipe"
    entity: null
  tensorboard:
    enabled: true

# Reward weights
rewards:
{self._format_reward_weights_yaml(config.reward_weights)}

# Domain randomization
randomization:
  enabled: true
  on_reset:
{self._format_randomization_yaml(config.randomization_config.get("on_reset", []))}
  on_step:
{self._format_randomization_yaml(config.randomization_config.get("on_step", []))}
'''

    def _generate_randomization_hooks(
        self,
        config: TaskConfig,
        recipe: dict[str, Any],
        policy: dict[str, Any]
    ) -> str:
        """Generate domain randomization hooks file."""
        return f'''"""
Domain Randomization Hooks - {config.task_name}
Generated by BlueprintRecipe

These hooks are designed to work with Isaac Lab's EventManager.
They write directly to PhysX for RL speed mode compatibility.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import EventTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnvCfg


def randomize_object_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float, float, float, float, float] = (-0.2, 0.2, -0.2, 0.2, 0.0, 0.1),
    rotation_range: tuple[float, float] = (0.0, 6.28),
):
    """
    Randomize object poses on reset.

    Note: This writes directly to PhysX buffers for RL speed mode compatibility.
    """
    num_envs = len(env_ids)

    # Get objects to randomize
    objects = env.scene.get("objects")
    if objects is None:
        return

    # Generate random positions
    pos = torch.zeros(num_envs, 3, device=env.device)
    pos[:, 0] = torch.rand(num_envs, device=env.device) * (position_range[1] - position_range[0]) + position_range[0]
    pos[:, 1] = torch.rand(num_envs, device=env.device) * (position_range[3] - position_range[2]) + position_range[2]
    pos[:, 2] = torch.rand(num_envs, device=env.device) * (position_range[5] - position_range[4]) + position_range[4]

    # Generate random rotations (around Z axis)
    yaw = torch.rand(num_envs, device=env.device) * (rotation_range[1] - rotation_range[0]) + rotation_range[0]
    quat = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        yaw
    )

    # Write to physics
    objects.write_root_pose_to_sim(
        torch.cat([pos, quat], dim=-1),
        env_ids
    )


def randomize_articulation_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    articulation_name: str = "articulated_object",
    joint_range: tuple[float, float] = (0.0, 1.0),
):
    """
    Randomize articulation joint states on reset.
    """
    num_envs = len(env_ids)

    articulation = env.scene.get(articulation_name)
    if articulation is None:
        return

    # Generate random joint positions
    num_joints = articulation.num_joints
    joint_pos = torch.rand(num_envs, num_joints, device=env.device)
    joint_pos = joint_pos * (joint_range[1] - joint_range[0]) + joint_range[0]

    # Respect joint limits
    joint_limits = articulation.data.joint_limits
    joint_pos = torch.clamp(
        joint_pos,
        joint_limits[..., 0],
        joint_limits[..., 1]
    )

    # Write to physics
    articulation.write_joint_state_to_sim(
        joint_pos,
        torch.zeros_like(joint_pos),  # zero velocity
        env_ids=env_ids
    )


def randomize_lighting(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (500.0, 2000.0),
):
    """
    Randomize lighting intensity.

    Note: Lighting randomization typically happens via USD/RTX path
    and may require updates through the simulation context.
    """
    # Lighting randomization is typically handled through Replicator
    # or direct USD manipulation, not through PhysX
    pass


def randomize_materials(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float] = (0.3, 1.0),
    restitution_range: tuple[float, float] = (0.0, 0.3),
):
    """
    Randomize physics material properties.
    """
    num_envs = len(env_ids)

    # This would modify PhysX material properties
    # Implementation depends on how materials are organized in the scene
    pass


# Event term configurations for EventManager
def get_reset_events() -> dict[str, EventTermCfg]:
    """Get reset event configurations."""
    return {{
        "randomize_objects": EventTermCfg(
            func=randomize_object_poses,
            mode="reset",
        ),
        "randomize_articulations": EventTermCfg(
            func=randomize_articulation_state,
            mode="reset",
        ),
    }}


def get_interval_events() -> dict[str, EventTermCfg]:
    """Get interval event configurations."""
    return {{
        # Add periodic randomization events here
    }}
'''

    def _generate_init_file(self, config: TaskConfig, policy_id: str) -> str:
        """Generate package __init__.py."""
        class_name = self._to_class_name(config.task_name)
        return f'''"""
Isaac Lab Task Package - {config.task_name}
Generated by BlueprintRecipe

This package provides:
- {class_name}EnvCfg: Environment configuration
- {class_name}Task: Task implementation
- Domain randomization hooks
"""

from .env_cfg import {class_name}EnvCfg
from .task_{policy_id} import {class_name}Task
from .randomizations import get_reset_events, get_interval_events

__all__ = [
    "{class_name}EnvCfg",
    "{class_name}Task",
    "get_reset_events",
    "get_interval_events",
]
'''

    def _format_joint_pos(self, robot_type: str) -> str:
        """Format default joint positions dict."""
        config = self.ROBOT_CONFIGS.get(robot_type, self.ROBOT_CONFIGS["franka"])
        positions = config["default_joint_pos"]
        return "{" + ", ".join(f'"joint_{i}": {p}' for i, p in enumerate(positions)) + "}"

    def _to_class_name(self, task_name: str) -> str:
        """Convert task name to class name."""
        return "".join(word.capitalize() for word in task_name.split("_"))

    def _generate_reward_terms(self, config: TaskConfig, policy: dict[str, Any]) -> str:
        """Generate reward term configurations."""
        terms = []
        for component, weight in config.reward_weights.items():
            func_name = f"reward_{component}"
            terms.append(f'''    {component} = RewardTermCfg(
        func="{func_name}",
        weight={weight},
    )''')

        return "\n\n".join(terms) if terms else "    pass  # No reward terms configured"

    def _generate_event_terms(
        self,
        config: TaskConfig,
        recipe: dict[str, Any],
        policy: dict[str, Any]
    ) -> str:
        """Generate event term configurations."""
        terms = []

        # Reset events
        for event in config.randomization_config.get("on_reset", []):
            terms.append(f'''    {event["name"]} = EventTermCfg(
        func="randomize_{event["name"]}",
        mode="reset",
    )''')

        return "\n\n".join(terms) if terms else "    pass  # No event terms configured"

    def _format_reward_weights_yaml(self, weights: dict[str, float]) -> str:
        """Format reward weights for YAML."""
        if not weights:
            return "    # No reward weights configured"
        return "\n".join(f"  {k}: {v}" for k, v in weights.items())

    def _format_randomization_yaml(self, events: list[dict]) -> str:
        """Format randomization events for YAML."""
        if not events:
            return "    []"
        lines = []
        for event in events:
            lines.append(f"    - name: {event['name']}")
            if event.get("params"):
                lines.append("      params:")
                for k, v in event["params"].items():
                    lines.append(f"        {k}: {v}")
        return "\n".join(lines)

    def save(self, task: GeneratedTask, output_dir: str) -> dict[str, str]:
        """Save generated task to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}
        for filename, content in task.files.items():
            file_path = output_path / filename
            with open(file_path, "w") as f:
                f.write(content)
            saved_files[filename] = str(file_path)

        return saved_files

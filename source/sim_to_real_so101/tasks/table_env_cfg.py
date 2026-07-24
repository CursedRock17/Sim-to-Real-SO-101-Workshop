# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pick-and-place task: drop rectangular blocks into a cardboard box on a table.

This is the "custom workspace" starting example described in the naval-research
notes. It reuses the working SO-101 + lightbox + mat + camera setup from
``SO101TaskSceneCfg`` and adds three CAD assets:

* ``table.usd``        - static visual table the lightbox is mounted on
* ``CardboardBox.usd`` - static drop target (stable position, never moves)
* blocks               - dynamic rigid bodies with per-instance colors

The CAD USDs are visual-only meshes with no baked colliders. That is fine for the
static table (visual only) and the static box (a triangle-mesh collider is valid
for a non-moving body). It is NOT enough for the blocks: a dynamic rigid body
needs a convex collider, and the runtime spawn cfg can only toggle collision, not
generate an approximation. So the blocks are spawned as ``CuboidCfg`` primitives
(exact dimensions, a valid box collider, reliable per-instance color). To use the
CAD ``Block.usd`` instead, re-convert it with a baked convex collision mesh and
swap ``block_base.spawn`` back to a ``UsdFileCfg``.

NOTE: The placement constants and scales below are best-effort. Exact values
(surface height, block/box positions, whether any ``scale`` is needed) must be
verified in Isaac Sim inside the container -- the notes already flagged a
CAD unit/scale gotcha with the table.
"""
import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.envs.mdp import reset_root_state_uniform

from sim_to_real_so101 import assets
from sim_to_real_so101.mdp import randomize_robot_color, ROBOT_COLORS

from .task_env_cfg import (
    SO101TaskSceneCfg,
    SO101TaskEnvCfg,
    TaskEventCfg,
    TaskObservationsCfg,
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

# --------------------------------------------------------------------------- #
# Tunable placement constants -- VERIFY / TUNE IN THE CONTAINER.
# Coordinates follow the existing task frame: arm at origin, working surface
# (mat) at ~z=0.03, objects rest at ~z=0.05 (see vials_to_rack_env_cfg.py).
# --------------------------------------------------------------------------- #
SURFACE_Z = 0.05  # spawn height for objects resting on the working surface
TABLE_POS = (0.1, 0.0, 0.0)  # visual table; tune so its top meets the surface
BOX_POS = (0.20, 0.12, SURFACE_Z)  # cardboard box drop target (stable)
BLOCK_SIZE = (0.022, 0.022, 0.05)  # rectangular block (m); tune to the CAD block

# Block colors applied in Isaac Sim (matches the red/blue blocks in the notes).
BLOCK_COLORS = {
    "red": (0.8, 0.1, 0.1),
    "blue": (0.1, 0.2, 0.8),
}

# Base rigid block as a cuboid primitive: valid box collider + mass, colored in-sim.
block_base = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Block",
    spawn=sim_utils.CuboidCfg(
        size=BLOCK_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, -0.06, SURFACE_Z)),
)


@configclass
class SO101TableTaskSceneCfg(SO101TaskSceneCfg):
    """Table pick-and-place scene: reuse lightbox/mat/cameras, add table+box+blocks."""

    # Static visual table the lightbox is mounted on. Left collision-free: the
    # existing mat/ground already provides the support surface the arm works on,
    # so the table is a visual backdrop and won't fight the established physics.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{assets_path}/usd/table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )

    # Static cardboard box -- the stable drop target.
    # KNOWN LIMITATION: CardboardBox.usd is an *instanceable* prim, so Isaac Lab
    # skips the runtime collision API below (logs "Could not perform
    # 'modify_collision_properties'") and the box currently has NO collider --
    # blocks pass through it and rest on the surface at the box footprint. To get
    # real containment, bake a collider into the asset (re-convert / author a
    # collision mesh) or mark the prim non-instanceable, then this cfg applies.
    cardboard_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CardboardBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/CardboardBox.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=BOX_POS),
    )

    # Colored rigid blocks. Colors are applied in-sim via a bound preview surface.
    block_red = block_base.replace()
    block_red.prim_path = "{ENV_REGEX_NS}/Block_Red"
    block_red.init_state.pos = (0.22, -0.06, SURFACE_Z)
    block_red.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=BLOCK_COLORS["red"]
    )

    block_blue = block_base.replace()
    block_blue.prim_path = "{ENV_REGEX_NS}/Block_Blue"
    block_blue.init_state.pos = (0.22, -0.12, SURFACE_Z)
    block_blue.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=BLOCK_COLORS["blue"]
    )


@configclass
class TableTaskEventCfg(TaskEventCfg):
    """Events: inherit lightbox/mat/camera randomization, add block + arm-color DR."""

    # Randomize arm color across the full palette (notes: "random arm colors").
    # Overrides the base term, which only used orange.
    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={"color_names": list(ROBOT_COLORS.keys())},
    )

    # Randomize block start positions on reset; box stays put (stable target).
    reset_block_red = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-np.pi, np.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block_red"),
        },
    )

    reset_block_blue = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-np.pi, np.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block_blue"),
        },
    )


@configclass
class SO101TableTaskEnvCfg(SO101TaskEnvCfg):
    """Runner config for the table pick-and-place task."""

    scene: SO101TableTaskSceneCfg = SO101TableTaskSceneCfg()
    events: TableTaskEventCfg = TableTaskEventCfg()
    # Reuse the task observations (ego + external cameras + policy state).
    observations: TaskObservationsCfg = TaskObservationsCfg()

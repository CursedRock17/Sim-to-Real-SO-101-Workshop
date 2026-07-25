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
from isaacsim.core.utils.rotations import euler_angles_to_quat

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
# The CAD USDs are authored in millimeters; Isaac references them into a meters
# stage without applying metersPerUnit, so raw they spawn 1000x too big (the table
# was ~1 km wide). CAD_MM_TO_M rescales them to real size.
CAD_MM_TO_M = (0.001, 0.001, 0.001)
TABLE_POS = (0.1, 0.0, -0.025)  # tabletop ~ground level so arm/lightbox sit on it
# Box CENTER position. The mesh origin was recentered in CardboardBox.usd, so
# rotations pivot in place and this is the true geometric center (not a corner).
# z=0.091 puts the box bottom on the mat for the current rotation.
BOX_POS = (0.10, 0.20, 0.091)
BLOCK_SIZE = (0.022, 0.022, 0.05)  # rectangular block (m); tune to the CAD block

BOX_COLOR = (0.757, 0.604, 0.424)  # cardboard tan (from the CAD/STEP material)

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
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/table.usd",
            scale=CAD_MM_TO_M,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )

    # Static cardboard box -- the stable drop target. Converted from the STEP
    # file, which is authored in METERS (metersPerUnit=1.0), so unlike the table
    # it needs NO scale override. It is non-instanceable, renders its CAD
    # cardboard-brown material, and has a concave triangle-mesh collider
    # (UsdPhysics MeshCollisionAPI, approximation="none") baked in so the open top
    # is preserved -- blocks can be placed INSIDE and are contained by the walls/
    # floor. `collision_props` below enables that baked collider.
    cardboard_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CardboardBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/CardboardBox.usd",
            # The CAD-imported material doesn't render in Isaac; author the tan
            # here (the same known-good path the blocks use).
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BOX_COLOR),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        # Rotate so the open top faces up (roll/pitch/yaw about world X/Y/Z, deg).
        # Origin is recentered, so this pivots about the box center.
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BOX_POS,
            rot=euler_angles_to_quat(np.array([-90, 180, 90]), degrees=True),
        ),
    )

    # Colored rigid blocks. Colors are applied in-sim via a bound preview surface.
    # Spawn positions sit inside the top-down-reachable zone (reach_zone.py:
    # ENV x[0.034,0.194], y[-0.139,0.181]); with the +/-0.03 reset randomization
    # every sampled pose stays graspable.
    block_red = block_base.replace()
    block_red.prim_path = "{ENV_REGEX_NS}/Block_Red"
    block_red.init_state.pos = (0.16, 0.06, SURFACE_Z)
    block_red.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=BLOCK_COLORS["red"]
    )

    block_blue = block_base.replace()
    block_blue.prim_path = "{ENV_REGEX_NS}/Block_Blue"
    block_blue.init_state.pos = (0.16, -0.06, SURFACE_Z)
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

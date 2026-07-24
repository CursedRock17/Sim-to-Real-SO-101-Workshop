# Standard Library Imports
import os
import numpy as np

# Isaac Lab Imports
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat

# Local Imports
from sim_to_real_so101 import assets
from sim_to_real_so101.mdp import (
    randomize_light_exposure,
    randomize_sky_light,
    randomize_camera_focal_length,
    randomize_camera_pose,
    image,
    image_raw,
)

from .so101_env_cfg import (
    SO101TeleopEnvCfg,
    LerobotSo101BaseSceneCfg,
    EventCfg,
    ObservationsCfg,
)

# Where our local assets are located
assets_path = os.path.dirname(os.path.abspath(assets.__file__))

camera_height=480
camera_width=640
camera_object = TiledCameraCfg(
    prim_path="",
    update_period=0.0,
    height=camera_height,
    width=camera_width,
    data_types=["rgb"],
    colorize_instance_segmentation=True,
    spawn=sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        f_stop=100,  # x10 of real
        focal_length=13.5,  # 10th of real
        focus_distance=0.05,  # 5cm in front of the camera
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
        convention="opengl",
    ),
)


@configclass
class TemplateTaskSceneCfg(LerobotSo101BaseSceneCfg):
    # Core constructor class for what the scene surrounding the arm will look like
    # Add additional objects to the scene here, things like scene objects our arm will interact with, domain randomization traits, and containers for the arm (like a lightbox)

    # Camera - Setup the Isaac Sim makeup for additional cameras for the arm
    camera_ego = camera_object.replace()
    camera_ego.prim_path = "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
    camera_ego.offset.pos = (-0.005, 0.06, -0.062)
    camera_ego.offset.rot = euler_angles_to_quat(np.array([-45, 0, 0]), degrees=True)


@configclass
class TaskEventCfg(EventCfg):
    """Configuration for events."""

    reset_camera_ego_fov = EventTerm(
        func=randomize_camera_focal_length,
        mode="reset",
        params={
            "focal_length_range": (12.0, 15.0),  # ~±10% around 13.5mm
            "asset_cfg": SceneEntityCfg("camera_ego"),
        },
    )


@configclass
class TaskObservationsCfg(ObservationsCfg):

    @configclass
    class VisualCfg(ObsGroup):

        rgb_ego = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_ego"),
                "data_type": "rgb",
                "normalize": False,
            }
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    visual: VisualCfg = VisualCfg()


@configclass
class SO101TableTaskEnvCfg(SO101TeleopEnvCfg):
    """Configuration for the task environment. This class acts as a runner"""
    # Inherit SO101 Basic Information: Actions the arm can take, the physical construction of the USD, and joint physics

    # Key pieces of information:
    # Grab our core scene information
    scene: TemplateTaskSceneCfg = TemplateTaskSceneCfg()
    events: TaskEventCfg = TaskEventCfg()
    # Grab the observations our policy will be allowed to work with
    observations: TaskObservationsCfg = TaskObservationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        self.sim.render.enable_translucency = True
        carb_settings = {
            "rtx.reflections.enabled": True,
            "rtx.translucency.reflectAtAllBounce": True,
            "rtx.translucency.sampleRoughness": True,
            "rtx.translucency.reflectionThroughputThreshold": 0.05,
            "rtx.translucency.maxRefractionBounces": 5,
            "rtx.raytracing.fractionalCutoutOpacity": True,
        }
        self.sim.render.carb_settings = carb_settings

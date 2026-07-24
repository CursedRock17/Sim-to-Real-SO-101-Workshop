import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg

# import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Load in our OpenUSD Assets from the top of our package
from sim_to_real_so101 import assets
assets_path = os.path.dirname(os.path.abspath(assets.__file__))

# Create our Initailization Base Class
@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    # Basic Scene Info
    env_spacing = 4.0
    num_envs = 1

# Create our Basic Actions Class
@configclass
class ActionsCfg:
    # Basic Actions
    # TODO: define real action terms here. The previous placeholder instantiated
    # the abstract ActionTerm() at import time, which crashed import_packages()
    # and took down the entire tasks package.
    pass

# Create our Basic Observations Class
@configclass
class ObservationsCfg:
    # Basic Obs
    policy = 1

# Create our Configuration Class for a Simple Scene
@configclass
class SimpleSceneEnvCfg(ManagerBasedRLEnvCfg):
    # Scene configuration - required
    scene: BaseSceneCfg = BaseSceneCfg()

    # Simulation configuration - requires both
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # Reset configuration - requires both
    rewards = None  # No rewards for teleoperation
    terminations = None  # No terminations for teleoperation

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # General Settings
        self.decimation = 2
        self.episode_length_s = 5

        self.scene.num_envs = 1  # Always 1 env for teleoperation

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
    

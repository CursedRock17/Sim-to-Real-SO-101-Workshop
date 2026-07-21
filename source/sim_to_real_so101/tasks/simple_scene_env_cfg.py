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

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

# Create our Configuration Class for a Simple Scene
@configclass
class SimpleSceneEnvCfg(ManagerBasedRLEnvCfg):
    # simulation configuration

    # scene configuration

    # reset configuration

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
    

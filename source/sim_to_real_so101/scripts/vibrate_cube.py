import argparse
import math
import torch

from isaaclab.app import AppLauncher

# 1. Parse Launcher Arguments
parser = argparse.ArgumentParser(description="Vibration profile example in Isaac Lab.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulation app runtime
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports dependent on Omniverse App runtime
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext, SimulationCfg


def main():
    # 2. Simulation Context Setup
    sim_cfg = SimulationCfg(
        dt=0.002,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            enable_external_forces_every_iteration=True,  # Solves PhysX warning
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.2], target=[0.0, 0.0, 0.4])

    # 3. Spawn Scene Base
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0)
    cfg_light.func("/World/light", cfg_light)

    # 4. Spawn Dynamic Cube Object
    cube_cfg = RigidObjectCfg(
    prim_path="/World/VibratingCube",
    spawn=sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.3),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # Re-enable gravity if resting on ground
            linear_damping=0.2,
            angular_damping=0.5,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        # -------------------------------------------------------------
        # REQUIRED FOR COLLISIONS: Enables PhysX Collision Mesh/API
        # -------------------------------------------------------------
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.1,  # Low bounce
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.9)),
    ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cube = RigidObject(cfg=cube_cfg)

    # Initialize handles
    sim.reset()
    cube.reset()

    # 5. Vibration Profile Parameters Hertz & Newtons respectively
    freq_z = 100.0
    amp_z = 170.0
    
    freq_x = 525.0   
    amp_x = 75.0

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0

    print(f"[INFO]: Running vibration test. Physics dt = {sim_dt}s...")

    while simulation_app.is_running():
        # Calculate dynamic force scalars
        fx = amp_x * math.sin(2.0 * math.pi * freq_x * sim_time)
        fz = amp_z * math.sin(2.0 * math.pi * freq_z * sim_time)

        # Build tensors with 3D shape: (num_envs, num_bodies, 3) -> (1, 1, 3)
        forces = torch.tensor([[[fx, 0.0, fz]]], device=sim.device)
        torques = torch.zeros((1, 1, 3), device=sim.device)

        # Write force buffers using modern Wrench Composer API
        cube.permanent_wrench_composer.set_forces_and_torques(forces=forces, torques=torques)
        cube.write_data_to_sim()

        # Advance physics step
        sim.step()
        sim_time += sim_dt


if __name__ == "__main__":
    main()
    simulation_app.close()

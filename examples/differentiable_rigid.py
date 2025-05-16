import argparse

import torch

import genesis as gs

import numpy as np

import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="error")

    ########################## create a scene ##########################
    dt = 1e-2
    horizon = 100
    substeps = 1
    goal_pos = gs.tensor([0.7, 1.0, 0.05])
    goal_quat = torch.nn.functional.normalize(gs.tensor([0.3, 0.2, 0.1, 0.9]), p=2, dim=-1)
    optimize_init_pos = True
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            requires_grad=True,
            gravity=(0, 0, -1)           # disable gravity
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,         # disable collision for now
            enable_self_collision=False,    # disable self-collision for now
            enable_joint_limit=False,       # disable joint limit for now
            disable_constraint=True,        # disable constraint (e.g. collision) for now
            use_contact_island=False,       # disable contact island for now
            use_hibernation=False,          # disable hibernation for now
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -0.15, 2.42),
            camera_lookat=(0.5, 0.5, 0.1),
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    # plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    box = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0),
            size=(0.1, 0.1, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.0, 0.0, 1.0),
        ),
    )
    target = scene.add_entity(
        gs.morphs.Box(
            pos=goal_pos.cpu().tolist(),
            quat=goal_quat.cpu().tolist(),
            size=(0.1, 0.1, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 0.9, 0.0, 0.5),
        ),
    )

    ########################## cameras ##########################
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(-3.0, 1.5, 2.0),
        lookat=(0.5, 0.5, 0.1),
        fov=30,
        GUI=True,
    )

    ########################## build ##########################
    scene.build()

    ########################## optimize ##########################
    num_iter = 200
    lr = 1e-2
    record_every = 50
    
    if optimize_init_pos:
        init_pos = gs.tensor([0.3, 0.1, 0.28], requires_grad=True)
        init_quat = gs.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        init_vel = gs.tensor([0.0, 0.0, 1.0], requires_grad=True)
        init_ang = gs.tensor([1.0, 1.0, 1.0], requires_grad=True)
        optimizer = torch.optim.Adam([init_pos, init_quat, init_vel, init_ang], lr=lr)
    else:
        init_pos = gs.tensor([0.3, 0.1, 0.28], requires_grad=False)
        init_quat = gs.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=False)
        init_vel = gs.tensor([0.0, 0.0, 1.0], requires_grad=True)
        init_ang = gs.tensor([1.0, 1.0, 1.0], requires_grad=True)
        optimizer = torch.optim.Adam([init_vel, init_ang], lr=lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-3)
    
    
    bar = tqdm.tqdm(range(num_iter))
    for iter in bar:
        scene.reset()
        
        do_record = (iter % record_every == 0) or (iter == num_iter - 1)
        
        box.set_position(init_pos)
        box.set_quaternion(init_quat)
        box.set_velocity(init_vel)
        box.set_angular_velocity(init_ang)
        if do_record:
            cam.start_recording()
        
        loss = 0
        for i in range(horizon):
            scene.step()
            target.set_pos(0, goal_pos)
            target.set_quat(0, goal_quat)
            if do_record:
                cam.render()

        box_state = box.get_state()
        box_pos = box_state.pos[0]
        box_quat = box_state.quat[0]
        loss = torch.abs(box_pos - goal_pos).sum() + torch.abs(box_quat - goal_quat).sum()

        optimizer.zero_grad()
        loss.backward()  # this lets gradient flow all the way back to tensor input
        optimizer.step()
        scheduler.step()
        
        bar.set_description(f"loss: {loss.item():.4f} | lr: {scheduler.get_last_lr()[0]:.4f}")
        with torch.no_grad():
            init_quat.data = torch.nn.functional.normalize(init_quat.data, p=2, dim=-1)
            
        ## save the video
        if do_record:
            fps = 1.0 / dt
            script_name = __file__.split("/")[-1].split(".")[0]
            dir_name = "posvel" if optimize_init_pos else "vel"
            cam.stop_recording(save_to_filename=f"output/{script_name}/{dir_name}/cam_{iter:04d}.mp4", fps=fps)
        
    print("====================== Optimization Results")
    print("goal pos: ", goal_pos)
    print("init pos: ", init_pos)
    print()
    
    print("goal quat: ", goal_quat)
    print("init quat: ", init_quat)
    print()
    
    print("init vel: ", init_vel)


if __name__ == "__main__":
    main()

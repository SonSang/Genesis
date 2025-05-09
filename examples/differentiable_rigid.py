import argparse

import torch

import genesis as gs

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            requires_grad=True,
            gravity=(0, 0, 0)           # disable gravity
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
    plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    box = scene.add_entity(
        gs.morphs.Box(
            lower=(0.2, 0.1, 0.05),
            upper=(0.4, 0.3, 0.35),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.0, 0.0, 1.0),
        ),
    )

    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        pos=(1.5, 0.5, 2.42),
        lookat=(0.5, 0.5, 0.1),
        fov=30,
        GUI=True,
    )
    cam_1 = scene.add_camera(
        pos=(-3.0, 1.5, 2.0),
        lookat=(0.5, 0.5, 0.1),
        fov=30,
        GUI=True,
    )

    ########################## build ##########################
    scene.build()

    ########################## forward + backward twice ##########################
    for trial in range(1):
        scene.reset()
        horizon = 1
        
        init_pos = gs.tensor([0.3, 0.1, 0.28], requires_grad=True)

        # forward pass
        print("forward")
        timer = gs.tools.Timer()
        box.set_position(init_pos)
        loss = 0
        # v_list = []
        cam_0.start_recording()
        cam_1.start_recording()
        for i in range(horizon):
            # v_i = gs.tensor(np.ones(box.n_dofs), requires_grad=False)
            # box.set_dofs_velocity(v_i)
            # v_list.append(v_i)
            
            scene.step()
            cam_0.render()
            cam_1.render()

            # you can use a scene_state
            if i == 25:
                # compute loss
                goal = gs.tensor([0.5, 0.8, 0.05])
                box_pos = box.get_pos()
                loss += torch.pow(box_pos - goal, 2).sum()

            # you can also use an entity's state
            if i == horizon - 1:
                # compute loss
                goal = gs.tensor([0.5, 0.8, 0.05])
                box_pos = box.get_pos()
                loss += torch.pow(box_pos - goal, 2).sum()

        timer.stamp("forward took: ")
        # backward pass
        print("backward")
        loss.backward()  # this lets gradient flow all the way back to tensor input
        timer.stamp("backward took: ")
        # for v_i in v_list:
        #     print(v_i.grad)
        #     v_i.zero_grad()
        # # for w_i in w_list:
        # #     print(w_i.grad)
        # #     w_i.zero_grad()
        print(init_pos.grad)
        # init_pos.zero_grad()
        
        ### save the video
        script_name = __file__.split("/")[-1].split(".")[0]
        cam_0.stop_recording(save_to_filename=f"output/{script_name}/{trial}/cam_0.mp4", fps=30)
        cam_1.stop_recording(save_to_filename=f"output/{script_name}/{trial}/cam_1.mp4", fps=30)


if __name__ == "__main__":
    main()

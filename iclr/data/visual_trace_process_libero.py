import os
import h5py
import json
from pathlib import Path
import numpy as np
from PIL import Image
import transforms3d as t3d
import pickle
import tyro
from tqdm import tqdm
import argparse
from collections import defaultdict
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path

def get_gripper_pixel_position(obs, sim, camera_name="agentview", image_width=256, image_height=256):
    """
    Get gripper position in pixel coordinates from observation.
    
    Args:
        obs: Environment observation dictionary
        sim: MuJoCo simulation object
        camera_name: Name of the camera to project onto
        image_width: Width of the rendered image
        image_height: Height of the rendered image
    
    Returns:
        tuple: (u, v) pixel coordinates of gripper, or None if not visible
    """
    gripper_pos = obs["robot0_eef_pos"]
    
    # Get camera parameters
    camera_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[camera_id].copy()
    camera_mat = sim.data.cam_xmat[camera_id].reshape(3, 3).copy()
    fovy = sim.model.cam_fovy[camera_id]
    
    # Calculate focal length
    focal_length = (image_height / 2.0) / np.tan(np.radians(fovy) / 2.0)
    
    # Transform gripper position to camera coordinates
    translated_point = gripper_pos - camera_pos
    camera_point = camera_mat.T @ translated_point
    
    # Project to pixel coordinates
    u = image_width - ((camera_point[0] * focal_length / camera_point[2]) + (image_width / 2.0))
    v = (camera_point[1] * focal_length / camera_point[2]) + (image_height / 2.0)

    return u, v

def main(args):
    # Get default paths
    datasets_default_path = get_libero_path("datasets")
    bddl_files_default_path = get_libero_path("bddl_files")
    # init_states_default_path = get_libero_path("init_states")
    
    # Get benchmark instance
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[args.benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()

    # Load metainfo json file because the regeneration code removes some episodes from the dataset
    metainfo_json_path = os.path.join(datasets_default_path, f"{args.benchmark}_metainfo.json")
    with open(metainfo_json_path, "r") as f:
        metainfo = json.load(f)

    print(f"Number of tasks in {args.benchmark} benchmark: {num_tasks}")
    visual_trace_data = {}

    # Get demonstration files
    demo_files = [os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i)) for i in range(num_tasks)]

    for task_id in range(num_tasks):
        task_visual_trace_data = defaultdict(list)
        # Choose a demo file
        demo_file = demo_files[task_id]
        print(f"Demo file: {demo_file}")
        # Get task information
        task = benchmark_instance.get_task(task_id)
        print(f"Task name: {task.problem}")
        task_language = task.language
        print(f"Language instruction: {task_language}")
        # task_verb = task_language.replace(" ", "_") # get the task verb by replacing spaces with underscores
        task_verb = task.name
        print(f"Task verb: {task_verb}")
        # Create environment for rendering
        bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
        print(f"BDDL file: {bddl_file}")
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 256,
            "camera_widths": 256,
        }

        print("Creating offscreen render environment...")
        env = OffScreenRenderEnv(**env_args)

        # Load demonstration data
        with h5py.File(demo_file, "r") as f:
            for demo_id in tqdm(range(50)):
                demo_key = f"demo_{demo_id}"
                # only process the successful demonstrations
                if metainfo[task_verb][demo_key]["success"]:
                    # Load states
                    states = f[f"data/{demo_key}/states"][()]
            
                    # Reset environment to initial state, currently choose the first possible initial state
                    env.reset()
                    for state in states:
                        obs = env.set_init_state(state)
                        
                        # Get pixel position
                        u, v = get_gripper_pixel_position(obs, env.sim, "agentview", 256, 256)
                        task_visual_trace_data[demo_key].append((u, v))
        
        visual_trace_data[task_verb] = task_visual_trace_data

    with open(f"{datasets_default_path}/{args.benchmark}/visual_trace_im256.pkl", "wb") as f:
        pickle.dump(visual_trace_data, f)

    # Clean up
    env.close()
    print("\nVisual trace data collected complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="libero_object")
    args = parser.parse_args()
    main(args)
import json
import numpy as np
import h5py
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_path", type=str)
    parser.add_argument("--task_suite", type=str)
    parser.add_argument("--root_path", type=str)
    args = parser.parse_args()
    # Verb to Episode Mapping
    # Path to the input and output files
    metainfo_json_path = os.path.join(args.libero_path, f"{args.task_suite}_metainfo.json")
    verb_to_episode_json_path = os.path.join(args.root_path, f"config/data_config_libero/{args.task_suite}/verb_to_episode.json")

    # Load the original JSON file
    with open(metainfo_json_path, "r") as f:
        data = json.load(f)

    verb_to_episode = {}

    # Iterate through the top-level keys (verbs/tasks)
    for verb, demos in data.items():
        # demos is a dict of demo_name: {success: bool, ...}
        successful_demos = []
        for demo_name, demo_info in demos.items():
            if demo_info.get("success", False):
                successful_demos.append(demo_name)
        verb_to_episode[verb] = successful_demos

    # Save the new mapping to a JSON file
    with open(verb_to_episode_json_path, "w") as f:
        json.dump(verb_to_episode, f, indent=2)

    print(f"Saved verb_to_episode mapping to {verb_to_episode_json_path}")


    # Generate episode length mapping
    with open(verb_to_episode_json_path, "r") as f:
        verb_to_episode = json.load(f)
    episode_length_mapping = {}
    for verb, demos in verb_to_episode.items():
        with h5py.File(os.path.join(args.libero_path, f"{args.task_suite}/{verb}_demo.hdf5"), "r") as f:
            for demo in demos:
                episode_length_mapping[f"{verb}_{demo[5:]}"] = f[f"data/{demo}/obs/agentview_rgb"][()].shape[0]
    episode_length_mapping_json_path = os.path.join(args.root_path, f"config/data_config_libero/{args.task_suite}/epi_len_mapping_json.json")
    with open(episode_length_mapping_json_path, "w") as f:
        json.dump(episode_length_mapping, f, indent=2)
    
    print(f"Saved episode length mapping to {episode_length_mapping_json_path}")


    # Task Grouping
    task_grouping = {}
    task_grouping["ratios"] = {}
    task_grouping["tasks"] = {}
    task_grouping["ratios"][args.task_suite] = 1.0
    task_grouping["tasks"][args.task_suite] = list(verb_to_episode.keys())
    task_grouping_json_path = os.path.join(args.root_path, f"config/data_config_libero/{args.task_suite}/task_grouping.json")
    with open(task_grouping_json_path, "w") as f:
        json.dump(task_grouping, f, indent=2)

    print(f"Saved task grouping to {task_grouping_json_path}")



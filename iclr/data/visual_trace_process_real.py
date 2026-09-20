# Remember to switch to the molmo2 conda environment to run this file.

import os
import re
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import h5py
import tyro
from collections import defaultdict
import pickle
from tqdm import tqdm

def extract_coordinates(text):
    """
    This function extracts the coordinates from the generated text of Molmo 2
    """
    try:
        coords_str = text.split('coords="')[1].split('"')[0]
        numbers = list(map(int, coords_str.split()))
    except:
        numbers = []

    return numbers

def get_gripper_point_from_image(image, processor, model) -> tuple[float, float]:
    """
    This function gets the gripper point from the image using Molmo
    """
    if image is None:   # if the image is not available, return (-1., -1.)
        return (-1., -1.)
    
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text="Point to the robot gripper."),
                dict(type="image", image=image),
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # generate output
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    # only get generated tokens; decode them to text
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return extract_coordinates(generated_text)

def get_side_images_from_h5(data, episode_name, resolution=(240, 424, 3)):
    # img, action and proprio keys can be found in config/dataset_config_template.yaml
    side_images_binary = data[f"{episode_name}/observation/exterior_image_1_left"]

    side_images = []
    for side_image in side_images_binary:
        if len(side_image) == 0:
            side_images.append(None)
        elif len(side_image) == resolution[0] * resolution[1] * resolution[2] - 1:
            side_images.append(np.append(np.frombuffer(side_image, dtype="uint8"), 0).reshape(resolution))
        else:
            side_images.append(np.frombuffer(side_image, dtype="uint8").reshape(resolution))  # list of 240 x 424 x 3 images
    return side_images

def main(dataset_path: str, save_path: str):
    if os.path.exists(save_path):
        print(f"Loading visual trace data from {save_path}")
        with open(save_path, 'rb') as f:
            visual_trace_data = pickle.load(f)
    else:
        print(f"No visual trace data found at {save_path}, creating new one")
        visual_trace_data = defaultdict(list)
    data = h5py.File(dataset_path, "r")
    episode_names = sorted(list(data.keys()))

    # load the processor
    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo2-8B",
        trust_remote_code=True,
        dtype="auto",
        device_map="cuda:0"
    )

    # load the model
    model = AutoModelForImageTextToText.from_pretrained(
        "allenai/Molmo2-8B",
        trust_remote_code=True,
        dtype="auto",
        device_map="cuda:0"
    )

    for episode_name in episode_names:
        print(f"Processing episode {episode_name}")
        if episode_name in visual_trace_data.keys():    # if the episode is already processed, skip
            continue

        try:
            side_images = get_side_images_from_h5(data, episode_name)
            gripper_points = []
            for side_image in tqdm(side_images):
                gripper_point = get_gripper_point_from_image(side_image, processor, model)
                gripper_points.append(gripper_point)
            visual_trace_data[episode_name] = gripper_points
            with open(save_path, 'wb') as f:
                pickle.dump(visual_trace_data, f)
        except Exception as e:
            print(f"Error processing episode {episode_name}: {e}")
            continue

if __name__ == "__main__":
    tyro.extras.set_accent_color("yellow")
    tyro.cli(main)
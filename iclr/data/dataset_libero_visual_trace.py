# this is for data processing of the visual trace format similar to the MolmoAct paper

import json
import os
import h5py
import torch
import numpy as np
import torchvision.transforms as transforms
import pickle
from typing import Union
from .utils import euler_to_rot_6d, quat_to_rot_6d, euler_to_quat, load_json, convert_multi_step, convert_delta_action,\
    find_increasing_subsequences, create_prompt_mask, scale_action, make_visual_trace
from iclr.util.args import DatasetConfig, SharedConfig
from collections import defaultdict

class SequenceDataset_Libero_Visual_Trace(torch.utils.data.Dataset):
    proprio_keys = ["ee_states", "gripper_states"]
    image_keys = ["agentview_rgb", "eye_in_hand_rgb"]
    action_keys = ["actions"]

    # set minimum trajectory length
    # we use 30 as the control frequency of the robot is 15 Hz
    minimum_length : int = 30 
    maximum_length : int = 450

    # remove long tail situations 
    min_demos : int = 4 # each task group contains at least min_demos trajectories
    
    def __init__(self,
        dataset_config : DatasetConfig,
        shared_config : SharedConfig,
        vision_transform : transforms.Compose,
        no_aug_vision_transform : transforms.Compose = None,
        split : str = "train",
    ):
        # parse the dataset config 
        dataset_json = load_json(dataset_config.dataset_json)

        # dataset_paths: List of hdf5 paths
        dataset_paths = dataset_json["dataset_paths"]

        self.verbs_to_file = {}
        for dataset_path in dataset_paths:
            hdf5_file = h5py.File(dataset_path, 'r')
            self.verbs_to_file[os.path.basename(dataset_path)[:-10]] = hdf5_file

        self.shuffle_repeat_traj = dataset_config.shuffle_repeat_traj
        if self.shuffle_repeat_traj:
            assert dataset_config.sort_by_lang, "Shuffle repeat trajectory only works with sort by lang"

        # self.epi_len_mapping_json: mapping from episode names to episode lengths
        epi_len_mapping_jsons = dataset_json["epi_len_mapping_json"]
        self.epi_len_mapping_json = {}
        if isinstance(epi_len_mapping_jsons, str):
            epi_len_mapping_jsons = [epi_len_mapping_jsons]
        for epi_len_mapping_json in epi_len_mapping_jsons:
            self.epi_len_mapping_json.update(load_json(epi_len_mapping_json))
        
        # get all episode names
        self.episode_names = [name for name in self.epi_len_mapping_json.keys()]

        # filter episodes by their length
        self.episode_names = [
            name for name in self.episode_names if self.minimum_length <= self.epi_len_mapping_json[name] <= self.maximum_length
        ]

        # self.verb_to_episode: mapping from verb to a list of episode names
        verb_to_episode_jsons = dataset_json["verb_to_episode"]
        self.verb_to_episode = defaultdict(list)
        if isinstance(verb_to_episode_jsons, str):
            verb_to_episode_jsons = [verb_to_episode_jsons]

        # skip the verbs that are in val_verbs
        val_verbs = dataset_json["val_verbs"]
        for verb_to_episode_json in verb_to_episode_jsons:
            verb_to_episode = load_json(verb_to_episode_json)
            for k, v in verb_to_episode.items():
                if k in val_verbs:
                    continue    # skip the verb that is in val_verbs
                self.verb_to_episode[k].extend([f"{k}_{vi[5:]}" for vi in v])
        
        # sort the episode names so that the permutation is consistent
        self.episode_names = sorted(self.episode_names)

        # check if we use a fraction of the dataset, not used in LIBERO
        if dataset_config.dataset_fraction < 1.0:
            print("Using only a fraction of the dataset: ", dataset_config.dataset_fraction)
            if not dataset_config.sort_by_lang: 
                # if sort_by_lang, process the dataset_fraction by task
                num_demos = int(len(self.episode_names) * dataset_config.dataset_fraction)
                self.episode_names = self.episode_names[:num_demos]
        
        # define train test split
        self.split = split
        self.train_split = dataset_json["train_split"]  # current value is 0.95

        # set seed and shuffle the episode names
        rng = np.random.RandomState(seed=shared_config.seed)
        rng.shuffle(self.episode_names)
        num_train = int(len(self.episode_names) * self.train_split) # number of training episodes

        if self.split == "train": 
            self.episode_names = self.episode_names[:num_train]
        else:
            self.episode_names = self.episode_names[num_train:]

        # if sort by lang, we first shuffle the task permutation and then the episodes 
        # this ensures that for most indices, there's no overlap between tasks
        self.sort_by_lang = dataset_config.sort_by_lang # always be true

        if self.split == "val": # for validation, we only need at least one demo
            self.min_demos = 1

        if self.sort_by_lang:
            # remove all verbs that have less than min_demos episodes
            self.verb_to_episode = {k: sorted(v) for k, v in self.verb_to_episode.items() if len(v) >= self.min_demos}

            # ablation: use only a fraction of the dataset 
            if dataset_config.dataset_fraction < 1.0:   # not used, as dataset_fraction is always 1.0
                for k in self.verb_to_episode:
                    num_demos = max(int(len(self.verb_to_episode[k]) * dataset_config.dataset_fraction), self.min_demos)
                    self.verb_to_episode[k] = self.verb_to_episode[k][:num_demos]
                self.episode_names = [item for sublist in self.verb_to_episode.values() for item in sublist]

            # to support task_barrier, we need to calculate how many steps (in total) are available for each verb/task
            self.verb_to_numsteps = {
                k : sum([self.epi_len_mapping_json[epi] for epi in v]) for k, v in self.verb_to_episode.items()
            }
        
        # rebalance tasks
        self.rebalance_tasks = dataset_config.rebalance_tasks   # always be true
        if self.rebalance_tasks:
            assert self.sort_by_lang, "Rebalance tasks only works with sort by lang"
            # calculate median of the number of trajectories for each verb
            if self.split == "train": 
                self.rebalance_length = int(np.median([len(i) for i in self.verb_to_episode.values()]))
            else:
                self.rebalance_length = 5
            print("Each task is rebalanced to have length: ", self.rebalance_length)

        self.seq_length = shared_config.seq_length  # 512

        self.num_weighted_steps = dataset_config.num_weighted_steps # 30

        self.goal_conditioned = dataset_config.goal_conditioned # False

        #define rotation format 
        self.rot_6d = shared_config.rot_6d # True

        #non overlapping subsequence? 
        self.non_overlapping : Union[bool, int] = dataset_config.non_overlapping    # 32

        #enable repeating trajectory so that it can learn the copying behavior
        self.num_repeat_traj = dataset_config.num_repeat_traj   # always be 1 for now

        self.task_barrier = dataset_config.task_barrier # always be true
        self.skip_step = dataset_config.skip_step   # always be false
        self.proprio_noise = dataset_config.proprio_noise
        self.visual_trace_noise = dataset_config.visual_trace_noise
        self.action_noise = dataset_config.action_noise

        # vision transform, used for side camera 
        # we do not need normalization, see get_item
        if vision_transform is not None:    # True
            # remove the ToTensor and ColorJitter transforms
            self.vision_transform = transforms.Compose([t for t in vision_transform.transforms if not isinstance(t, transforms.ToTensor) and not isinstance(t, transforms.ColorJitter)])
        else:
            print("warning: vision transforms are not defined. Using default transforms.")
            self.vision_transform = transforms.Compose([
                transforms.Resize(size=248, max_size=None, interpolation=transforms.InterpolationMode.BICUBIC, antialias='warn'), # kept consistent with default
                transforms.CenterCrop(size=224),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])

        # no_aug_vision_transform, used for wrist camera 
        if no_aug_vision_transform is not None: # True
            self.no_aug_vision_transform = transforms.Compose([t for t in no_aug_vision_transform.transforms if not isinstance(t, transforms.ToTensor) and not isinstance(t, transforms.ColorJitter)])
        else:
            self.no_aug_vision_transform = self.vision_transform
        print("vision transforms")

        print(self.vision_transform)

        if dataset_config.vision_aug:   # always be true
            self.vision_aug = True
            self.contrast_range = [0.8, 1.2]
            self.brightness_range = [-0.1, 0.1]
            print("using numeric brightness and contrast augmentation")
            print("contrast range: ", self.contrast_range)
            print("brightness range: ", self.brightness_range)
        else:
            self.vision_aug = False

        # change prediction to be k steps 
        self.num_pred_steps = shared_config.num_pred_steps  # 16
        assert self.num_pred_steps >= 1, "Number of prediction steps must be at least 1"
        print("Number of prediction steps: ", self.num_pred_steps)

        # rebalance the dataset with respect to the number of tasks in each group. The grouping is calculated so that
        # each group is repeated the same number of times 
        self.task_grouping = dataset_json.get("task_grouping", None)    # group verbs by jobs, e.g., "push", "poke", etc.
        if self.task_grouping is not None: 
            task_grouping = json.load(open(self.task_grouping, 'r'))
            average_num_tasks = np.mean([len(v) for v in task_grouping["tasks"].values()])
            print("Average number of tasks: ", average_num_tasks)
            self.upweight_tasks = {}    # a dictionary that maps task to its upweighting factor

            ratios = task_grouping.get("ratios", None)
            if ratios is not None:
                print("overriding with known ratio: ", ratios)
                for k, task_lists in task_grouping["tasks"].items():
                    task_ratio = ratios[k]
                    for t in task_lists:
                        self.upweight_tasks[t] = task_ratio
            else:
                for _, task_lists in task_grouping["tasks"].items():
                    task_len = len(task_lists)
                    upweight_factor = average_num_tasks / task_len
                    for t in task_lists:
                        self.upweight_tasks[t] = upweight_factor

        self.num_visual_trace_points = shared_config.num_visual_trace_points
        visual_trace_path = dataset_json["visual_trace_path"]
        visual_trace_data = pickle.load(open(visual_trace_path, 'rb'))
        self.key_to_visual_trace_data = self.load_and_process_visual_trace_data(visual_trace_data)
        # load the dataset 
        self.shuffle_dataset(seed=0)

    def load_and_process_visual_trace_data(self, visual_trace_data):
        """
        Load and process the visual trace data
        """
        key_to_visual_trace_data = {}
        for task in visual_trace_data.keys():
            for demo_id, data in visual_trace_data[task].items():
                data = np.array(data)
                data = data / np.array([256., 256.])
                data = make_visual_trace(data, have_depth=False, n=self.num_visual_trace_points)
                data = torch.from_numpy(data).float()
                data = data.reshape(-1, self.num_visual_trace_points * 2)
                key_to_visual_trace_data[f"{task}_{demo_id[5:]}"] = data
        return key_to_visual_trace_data
    
    def total_seq_length(self):
        """
        Calculate the total sequence length of the dataset
        """
        total_seq_length = 0
        for name in self.episode_names:
            total_seq_length += self.epi_len_mapping_json[name]
        return total_seq_length
    
    def update_seq_length(self, new_seq_length : int):
        """
        Update the sequence length
        """
        self.seq_length = new_seq_length
    
    def save_split(self, path : str):
        """
        Save the train test split to a json file
        """
        with open(path, 'w') as f:
            json.dump(self.episode_names, f)
    
    def shuffle_dataset(self, seed=0):
        if self.goal_conditioned:   # false
            self.shuffle_dataset_goal_conditioned(seed)
        elif self.sort_by_lang:   # true
            self.shuffle_dataset_sort_by_lang(seed)
        else:
            self.shuffle_dataset_default(seed)
    
    def shuffle_dataset_default(self, seed=0):
        pass

    def shuffle_dataset_goal_conditioned(self, seed=0): # don't care for now
        pass

    def shuffle_dataset_sort_by_lang(self, seed=0): 
        """
        Shuffle the dataset according to the seed
        """
        rng = np.random.RandomState(seed=seed)
        # first we shuffle the verbs 
        verbs = list(self.verb_to_episode.keys())
        rng.shuffle(verbs)

        self.steps = [] # A list that stores all the individual steps from all episodes across all verbs/tasks
        verb_to_idx = defaultdict(list) # a dictionary that maps verbs to lists of global step indices
        # count_discard = 0
        for v in verbs: # loop through the verbs
            # shuffle the episode ids
            if self.rebalance_tasks:    # True
                # update rebalance length based on task grouping if defined 
                rl = self.rebalance_length
                if self.task_grouping is not None:
                    rl = int(rl * self.upweight_tasks[v])
                if len(self.verb_to_episode[v]) < rl:
                    replace = True
                else:
                    replace = False
                indices = rng.choice(len(self.verb_to_episode[v]), size=rl, replace=replace)
            else:   # if not rebalancing, we just shuffle the episodes
                indices = rng.permutation(len(self.verb_to_episode[v]))
            
            repeats = rng.choice(np.arange(self.num_repeat_traj), size=len(self.verb_to_episode[v]), replace=True) + 1  # a list numbers of repeats for each episode, length is the number of episodes of the verb v
            episode_keys = [self.verb_to_episode[v][i] for i in indices]    # a list of episode keys of verb v, with indices from indices (length is rl)
            task_length = sum([self.epi_len_mapping_json[key] * r for key, r in zip(episode_keys, repeats)])    # total number of steps of the verb v (eisodes can be repeated).
            if task_length < self.seq_length:
                continue
            
            # initialize the current step index for the verb 
            verb_step_idx = 0   # increasing step index for the steps of the verb v, renew for each verb
            cache = []   # a list of steps across all episodes of the verb v, renew for each verb
            ranges = [] # a list of tuples, each tuple contains the start and end index of the trajectory of an episode (can be repeated) in the global step space

            # calculate the ranges of trajectories in index space 
            start_idx = 0 # renew for each verb
            for key, num_repeat in zip(episode_keys, repeats):  # loop through the episodes of the verb v and their repeats
                trajectory_length = self.epi_len_mapping_json[key]  # trajectory length of the episode
                for repeat_i in range(num_repeat):  # loop through the repeats of the episode
                    ranges.append((start_idx, start_idx + trajectory_length)) # (inclusive, exclusive)
                    start_idx += trajectory_length

            for key, num_repeat in zip(episode_keys, repeats): # loop through the episodes of the verb v and their repeats
                for repeat_i in range(num_repeat):  # loop through the repeats of the episode
                    for s in range(self.epi_len_mapping_json[key]): # loop through the steps of the episode
                        cache.append(
                            {
                                "episode_id" : key, 
                                "step" : s, # local step index in the episode
                                "eos" : s == self.epi_len_mapping_json[key] - 1, # whether the step is the last step of the episode
                            }
                        )
                        
                        # if task_barrier = True, it ensures that within each batch, there is only one verb/task 
                        # if verb_step_idx + self.seq_length > task_length (which is the total number of steps of the verb v), it means that 
                        # we have reached the end of the all steps of the verb v, and we shouldn't include the next step
                        # This ensures that when creating a sequence of length 512, it doesn't span beyond the current task
                        if self.task_barrier and verb_step_idx + self.seq_length > task_length:
                            continue
                        else:
                            # update the verb to idx mapping
                            verb_to_idx[v].append(len(self.steps) + verb_step_idx)  # ver_to_idx[v] is a list of global step indices of the verb v, where each index is an valid starting point for a training sequence
                        verb_step_idx += 1
            if self.shuffle_repeat_traj:    # true
                for i in range(len(ranges) - 1):
                    if rng.uniform() < 0.5:
                        ranges[i], ranges[i + 1] = ranges[i + 1], ranges[i] # ranges[i] is a tuple of (start, end) index of the trajectory in the global step space
                # then shuffle the cache based on the ranges
                cache = [cache[i] for r in ranges for i in range(r[0], r[1])]
            self.steps.extend(cache)    # only extend the cache to the steps after finish processing the verb v
            
        # every index in usable_indices is a valid starting point for a training sequence, because if it is not, then we haven't append it to the verb_to_idx[v] list
        self.usable_indices = [idx for list_of_gobal_indices in verb_to_idx.values() for idx in list_of_gobal_indices]
        # len of self.steps is 106575, len of self.usable_indices is 93289

    def __len__(self):
        """
        return the length of the dataset
        """
        if (self.sort_by_lang and self.task_barrier) or self.goal_conditioned:
            data_length = len(self.usable_indices)  # if sort_by_lang and task_barrier, the number of possible sequences is the number of usable indices in self.usable_indices
        else:
            data_length = len(self.steps) - self.seq_length + 1
        
        if self.non_overlapping:    # currently be 32, this controls the stride between consecutive sequences in the dataset
            if isinstance(self.non_overlapping, bool):
                new_data_length = data_length // self.seq_length
            else:
                new_data_length = data_length // self.non_overlapping

            if data_length<self.seq_length and data_length>0:
                data_length = 1
            else:
                data_length = new_data_length

        return data_length
    
    def __getitem__(self, index):
        """
        Get the subsequence of the dataset starting from index to index + sequence_length
        return a diction of shape 
        {
            "observation": torch.Tensor, shape (seq_length, num_cameras, 3, 224, 224)
            "proprio": torch.Tensor, shape (seq_length, num_pred_steps, proprio_dim)
            "action": torch.Tensor, shape (seq_length, num_pred_steps, action_dim)
        }
        """
        if self.non_overlapping:
            if isinstance(self.non_overlapping, bool):  # if self.non_overlapping is True, we use the sequence length as the stride
                index = index * self.seq_length
            else:   # if self.non_overlapping is an integer, we use the integer as the stride
                index = index * self.non_overlapping

        if (self.sort_by_lang and self.task_barrier) or self.goal_conditioned:
            # use self.usable_indices to map index to a subsequence that only contains one task
            index = self.usable_indices[index]
        
        subseq = self.steps[index : index + self.seq_length + self.num_pred_steps - 1]  # includes both input sequence and prediction targets? seq_length + num_pred_steps - 1
        eos = np.array([s["eos"] for s in subseq]) # (seq_length + num_pred_steps - 1,)
        eos = torch.from_numpy(eos).float()
        start_end_epi = defaultdict(list)
        obs_start_end_epi = defaultdict(list)
        for idx, s in enumerate(subseq):
            start_end_epi[s["episode_id"]].append(s["step"])
            if idx < self.seq_length:
                obs_start_end_epi[s["episode_id"]].append(s["step"])    # observation data is limited to the sequence length
        
        start_end_epi = {
            episode_name : find_increasing_subsequences(list_of_steps) for episode_name, list_of_steps in start_end_epi.items() 
        }

        obs_start_end_epi = {
            episode_name : find_increasing_subsequences(list_of_steps) for episode_name, list_of_steps in obs_start_end_epi.items()
        }

        proprio = self.helper_load_proprio(start_end_epi) # (seq_length + num_pred_steps - 1, 10)
        action = self.helper_load_action(start_end_epi) # (seq_length + num_pred_steps - 1, 10)
        visual_trace = self.helper_load_visual_trace(start_end_epi) # (seq_length + num_pred_steps - 1, 32)
        # concatenate eos to action 
        action = torch.cat([action, eos[:, None]], dim=-1) # (seq_length + num_pred_steps - 1, 11), append eos to the action
        
        proprio = self.convert_multi_step(proprio, eos)[:self.seq_length] # (seq_length, num_pred_steps, 10)
        action = self.convert_multi_step(action, eos)[:self.seq_length] # (seq_length, num_pred_steps, 11)
        visual_trace = self.convert_multi_step(visual_trace, eos)[:self.seq_length] # (seq_length, num_pred_steps, 32)
        observation = self.helper_load_image(obs_start_end_epi)

        if self.goal_conditioned:
            pass
        else:
            prompt_mask, weight_mask = create_prompt_mask(action[...,0,-1], self.num_weighted_steps)    # prompt mask is the mask for the prompt, 0 = masked, 1 = not masked. weight mask is the mask for the weight, currently step_weight is 1.0, so does not matter.
            prompt_mask = torch.from_numpy(prompt_mask).float()
            weight_mask = torch.from_numpy(weight_mask).float() 
        
        return {
            "observation": observation,
            "proprio": proprio,
            "action": action,
            "visual_trace": visual_trace,
            "prompt_mask": prompt_mask,
            "weight_mask": weight_mask
        }
    def convert_multi_step(self, data : torch.Tensor, eos : Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 
        """Convert the data for multi step prediction 
        Args:
            data: torch.Tensor, of shape (seq_length + num_pred_steps - 1, 10) or (seq_length + num_pred_steps - 1, 11)
            eos: torch.Tensor, of shape (seq_length + num_pred_steps - 1,)
        Returns:
            torch.Tensor, of shape (seq_length + num_pred_steps - 1, num_pred_steps, 10) or (seq_length + num_pred_steps - 1, num_pred_steps, 11)
        """
        if self.num_pred_steps == 1:
            return data.unsqueeze(1)
        if isinstance(eos, torch.Tensor):
            eos = eos.numpy()
        pos = np.concatenate([np.array([0]), np.nonzero(eos)[0] + 1, np.array([self.seq_length + self.num_pred_steps - 1])])    # np.nonzero(eos)[0] is the global indices of the eos in the data
        data_chunked = []
        for i in range(1, len(pos)):
            demo_start = pos[i - 1]
            demo_end = pos[i]
            data_chunked.append(convert_multi_step(data[demo_start : demo_end], self.num_pred_steps))
        return torch.cat(data_chunked, dim=0)

    def helper_load_proprio(self, start_end_epi):
        """
        Load proprioception data from the dataset
        """
        proprio = {}
        for k in self.proprio_keys: # ee_states or gripper_states
            data = []
            for epi in start_end_epi:
                for s, e in start_end_epi[epi]:
                    data.append(self.get_key_from_demo(epi, k, s, e))
            proprio[k] = np.concatenate(data, axis=0)
        
        ret = proprio[self.proprio_keys[0]] # ee_states
        if self.proprio_noise > 0:
            ret += np.random.normal(0, self.proprio_noise, ret.shape)
            if ret.shape[1] == 7:
                ret[:, 3:] /= np.linalg.norm(ret[:, 3:], axis=-1, keepdims=True)
        
        rot = ret[:, 3:]
        # deal with rot_6d 
        if self.rot_6d:
            if rot.shape[1] == 4:
                # robomimic dataset has format wxyz
                rot = quat_to_rot_6d(rot)
            elif rot.shape[1] == 3:
                rot = euler_to_rot_6d(rot)
            ret = np.concatenate([ret[:, :3], rot], axis=-1)
            proprio[self.proprio_keys[0]] = ret
        else:
            if rot.shape[1] == 3:
                # convert to quaternion (only happens for droid, which uses XYZ as the rotation format)
                rot = euler_to_quat(rot)
                # update the proprio
                ret = np.concatenate([ret[:, :3], rot], axis=-1)
                proprio[self.proprio_keys[0]] = ret
        proprio_vec = np.concatenate([proprio[k] for k in self.proprio_keys], axis=-1)
        proprio_vec = torch.from_numpy(proprio_vec).float()
        # print(proprio_vec.shape)  # (seq_length + num_pred_steps - 1, 11), 11 since we have 2 values for gripper_states in LIBERO
        return proprio_vec

    def helper_load_action(self, start_end_epi):
        """
        Load action data from the dataset
        """
        action = {}
        for k in self.action_keys:
            data = []
            for epi in start_end_epi:
                for s, e in start_end_epi[epi]:
                    data.append(self.get_key_from_demo(epi, k, s, e))
            action[k] = np.concatenate(data, axis=0)
        
        # This is not very good, as the action noise is added to the entire action sequence (including the gripper part), not just the action part
        if self.action_noise > 0:
            ret = action[self.action_keys[0]]
            ret += np.random.normal(0, self.action_noise, ret.shape)
            action[self.action_keys[0]] = ret
        
        if self.rot_6d:
            ret = action[self.action_keys[0]]
            rot = ret[:, 3:-1]
            rot = euler_to_rot_6d(rot)
            ret = np.concatenate([ret[:, :3], rot, ret[:, -1:]], axis=-1)
            action[self.action_keys[0]] = ret

        action_vec = np.concatenate([action[k] for k in self.action_keys], axis=-1)
        action_vec = torch.from_numpy(action_vec).float()
        return action_vec
    
    def helper_load_image(self, start_end_epi):
        image = {}
        dtype = None
        for k in self.image_keys:
            data = []
            for epi in start_end_epi:
                for s, e in start_end_epi[epi]:
                    subsequence = self.get_key_from_demo(epi, k, s, e)
                    if dtype is None:
                        dtype = subsequence.dtype
                        if dtype == 'uint8':
                            norm = 255.0
                        else:
                            norm = 1.0
                    subsequence = torch.from_numpy(subsequence / norm)
                    # data aug for brightness and contrast 
                    if self.vision_aug:
                        contrast = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
                        brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
                        subsequence = contrast * subsequence + brightness
                    # permute from T, H, W, C to T, C, H, W
                    subsequence = subsequence.permute(0, 3, 1, 2)
                    # transform each subsequence independently
                    if "wrist" in k or "hand" in k: # wrist camera
                        subsequence = self.no_aug_vision_transform(subsequence).float()
                    else:
                        subsequence = self.vision_transform(subsequence).float()
                    data.append(subsequence)
            image[k] = torch.cat(data, dim=0) # concat on the time axis 
        image_vec = torch.stack([image[k] for k in self.image_keys], dim=1).float()
        return image_vec

    def helper_load_visual_trace(self, start_end_epi):
        visual_trace_data = []
        for epi in start_end_epi:
            for s, e in start_end_epi[epi]:
                visual_trace_data.append(self.get_visual_trace_from_demo(epi, s, e))
        visual_trace_vec = np.concatenate(visual_trace_data, axis=0)
        if self.visual_trace_noise > 0:
            visual_trace_vec += np.random.normal(0, self.visual_trace_noise, visual_trace_vec.shape)
        visual_trace_vec = torch.from_numpy(visual_trace_vec).float()
        return visual_trace_vec

    def get_key_from_demo(self,
        demo_id : str,
        key : str,
        seq_begin_index : int,
        seq_end_index : int
    ) -> np.ndarray:
        """
        Get the key from the demo
        Args:
            demo_id: str, episode id
            key: str, the key to get from the demo
            seq_begin_index: int, the beginning index of the sequence
            seq_end_index: int, the ending index of the sequence (inclusive)
        Returns:
            np.ndarray, the data from the demo
        """
        # get the verb from the demo id
        verb = demo_id.rsplit("_", 1)[0]
        demo_number = demo_id.rsplit("_", 1)[-1]
        # obtain the hdf5 file handle
        f_handle = self.verbs_to_file[verb]
        # get the data from the hdf5 file
        if key == "actions":
            data = f_handle[f"data/demo_{demo_number}/{key}"][seq_begin_index:seq_end_index + 1]
        else:
            data = f_handle[f"data/demo_{demo_number}/obs/{key}"][seq_begin_index:seq_end_index + 1]
        if 'rgb' in key:
            data = np.frombuffer(data, dtype='uint8').reshape(-1, 256, 256, 3)[:, ::-1, :, :]  # flip the image vertically
        return data

    def get_visual_trace_from_demo(
        self,
        demo_id : str,
        seq_begin_index : int,
        seq_end_index : int,
    ) -> np.ndarray:
        """
        Get the visual trace from the demo
        Args:
            demo_id : str, episode id, with task prefix
            seq_begin_index : int, the beginning index of the sequence
            seq_end_index : int, the ending index of the sequence (inclusive)
        Returns:
            np.ndarray, the visual trace from the demo
        """
        data = self.key_to_visual_trace_data[demo_id][seq_begin_index:seq_end_index + 1]
        return data
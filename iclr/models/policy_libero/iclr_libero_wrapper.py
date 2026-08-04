import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import yaml
import numpy as np
import timm
from typing import Union, Optional, List
from pathlib import Path
import PIL

from iclr.util.args import ExperimentConfig
import iclr.util.misc as misc
from iclr.util.model_constructor_libero import model_constructor_libero_visual_trace_mma
from iclr.data.utils import rot_6d_to_euler, quat_to_rot_6d, euler_to_rot_6d
from iclr.data.utils import convert_delta_action

def undo_vision_transform(obs : torch.Tensor, mean : tuple, std : tuple):   # never used
    """
    Undo the vision transform applied to the observations.
    torch tensor has shape T, num_cam, 3, H, W
    return np.ndarray with shape T, num_cam * H, W, 3 at np.uint8
    """
    # undo normalization
    mean, std = torch.tensor(mean), torch.tensor(std)
    obs = obs.permute(0, 1, 3, 4, 2)
    obs = obs * std + mean
    obs = obs.numpy()
    obs = np.clip(obs * 255, 0, 255).astype(np.uint8)
    obs = np.concatenate([obs[:, i] for i in range(obs.shape[1])], axis=1)
    return obs

class ICLRLiberoWrapper(nn.Module):
    def __init__(
        self, 
        train_yaml_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        vision_encoder_path: Optional[Union[str, Path]] = None,
        llama_checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()

        # loading experiment config 
        args : ExperimentConfig = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader)
        self.args = args

        if llama_checkpoint_path is not None:
            args.model_cfg.policy_cfg.llama_ckpt_dir = llama_checkpoint_path

        self.device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.shared_cfg.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True
        
        # start model construction
        if vision_encoder_path is not None:
            args.model_cfg.vision_encoder_cfg.vision_encoder = vision_encoder_path
        else:
            print("Vision encoder is loaded from the model checkpoint! ")

        model = model_constructor_libero_visual_trace_mma(
            model_config=args.model_cfg, 
            shared_config=args.shared_cfg,
            train=False,
        )
        # model.vision_encoder.model.pretrained_cfg now is {'url': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth', 'hf_hub_id': 'timm/vit_base_patch16_224.mae', 'architecture': 'vit_base_patch16_224', 'tag': 'mae', 'custom_load': False, 'input_size': (3, 224, 224), 'fixed_input_size': True, 'interpolation': 'bicubic', 'crop_pct': 0.9, 'crop_mode': 'center', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'num_classes': 0, 'pool_size': None, 'first_conv': 'patch_embed.proj', 'classifier': 'head', 'license': 'cc-by-nc-4.0'}

        # obtain vision transforms 
        timm_data_cfg = timm.data.resolve_data_config(model.vision_encoder.model.pretrained_cfg)    # timm_data_cfg:  {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 0.9, 'crop_mode': 'center'}
        self.preprocess = timm.data.create_transform(**timm_data_cfg)
        self.mean, self.std = timm_data_cfg["mean"], timm_data_cfg["std"]
        
        print("vision transform: ", self.preprocess)
        model.to(self.device)

        total, trainable = model.get_total_parameters(), model.get_trainable_parameters()
        print("trainable: ", trainable)
        print("Total params: ", total)
        print("percentage trainable: ", trainable / total)
        
        # loading pretrained checkpoint
        print("loading pretrained model from: ", checkpoint_path)
        misc.load_model(model, checkpoint_path)
        model.eval()

        self.model = model

        # removet the ToTensor and ColorJitter transforms
        if self.preprocess is not None:
            self.preprocess_PIL = transforms.Compose(
                [t for t in self.preprocess.transforms if not isinstance(t, transforms.ColorJitter)]
            )
            self.preprocess_tensor = transforms.Compose(
                [t for t in self.preprocess.transforms if not isinstance(t, transforms.ToTensor) and not isinstance(t, transforms.ColorJitter)]
            )
        else:
            print("warning: vision transforms are not defined. Using default transforms.")
            self.preprocess_PIL = transforms.Compose([
                transforms.Resize(size=248, max_size=None, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), 
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
            self.preprocess_tensor = transforms.Compose([
                transforms.Resize(size=248, max_size=None, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), 
                transforms.CenterCrop(size=224),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        self.reset()    # reset with action_exec_horizon = num_pred_steps

    def reset(self, action_exec_horizon = None):
        if action_exec_horizon is None:
            action_exec_horizon = self.model.num_pred_steps
        self.model.reset(action_exec_horizon)

    def prompt(
        self,
        side_image: Union[PIL.Image.Image, List[PIL.Image.Image]], 
        wrist_image : Union[PIL.Image.Image, List[PIL.Image.Image]], 
        proprio : Union[np.ndarray], 
        visual_trace : Union[np.ndarray],
        action : Optional[np.ndarray] = None,
    ):
        """
        Prompt the model with a demo

        Args:
            side_image (PIL.Image.Image or List[PIL.Image.Image]): side camera image
            wrist_image (PIL.Image.Image or List[PIL.Image.Image]): wrist camera image
            proprio (np.ndarray): proprioceptive information
            action (np.ndarray): action information
        """
        demo_sequence = self.prepare_observations(side_image, wrist_image, proprio, visual_trace, action)
        # self.reset()    # This set action_exec_horizon to num_pred_steps and call to the reset function of the model
        # demo_sequence is a dictionary
        for k, v in demo_sequence.items():
            demo_sequence[k] = v.to(self.device, non_blocking=True)
        # demo_sequence now is a dictionary with the following keys: "observation", "proprio", "action"
        self.model.prompt(demo_sequence)

    def batch_prompt(
        self,
        side_image: List[Union[PIL.Image.Image, List[PIL.Image.Image]]], 
        wrist_image : List[Union[PIL.Image.Image, List[PIL.Image.Image]]], 
        proprio : Union[np.ndarray],    # (batch_size, seq_len, proprio_dim) 
        visual_trace : Union[np.ndarray],    # (batch_size, seq_len, num_visual_trace_points * 3)
        action : Optional[np.ndarray] = None,    # (batch_size, seq_len, action_dim)
    ):
        """
        Prompt the model with a batch of demos: same demo but mutiplied by batch_size

        Args:
            side_image (List[Union[PIL.Image.Image, List[PIL.Image.Image]]]): side camera image
            wrist_image (List[Union[PIL.Image.Image, List[PIL.Image.Image]]]): wrist camera image
            proprio (np.ndarray): proprioceptive information
            action (Optional[np.ndarray]): action information
        """
        batch_demo_sequence = self.prepare_batch_observations(side_image, wrist_image, proprio, visual_trace, action)
        # self.reset()    # This set action_exec_horizon to num_pred_steps and call to the reset function of the model
        # batch_demo_sequence is a dictionary
        for k, v in batch_demo_sequence.items():
            batch_demo_sequence[k] = v.to(self.device, non_blocking=True)
        # batch_demo_sequence now is a dictionary with the following keys: "observation", "proprio", "action"
        self.model.prompt(batch_demo_sequence)

    def prepare_observations(
        self, 
        side_image: Union[PIL.Image.Image, List[PIL.Image.Image], np.ndarray], 
        wrist_image : Union[PIL.Image.Image, List[PIL.Image.Image], np.ndarray], 
        proprio : Union[np.ndarray], 
        visual_trace : Union[np.ndarray],
        action : Optional[np.ndarray] = None
    ):
        """
        assume the observation is a dictionary with the following keys:
        "observation", "proprio", "action"
        we also have visual_trace
        proprio in xyzXYZGripper format
        action in xyzXYZGripper format
        """
        # Following is just the preprocessing of side and wrist images
        if isinstance(side_image, PIL.Image.Image): # this happens in __call__, when there is only one side image
            side_image = [side_image]
            preprocess = self.preprocess_PIL
        elif isinstance(side_image, np.ndarray):
            preprocess = self.preprocess_tensor
            if side_image.dtype == np.uint8:
                side_image = torch.from_numpy(side_image / 255.0)
                side_image = side_image.permute(0, 3, 1, 2)
        else:   # now, this is the case
            preprocess = self.preprocess_PIL

        if isinstance(wrist_image, PIL.Image.Image): # this happens in __call__, when there is only one wrist image
            wrist_image = [wrist_image]
        elif isinstance(wrist_image, np.ndarray):
            if wrist_image.dtype == np.uint8:
                wrist_image = torch.from_numpy(wrist_image / 255.0)
                wrist_image = wrist_image.permute(0, 3, 1, 2)

        # find the length of the sequence, which is the length of side_image, wrist_image and proprio
        seq_len = len(side_image)   # this is always 1 in __call__
        assert len(side_image) == len(wrist_image) == len(proprio), "Length of the sequence must be the same"
        if action is not None:   # action can be None in __call__; in prompting, there are episodes with action length less than the length of the observation
            assert len(action) + 1 == seq_len or len(action) == seq_len, "Length of the action sequence must be one less than or equal to the observation sequence"
        assert proprio.shape[0] == seq_len, "Length of proprio sequence must match the length of the observation sequence"
        
        # construct input dictionary 
        side_image = torch.cat([preprocess(img)[None] for img in side_image], dim=0)    # (seq_len, 3, 224, 224), (1, 3, 224, 224) in __call__
        wrist_image = torch.cat([preprocess(img)[None] for img in wrist_image], dim=0)    # (seq_len, 3, 224, 224), (1, 3, 224, 224) in __call__
        # interleave the images, side_image and wrist_image are interleaved along the dimension 1
        image_vec = torch.stack([side_image, wrist_image], dim=1)[None].float() # stack along a new axis (seq_len, 2, 3, H, W), then add batch dim -> (1, seq_len, 2, 3, 224, 224)
        # process proprio 
        rot = proprio[:, 3:-2] # -2 because we have 2 values for gripper_states in LIBERO Object
        if self.args.shared_cfg.rot_6d:
            rot = euler_to_rot_6d(rot)
        proprio = np.concatenate([proprio[:, :3], rot, proprio[:, -2:]], axis=-1)   # -2 because we have 2 values for gripper_states in LIBERO Object
        proprio = torch.tensor(proprio)[None].float()

        # process action, doing conversion stuffs here
        if action is not None:  # this is always True in prompting, but can be False in __call__, where action can be None
            rot = action[:, 3:-1]
            if self.args.shared_cfg.rot_6d: # True
                rot = euler_to_rot_6d(rot)
            action = np.concatenate([action[:, :3], rot, action[:, -1:]], axis=-1)
            # if self.args.shared_cfg.use_delta_action:
            #     action = convert_delta_action(action[:, None, ...], proprio.numpy().transpose(1, 0, 2)).transpose(1, 0, 2)
            #     action = torch.tensor(action).float()
            # else:
            action = torch.tensor(action)[None].float()
        # action shape now is (1, seq_len, 10)
        # NOTE: here we didn't implement 1) support for eos prediction 2) support for euler outputs 

        if visual_trace is not None:
            visual_trace = torch.tensor(visual_trace)
            visual_trace = visual_trace[None].float() # (1, seq_len, 32)

        obs = {
            "observation": image_vec, # (1, seq_len, 2, 3, 224, 224)
            "proprio": proprio, # (1, seq_len, 10)
            "visual_trace": visual_trace, # is None in __call__
            "action": action, # is None in __call__
        }
        return obs

    def prepare_batch_observations(
        self, 
        side_image: List[Union[PIL.Image.Image, List[PIL.Image.Image]]], 
        wrist_image : List[Union[PIL.Image.Image, List[PIL.Image.Image]]], 
        proprio : Union[np.ndarray],    # (batch_size, seq_len, proprio_dim) 
        visual_trace : Union[np.ndarray],    # (batch_size, seq_len, num_visual_trace_points * 2)
        action : Optional[np.ndarray] = None    # (batch_size, seq_len, action_dim)
    ):
        """
        assume the observation is a dictionary with the following keys:
        "observation", "proprio", "action"
        we also have visual_trace
        proprio in xyzXYZGripper format
        action in xyzXYZGripper format
        """
        # Following is just the preprocessing of side and wrist images
        if isinstance(side_image[0], PIL.Image.Image): # this happens in __call__, when there is only one side image
            side_image = [[img] for img in side_image]  # list of 1-element lists
            preprocess = self.preprocess_PIL
        elif isinstance(side_image[0], np.ndarray):
            pass    # check later
        else:   # now, this is the case
            preprocess = self.preprocess_PIL

        if isinstance(wrist_image[0], PIL.Image.Image): # this happens in __call__, when there is only one wrist image
            wrist_image = [[img] for img in wrist_image]  # list of 1-element lists
        elif isinstance(wrist_image[0], np.ndarray): # never happens
            pass    # check later

        # find the length of the sequence, which is the length of side_image, wrist_image and proprio
        seq_len = len(side_image[0])   # this is always 1 in __call__
        assert len(side_image[0]) == len(wrist_image[0]) == proprio.shape[1], "Length of the sequence must be the same"
        if action is not None:   # action can be None in __call__; in prompting, there are episodes with action length less than the length of the observation
            assert action.shape[1] + 1 == seq_len or action.shape[1] == seq_len, "Length of the action sequence must be one less than or equal to the observation sequence"
        assert proprio.shape[1] == seq_len, "Length of proprio sequence must match the length of the observation sequence"
        
        # construct input dictionary 
        batch_image_vec = torch.cat([torch.stack([torch.cat([preprocess(img)[None] for img in side_img], dim = 0),
                                                torch.cat([preprocess(img)[None] for img in wrist_img], dim = 0)], dim=1)[None].float()
                                                for side_img, wrist_img in zip(side_image, wrist_image)], dim=0)
        # process proprio 
        rot = proprio[:, :, 3:-2] # -2 because we have 2 values for gripper_states in LIBERO Object
        if self.args.shared_cfg.rot_6d:
            rot = euler_to_rot_6d(rot)
        proprio = np.concatenate([proprio[:, :, :3], rot, proprio[:, :, -2:]], axis=-1)   # -2 because we have 2 values for gripper_states in LIBERO Object
        proprio = torch.tensor(proprio).float()

        # process action, doing conversion stuffs here
        if action is not None:  # this is always True in prompting, but can be False in __call__, where action can be None
            rot = action[:, :, 3:-1]
            if self.args.shared_cfg.rot_6d: # True
                rot = euler_to_rot_6d(rot)
            action = np.concatenate([action[:, :, :3], rot, action[:, :, -1:]], axis=-1)
            # if self.args.shared_cfg.use_delta_action:
            #     action = convert_delta_action(action[:, None, ...], proprio.numpy().transpose(1, 0, 2)).transpose(1, 0, 2)
            #     action = torch.tensor(action).float()
            # else:
            action = torch.tensor(action).float()
        # action shape now is (batch_size, seq_len, 10)
        # NOTE: here we didn't implement 1) support for eos prediction 2) support for euler outputs 

        if visual_trace is not None:
            visual_trace = torch.tensor(visual_trace)
            visual_trace = visual_trace.float() # (batch_size, seq_len, 10)

        obs = {
            "observation": batch_image_vec, # (batch_size, seq_len, 2, 3, 224, 224)
            "proprio": proprio, # (batch_size, seq_len, 10)
            "visual_trace": visual_trace, # is None in __call__
            "action": action, # can be None in __call__
        }
        return obs

    def __call__(
        self, 
        side_image: Union[PIL.Image.Image, List[PIL.Image.Image]], 
        wrist_image : Union[PIL.Image.Image, List[PIL.Image.Image]], 
        proprio : Union[np.ndarray], 
        visual_trace : Union[np.ndarray],
        action : Optional[np.ndarray] = None,
        abs_gripper_control=False, # ignore temporal essembling for gripper control
        binary_gripper=False, # discretize the gripper control 
        use_temporal=True, # temporal essembling 
        teacher_forcing=False, # use ground truth action for prediction
        pred_visual_trace=True, # predict the visual trace before predicting the action
    ):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            side_image (PIL.Image.Image or List[PIL.Image.Image]): side camera image
            wrist_image (PIL.Image.Image or List[PIL.Image.Image]): wrist camera image
            proprio (np.ndarray): proprioceptive information
            visual_trace (np.ndarray): visual trace
            action (np.ndarray): action information
        """ 
        obs = self.prepare_observations(side_image, wrist_image, proprio, visual_trace, action)   # first important function
        for k, v in obs.items():
            if v is not None:
                obs[k] = v.to(self.device, non_blocking=True)
        if teacher_forcing:
            action, visual_trace = self.model.get_action_eval(   # second important function
                obs,
                abs_gripper_control=abs_gripper_control,
                binary_gripper=binary_gripper,
                use_temporal=use_temporal,
                teacher_forcing=teacher_forcing,
            )
        else:
            action, visual_trace = self.model.get_action_eval_no_teacher_forcing(
                obs,
                abs_gripper_control=abs_gripper_control,
                binary_gripper=binary_gripper,
                use_temporal=use_temporal,
                pred_visual_trace=pred_visual_trace,
            )
        if self.args.shared_cfg.rot_6d:
            rotation = torch.tensor(rot_6d_to_euler(action[3:-1].cpu().numpy())).to(action.device).squeeze()
            action = torch.cat([action[:3], rotation, action[-1:]])
        action = action.cpu().numpy()
        visual_trace = visual_trace.cpu().numpy()
        
        return action, visual_trace

    def batch_call(
        self, 
        side_images: List[PIL.Image.Image],
        wrist_images : List[PIL.Image.Image],
        proprios : Union[np.ndarray],
        visual_traces : Union[np.ndarray],
        actions : Optional[np.ndarray] = None,
        abs_gripper_control=False, # ignore temporal essembling for gripper control
        binary_gripper=False, # discretize the gripper control 
        use_temporal=True, # temporal essembling 
        teacher_forcing=False, # use ground truth action for prediction
        pred_visual_trace=True, # predict the visual trace before predicting the action
    ):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            side_image (PIL.Image.Image or List[PIL.Image.Image]): side camera image
            wrist_image (PIL.Image.Image or List[PIL.Image.Image]): wrist camera image
            proprio (np.ndarray): proprioceptive information
            visual_trace (np.ndarray): visual trace
            action (np.ndarray): action information
        """ 
        obs = self.prepare_batch_observations(side_images, wrist_images, proprios, visual_traces, actions)   # first important function
        for k, v in obs.items():
            if v is not None:
                obs[k] = v.to(self.device, non_blocking=True)
        if teacher_forcing:
            pass    # don't care now
        else:
            actions, visual_traces = self.model.get_batch_action_eval_no_teacher_forcing(
                obs,
                abs_gripper_control=abs_gripper_control,
                binary_gripper=binary_gripper,
                use_temporal=use_temporal,
                pred_visual_trace=pred_visual_trace,
            )
        if self.args.shared_cfg.rot_6d:
            rotations = torch.tensor(rot_6d_to_euler(actions[:, 3:-1].cpu().numpy())).to(actions.device)
            actions = torch.cat([actions[:, :3], rotations, actions[:, -1:]], dim=-1)
        actions = actions.cpu().numpy()
        visual_traces = visual_traces.cpu().numpy()
        return actions, visual_traces
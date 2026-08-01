# ICLR: In-Context Imitation Learning with Visual Reasoning


This repo contains the unofficial implementation for *ICLR: In-Context Imitation Learning with Visual Reasoning*, which is accepted to the IROS 2026 conference.


## Setup
```bash
# download repo
git clone --recurse-submodules https://github.com/toannguyen1904/ICLR.git
cd ICLR

# create the environment and install packaged with uv
uv python pin 3.10
uv sync
```

Verify the GPU is visible:
```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Preparing LIBERO datase
*Make sure you added the LIBERO submodule before running following steps*

### 1. LIBERO Configuration

First you should set the LIBERO default path and config file path to your current folder for convenient. Make the following edits in the [LIBERO/libero/libero/\_\_init__.py](LIBERO/libero/libero/__init__.py):
```python
libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.expanduser("your_ICLR_folder")
)
config_file = os.path.join(libero_config_path, "libero_config.yaml")
```

After that, create the LIBERO config file with the same name as in your edits:
```bash
touch ~/libero_config.yaml
```

Then you can edit your config file:
```yaml
assets: your_ICLR_folder/LIBERO/libero/libero/assets
bddl_files: your_ICLR_folder/LIBERO/libero/libero/bddl_files
benchmark_root: your_ICLR_folder/LIBERO/libero/libero
datasets: your_desired_LIBERO_datasets_location
init_states: your_ICLR_folder/LIBERO/libero/libero/init_files
```

### 2. Download LIBERO-Object and LIBERO-90
Run following commands to download two LIBERO datasets used in ICLR
```bash
cd LIBERO
# LIBERO-Object
python benchmark_scripts/download_libero_datasets.py --datasets libero_object
# LIBERO-90. Here we download LIBERO-100 since LIBERO-90 is a part of LIBERO-100
python benchmark_scripts/download_libero_datasets.py --datasets libero_100
```

After downloading successfully, you should see the confirmation that LIBERO-Object, LIBERO-90, and LIBERO-10 are complete. However, as mentioned earlier, we don't need LIBERO-10, so you can just go delete it.

### 3. Preprocessing LIBERO datasets
In this step, we regenerate the LIBERO datasets to remove no-op actions and unsuccessful episodes.
```bash
cd ../tools

```

## Checkpoints
We host the checkpoints on [🤗HuggingFace](https://huggingface.co/mlfu7/ICRT). Please follow the following instructions to download them.
```bash 
# install git-lfs
sudo apt install git-lfs
git lfs install
# cloning checkpoints
git clone git@hf.co:mlfu7/ICRT checkpoints
```

## Model Training 

Please refer to [TRAIN.md](TRAIN.md) for training the model.

## Model Inference

Please look at [inference.ipynb](tools/inference.ipynb) for examples on inferencing ICRT.

## License
This project is under the Apache 2.0 license. See [LICENSE](LICENSE.txt) for details.

## Citation 
Please give us a star 🌟 on Github to support us!

Please cite our work if you find our work inspiring or use our code in your work:
```
@article{fu2024icrt,
    title={In-Context Imitation Learning via Next-Token Prediction}, 
    author={Letian Fu and Huang Huang and Gaurav Datta and Lawrence Yunliang Chen and William Chung-Ho Panitch and Fangchen Liu and Hui Li and Ken Goldberg},
    journal={arXiv preprint arXiv:2408.15980},
    year={2024}
}
```
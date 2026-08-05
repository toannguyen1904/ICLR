# ICLR: In-Context Imitation Learning with Visual Reasoning


This repo contains the unofficial implementation for *ICLR: In-Context Imitation Learning with Visual Reasoning*, which is accepted to the IROS 2026 conference.


## Setup
```bash
# download repo
git clone --recurse-submodules https://github.com/toannguyen1904/ICLR.git
cd ICLR

# create the environment and install packages with uv
uv python pin 3.10
uv sync
# Install iclr package
uv pip install -e .
```

Verify the GPU is visible:
```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Download the Pretrained Vision Encoder
You can download the pretrained vision encoder (originally provided by [Fu et al. 2025](https://icrt.dev/)) by running:
```bash
hf download nanidafvck/ICLR_vision_encoder --local-dir ./vision_encoder/
```

## Preparing LIBERO Datasets
*Make sure you added the LIBERO submodule before running following steps*

### 1. LIBERO Configuration and Installation

First you should set the LIBERO default path and config file path to your current folder for convenient. Make the following edits in the [LIBERO/libero/libero/\_\_init__.py](LIBERO/libero/libero/__init__.py):
```python
libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.expanduser("your_ICLR_folder")
)
config_file = os.path.join(libero_config_path, "libero_config.yaml")
```

After that, create the LIBERO config file with the same name as in your edits:
```bash
touch libero_config.yaml
```

Then you can edit your config file:
```yaml
assets: your_ICLR_folder/LIBERO/libero/libero/assets
bddl_files: your_ICLR_folder/LIBERO/libero/libero/bddl_files
benchmark_root: your_ICLR_folder/LIBERO/libero/libero
datasets: your_desired_LIBERO_datasets_location
init_states: your_ICLR_folder/LIBERO/libero/libero/init_files
```

The upstream LIBERO repo is missing a couple of `__init__.py` files (a known, still-open bug, see
[PR #15](https://github.com/Lifelong-Robot-Learning/LIBERO/pull/15)), which causes `setup.py`'s
`find_packages()` to silently discover zero packages, so `import libero` fails even after a
successful-looking install. Add them before installing:
```bash
touch LIBERO/libero/__init__.py
touch LIBERO/libero/lifelong/models/modules/__init__.py
```

To install the libero package, run:
```bash
uv pip install -e ./LIBERO
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

### 3. Preprocess LIBERO Datasets
In this step, we regenerate the LIBERO datasets to remove no-op actions and unsuccessful episodes. Basically, we follow OpenVLA's preprocessing.
```bash
cd ../tools
# LIBERO-Object
python3 regenerate_libero_dataset.py --libero_task_suite libero_object --libero_raw_data_dir path/to/libero_object --libero_target_dir .../libero_object_new

# LIBERO-90
python3 regenerate_libero_dataset.py --libero_task_suite libero_90 --libero_raw_data_dir path/to/libero/90 --libero_target_dir .../libero_90_new
```

After this, please delete the original `libero_object` and `libero_90` data folders and rename `libero_object_new` to `libero_object` and `libero_90_new` to `libero_90`.

### 4. Generate Visual Traces for LIBERO Datasets
Note that for LIBERO, we don't use Molmo2 to generate visual traces, instead, we infer the gripper position via the robot’s proprioceptive state and the known camera parameters. In particular, run these scritps:
```bash
# LIBERO-Object
python iclr/data/visual_trace_process_libero.py --benchmark libero_object

# LIBERO-90
python iclr/data/visual_trace_process_libero.py --benchmark libero_90
```

After this, you should see `visual_trace_im256.pkl` files in the two dataset folders.

### 5. Generate Metadata for LIBERO datasets
Run following scripts to generate metadata for LIBERO datasets (epiode grouping, lengths, etc.)
```bash
# Make folders for LIBERO metadata
mkdir -p config/data_config_libero/libero_object
mkdir -p config/data_config_libero/libero_90

# LIBERO-Object metadata
python3 tools/gen_libero_metadata.py --libero_path path_to_your_LIBERO_data_folder --task_suite libero_object --root_path path_to_ICLR_folder

# LIBERO-Object metadata
python3 tools/gen_libero_metadata.py --libero_path path_to_your_LIBERO_data_folder --task_suite libero_90 --root_path path_to_ICLR_folder
```

### 6. Data Configuration for ICLR
Change `/data/tientoan/LIBERO` to your folder that stores the LIBERO datasets in [config/dataset_config_libero_object_visual_trace.json](config/dataset_config_libero_object_visual_trace.json) and [config/dataset_config_libero_90_visual_trace.json](config/dataset_config_libero_90_visual_trace.json)
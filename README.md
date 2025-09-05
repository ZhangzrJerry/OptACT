# OptACT

## Installation

Create and activate a conda environment:

```bash
conda create -n aloha python=3.8.10
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
cd act/detr && pip install -e .
```

Set up the MUJOCO rendering backend:

```sh
export MUJOCO_GL=egl
```

## Usage

Record simulation episodes:

```sh
python record_sim_episodes.py \
    --dataset_dir data \
    --num_episodes 50
```

Visualize recorded episodes:

```sh
python3 visualize_episodes.py \
    --dataset_dir data \
    --episode_idx 0
```

Train the ACT policy:

```sh
python3 imitate_episodes.py \
    --ckpt_dir ckpt \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0
```

Evaluate a trained model with temporal aggregation:

```sh
python3 imitate_episodes.py \
    --ckpt_dir ckpt --policy_class ACT \
    --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --batch_size 8 \
    --dim_feedforward 3200 --num_epochs 500 \
    --lr 1e-5 --seed 0 --temporal_agg --eval
```

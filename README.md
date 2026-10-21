# STEMCrysNet: **A Retrieval-Generative Framework for 3D Crystal Structure Solution from Atomic-Resolution STEM Images**

STEMCrysNet is a deep learning framework for identifying known crystal structures and generating three-dimensional (3D) crystal structures from atomic-resolution high-angle annular dark-field scanning transmission electron microscopy (HAADF-STEM) images.

## Overview

STEMCrysNet consists of three main components:

- **STEMEncoder:** A modified ResNet-based encoder that extracts structural representations from single-view or dual-view HAADF-STEM images.
- **CrysEncoder:** A graph neural network based on CSPNet that encodes three-dimensional crystal structures.
- **CrysGenerator:** A conditional flow-matching generative model that produces three-dimensional crystal structures from STEM-derived representations and chemical constraints.

STEMEncoder and CrysEncoder are trained through contrastive learning to align image and crystal representations in a shared latent space.

STEMCrysNet supports two complementary tasks:

- **Crystal structure retrieval:** Identifies the most similar structures from a reference crystal database using learned image–structure representations.
- **Crystal structure generation:** Generates candidate crystal structures without requiring the target structure to be present in the reference database. Candidate structures are subsequently ranked using their predicted potential energies calculated by MatterSim.


## Installation

The codebase is developed with Python 3.10 and CUDA-enabled PyTorch. A Linux environment with NVIDIA GPUs is recommended for training and batched inference.

Clone the repository:

```bash
git clone git@github.com:meng-yuan321/STEMCrysNet.git
cd STEMCrysNet
```

Create and activate the recommended Python environment:

```bash
conda create -n stemcrysnet python=3.10 -y
conda activate stemcrysnet
```

Install the required dependencies:

```bash
pip install -r requirements.txt
pip install --no-deps mattersim  
```


## Dataset

Large files in this repository (the MP50 CSV and benchmark LMDB databases) are tracked with [Git LFS](https://git-lfs.com/). Install Git LFS before or immediately after cloning, then fetch the data blobs:
- For linux
```bash
sudo apt install git-lfs 
```
- For macOS
```bash
brew install git-lfs
```
```bash
git lfs install
git lfs pull
```

For a new clone, install Git LFS first so large files are downloaded during `git clone`:

```bash
git lfs install
git clone git@github.com:meng-yuan321/STEMCrysNet.git
cd STEMCrysNet
```

Check that LFS objects are present locally:

```bash
git lfs ls-files
```

### MP50 Dataset

The MP50 dataset was constructed from the Materials Project database by selecting inorganic crystal structures containing fewer than 50 atoms in their conventional unit cells.

- **Full dataset:** `data/mp_50/mp_50_7system_20250710.csv` (all MP50 structures with metadata and CIFs).
- **Test set:** `data/mp_50/benchmark_1000.xlsx` (held-out benchmark structures), `data/mp_50/lmdb_4_multi_axes/test1000.lmdb`.
- **Train / validation:** After excluding the test-set structures from the full catalog, the remainder is split **95:5** into training and validation sets.

The CIFs and their corresponding simulated HAADF-STEM images are packaged as LMDB databases for training and evaluation. `train.lmdb` and `val.lmdb` files are not included in this repository; please contact the authors if you need access.

### OQMD and MatterGen benchmark

200 OQMD structures and 100 MatterGen structures were used for external evaluation of generalization ability of STEMCrysNet

- The paths are `data/oqmd_benchmark/lmdb/test.lmdb` and `data/mattergen_benchmark/lmdb/test.lmdb`, respectively.


## Configuration

Create a `.env` file in the repository root and define the paths used by Hydra and logging:

```bash
PROJECT_ROOT=/absolute/path/to/STEMCrysNet
HYDRA_JOBS=/absolute/path/to/save/hydra/outputs
WANDB_DIR=/absolute/path/to/save/wandb/outputs
```

For full Hydra error traces, export:

```bash
export HYDRA_FULL_ERROR=1
```

## Training
for single-view:
```bash
python stemcrysent/run.py data=mp_50_noise \
    expname=stemcrysnet_v1 \
    model=uni3_flow \
    data.two_stem=False \
```

for dual-view:
```bash
python stemcrysent/run.py data=mp_50_noise \
    expname=stemcrysnet_v1 \
    model=uni3_flow \
    data.two_stem=True \
```


## Evaluation
### Retrieval

First generate model predictions:
```bash
python scripts/evaluate_contrastive.py \
    --ckpt_path <ckpt_path> \
    --save_path <save_path>
```

Then compute retrieval metrics:

```bash
python scripts/structure_retrieve.py \
    --atom_path <results_path> \
    --option 0 \
    && python scripts/structure_retrieve.py \
    --atom_path <results_path> \
    --option 1 \
    && python scripts/structure_retrieve.py \
    --atom_path <results_path> \
    --option 2 \
```

Retrieval options:

- `0`: Elemental Query (match by element types).
- `1`: Stoichiometric Query (match by element types and composition ratio).
- `2`: Unit Cell Content Query (match by element types, composition ratio, and atom count).
- `3`: Unconstrained Query (unconstrained retrieval against all structures in the database).


### Generation
Generate samples from a trained model:
```bash
torchrun --nproc_per_node=4 scripts/evaluate_generator.py \
    --generator_ckpt_path <ckpt_path> \
    --save_path <save_path> \
    --num_evals 20 \
    --test_path <test_lmdb_path> \
    --infer_timesteps 200
```

`--nproc_per_node` controls the number of GPUs used for inference. More GPUs usually reduce inference time.

Compute generation metrics:

```bash
python scripts/compute_generator_metrics.py \
    --root_path <results_path> \
    --save_path <save_path> \
    --multi_eval \
    --strict \
    --sort \
    --sort_mode mattersim
```

The workflow used for experimental data:

1. HAADF-image aquisition
2. Image rectification (needed for dual-view generation)
3. Resizing to 0.15Å/pixel and cropping to 256*256
4. Denoising when needed
5. STEMCrysNet inference


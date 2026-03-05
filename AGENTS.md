# Agent Guidelines: 1HP_NN_equivariance

This repository focuses on 1st stage Neural Network models (1HP-NN) for simulation data, with a specific focus on rotational equivariance.

## 🛠 Build, Lint, and Test Commands

### Environment Setup
- **Install dependencies:** `pip install -r requirements.txt`
- **GPU support:** Ensure CUDA is installed. Set device via `export CUDA_VISIBLE_DEVICES=<id>`.

### Running the Application
- **Train 1st stage model:** `python main.py --dataset_raw <NAME> --problem 2stages`
- **Inference/Test:** `python main.py --dataset_raw <NAME> --case test --model <PATH_TO_MODEL> --problem 2stages`
- **Hyperparameter Tuning:** `python main_hyperparam.py --dataset_raw <NAME> --problem 2stages`
- **Tensorboard:** `tensorboard --logdir=runs/ --host localhost --port 8088`

### Testing
- **Run all tests:** `python -m unittest discover unittests`
- **Run a single test file:** `python unittests/test_rotations.py` (or `python -m unittest unittests/test_rotations.py`)
- **Run a specific test case:** `python -m unittest unittests.test_rotations.TestRotation.test_rotation_logic` (if structured as classes)

## 📏 Code Style Guidelines

### 📦 Imports
- Order: Standard library, third-party libraries (torch, numpy, etc.), local modules.
- Local imports should be absolute from the project root (e.g., `from networks.unet import UNet`).

### 🛠 Formatting & Types
- **Indentation:** 4 spaces.
- **Typing:** Use Python type hints where possible, especially in function signatures (e.g., `def run(settings: SettingsTraining):`).
- **Docstrings:** Use sparingly for complex logic. Focus on "why" rather than "what".

### 🏷 Naming Conventions
- **Variables/Functions:** `snake_case` (e.g., `train_epoch_loss`, `init_data`).
- **Classes:** `PascalCase` (e.g., `SimulationDataset`, `Solver`).
- **Arguments:** Follow the `argparse` convention in `main.py` (e.g., `--augmentation_n`, `--equivariance_case`).

### ❌ Error Handling
- Use `try...except` blocks for non-critical failures in data loading or training loops to allow graceful degradation or logging.
- Manual interruption (Ctrl+C) is handled in `main.py` to save the current best model and LR history.

### 🧠 Model Architecture
- Models are located in `networks/`.
- Equivariant models: `G_UNet` (Equivariant CNN) and `Cont_G_UNet` (Continuous ECNN).
- Base models: `UNet`, `UNetHalfPad`.

## 🤖 Equivariance Specifics
- **Augmentation:** `--augmentation_n > 0` for uniform rotations, `< 0` for 90° increments.
- **Equivariance Cases:** `none`, `oriented_boxes` (alignment-based), `ecnn` (G-steerable CNN).
- **Scalar Fields:** Currently, only scalar fields like `pksi` are supported for equivariance operations.

## 📁 Repository Structure
- `data_stuff/`: Dataset and Transform classes.
- `networks/`: Neural network architectures (UNet, ECNN).
- `preprocessing/`: Scripts for data preparation.
- `processing/`: Core logic for training (`solver.py`) and rotations (`rotation.py`).
- `postprocessing/`: Visualization and measurement tools.
- `unittests/`: Unit tests for critical logic.

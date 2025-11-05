# MLOps Pipeline - Task Tracker

## Task Status Legend
- 🔲 Not Started
- 🔄 In Progress
- ✅ Completed
- ⏸️ Blocked

---

## Phase 1: Project Setup & Data (Week 1)

### Task 1: Project Initialization 🔄
**Status**: Ready to Start
**Estimated Time**: 2-3 hours
**Dependencies**: None

#### Objectives
Set up the foundational project structure, dependencies, and configuration files.

#### What You Need to Do

1. **Create Project Structure**
   ```
   distributed-mlops-pipeline/
   ├── config/
   ├── data/
   │   ├── raw/
   │   └── processed/
   ├── docker/
   ├── monitoring/
   ├── notebooks/
   ├── scripts/
   ├── src/
   │   ├── data/
   │   ├── models/
   │   ├── serving/
   │   ├── training/
   │   └── utils/
   └── tests/
       ├── test_data/
       ├── test_models/
       └── test_serving/
   ```

2. **Create Dependencies File**

   Choose ONE of the following:

   **Option A: Using `requirements.txt`** (simpler)
   ```
   # Core ML
   torch>=2.0.0
   numpy>=1.24.0
   pandas>=2.0.0
   scikit-learn>=1.3.0

   # MLflow
   mlflow>=2.8.0

   # API
   fastapi>=0.104.0
   uvicorn[standard]>=0.24.0
   pydantic>=2.0.0

   # Monitoring
   prometheus-client>=0.19.0

   # Data visualization
   matplotlib>=3.7.0
   seaborn>=0.12.0

   # Utils
   pyyaml>=6.0
   python-dotenv>=1.0.0

   # Development
   pytest>=7.4.0
   ruff>=0.1.0
   mypy>=1.7.0
   jupyter>=1.0.0
   ipykernel>=6.25.0
   ```

   **Option B: Using Poetry** (more professional)
   ```bash
   poetry init
   # Follow prompts, then:
   poetry add torch numpy pandas scikit-learn mlflow fastapi uvicorn pydantic prometheus-client matplotlib seaborn pyyaml python-dotenv
   poetry add --group dev pytest ruff mypy jupyter ipykernel
   ```

3. **Create `.gitignore`**
   ```
   # Python
   __pycache__/
   *.py[cod]
   *$py.class
   *.so
   .Python
   env/
   venv/
   ENV/
   .venv

   # Data
   data/raw/*
   data/processed/*
   !data/raw/.gitkeep
   !data/processed/.gitkeep

   # Models
   models/
   mlruns/
   checkpoints/

   # Jupyter
   .ipynb_checkpoints

   # IDE
   .vscode/
   .idea/
   *.swp
   *.swo

   # OS
   .DS_Store
   Thumbs.db

   # Secrets
   .env
   *.pem
   *.key

   # Docker
   .dockerignore

   # Logs
   logs/
   *.log
   ```

4. **Create `.gitkeep` files** in empty directories
   ```bash
   touch data/raw/.gitkeep
   touch data/processed/.gitkeep
   ```

5. **Create Basic Configuration Files**

   **`config/model_config.yaml`**
   ```yaml
   # DeepLOB Model Configuration
   model:
     name: "deeplob"

     # Input dimensions
     input_shape:
       channels: 1
       height: 100  # LOB features (10 levels * 10 features)
       width: 100   # Time steps

     # Architecture
     conv_layers:
       - filters: 32
         kernel_size: [1, 2]
         stride: [1, 2]
       - filters: 32
         kernel_size: [4, 1]
         stride: [1, 1]
       - filters: 32
         kernel_size: [4, 1]
         stride: [1, 1]

     inception:
       filters: 32

     lstm:
       hidden_size: 64
       num_layers: 1

     # Output
     num_classes: 3  # Up, Down, Stationary
   ```

   **`config/training_config.yaml`**
   ```yaml
   # Training Configuration
   training:
     # Data
     batch_size: 32
     num_workers: 4

     # Optimization
     learning_rate: 0.001
     weight_decay: 0.0001
     epochs: 50

     # Scheduler
     scheduler:
       type: "StepLR"
       step_size: 10
       gamma: 0.5

     # Early stopping
     early_stopping:
       patience: 10
       min_delta: 0.001

     # Checkpointing
     checkpoint_dir: "checkpoints"
     save_every_n_epochs: 5

     # Device
     device: "cuda"  # or "cpu"
     seed: 42

   # MLflow
   mlflow:
     tracking_uri: "http://localhost:5000"
     experiment_name: "lob-midprice-prediction"
   ```

6. **Create Initial README.md**
   ```markdown
   # Distributed MLOps Pipeline - LOB Mid-Price Prediction

   MLOps pipeline for predicting Limit Order Book mid-price movements using DeepLOB.

   ## Project Status
   🔄 In Development - MVP Phase

   ## Quick Start

   ### Prerequisites
   - Python 3.10+
   - Docker & Docker Compose
   - 8GB+ RAM
   - (Optional) NVIDIA GPU with CUDA

   ### Installation

   1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd distributed-mlops-pipeline
   ```

   2. Install dependencies:
   ```bash
   # Using pip
   pip install -r requirements.txt

   # OR using Poetry
   poetry install
   ```

   3. Download data (Task 2)

   ## Project Structure

   ```
   distributed-mlops-pipeline/
   ├── config/           # Configuration files
   ├── data/            # Data storage
   ├── docker/          # Docker configurations
   ├── src/             # Source code
   ├── scripts/         # Utility scripts
   ├── tests/           # Tests
   └── notebooks/       # Jupyter notebooks
   ```

   ## Documentation

   - [System Design](SYSTEM_DESIGN.md)
   - [MVP Plan](MVP_PLAN.md)
   - [Tasks](TASKS.md)

   ## License
   MIT
   ```

7. **Create `__init__.py` files**
   ```bash
   touch src/__init__.py
   touch src/data/__init__.py
   touch src/models/__init__.py
   touch src/training/__init__.py
   touch src/serving/__init__.py
   touch src/utils/__init__.py
   touch tests/__init__.py
   ```

8. **Test your setup**
   ```bash
   # Verify Python packages install
   pip install -r requirements.txt

   # Verify you can import torch
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

#### Deliverables Checklist

- [ ] All directories created
- [ ] `requirements.txt` or `pyproject.toml` created
- [ ] `.gitignore` configured
- [ ] Configuration files created (`model_config.yaml`, `training_config.yaml`)
- [ ] README.md created
- [ ] All `__init__.py` files created
- [ ] Dependencies install successfully
- [ ] Can import PyTorch

#### Evaluation Criteria

I will check:
1. ✅ Project structure matches the specification
2. ✅ All dependencies are specified with versions
3. ✅ `.gitignore` is comprehensive
4. ✅ Configuration files are valid YAML
5. ✅ README is clear and informative
6. ✅ Can run `pip install -r requirements.txt` without errors

#### Tips

- **Use exact versions** in requirements.txt to ensure reproducibility
- **Keep it simple** - don't add unnecessary dependencies yet
- **Test as you go** - verify each step works before moving on
- **Ask questions** if you're unsure about any step

#### Resources

- [Python Project Structure Best Practices](https://docs.python-guide.org/writing/structure/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [Poetry Documentation](https://python-poetry.org/docs/)

---

## Submission

When you're done:

1. **Commit your changes**
   ```bash
   git add .
   git commit -m "Task 1: Initialize project structure and dependencies"
   git push
   ```

2. **Share**:
   - Screenshot of your project structure
   - Output of `pip list` showing installed packages
   - Any issues you encountered

3. **Ready for review**: Let me know you're done!

---

### Task 2: Data Download & Exploration 🔲
**Status**: Not Started
**Estimated Time**: 3-4 hours
**Dependencies**: Task 1

(Details will be provided after Task 1 is completed)

---

### Task 3: Data Preprocessing Pipeline 🔲
**Status**: Not Started
**Estimated Time**: 4-5 hours
**Dependencies**: Task 2

(Details will be provided after Task 2 is completed)

---

## Phase 2: Model Development (Week 2)

### Task 4: DeepLOB Model Implementation 🔲
### Task 5: Training Pipeline with MLflow 🔲
### Task 6: Model Evaluation 🔲

(Details TBD)

---

## Phase 3: Infrastructure Setup (Week 3)

### Task 7: Docker Compose Services 🔲
### Task 8: Model Serving API 🔲
### Task 9: Monitoring Setup 🔲

(Details TBD)

---

## Phase 4: Polish & Documentation (Week 4)

### Task 10: Testing & CI 🔲
### Task 11: Documentation & README 🔲
### Task 12: End-to-End Demo 🔲

(Details TBD)

---

## Progress Tracking

- **Current Phase**: Phase 1 - Project Setup & Data
- **Current Task**: Task 1 - Project Initialization
- **Completion**: 0/12 tasks (0%)

---

## Notes

Use this space to track any:
- Challenges encountered
- Questions to ask
- Ideas for improvements
- Resources found helpful

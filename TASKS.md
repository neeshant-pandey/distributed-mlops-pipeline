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

2. **Create Dependencies File Using `uv`** ⭐ **RECOMMENDED**

   `uv` is the modern, blazingly fast Python package manager from Astral (creators of Ruff). It's what cutting-edge teams like Dagster use.

   **Step 1: Install uv**
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or with pip (if you already have Python)
   pip install uv

   # Verify installation
   uv --version
   ```

   **Step 2: Initialize project**
   ```bash
   # Create virtual environment
   uv venv

   # Activate it
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

   **Step 3: Create `pyproject.toml`**
   ```toml
   [project]
   name = "distributed-mlops-pipeline"
   version = "0.1.0"
   description = "MLOps pipeline for Limit Order Book mid-price prediction"
   requires-python = ">=3.10"
   dependencies = [
       "torch>=2.0.0",
       "numpy>=1.24.0",
       "pandas>=2.0.0",
       "scikit-learn>=1.3.0",
       "mlflow>=2.8.0",
       "fastapi>=0.104.0",
       "uvicorn[standard]>=0.24.0",
       "pydantic>=2.0.0",
       "prometheus-client>=0.19.0",
       "matplotlib>=3.7.0",
       "seaborn>=0.12.0",
       "pyyaml>=6.0",
       "python-dotenv>=1.0.0",
   ]

   [project.optional-dependencies]
   dev = [
       "pytest>=7.4.0",
       "ruff>=0.1.0",
       "mypy>=1.7.0",
       "jupyter>=1.0.0",
       "ipykernel>=6.25.0",
   ]

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.ruff]
   line-length = 100
   target-version = "py310"

   [tool.ruff.lint]
   select = ["E", "F", "I", "N", "W", "UP"]
   ignore = []

   [tool.mypy]
   python_version = "3.10"
   warn_return_any = true
   warn_unused_configs = true
   disallow_untyped_defs = false
   ```

   **Step 4: Install dependencies**
   ```bash
   # Install all dependencies (creates uv.lock file automatically)
   uv pip install -e ".[dev]"

   # Or install individually
   uv pip install torch numpy pandas scikit-learn mlflow fastapi uvicorn pydantic prometheus-client matplotlib seaborn pyyaml python-dotenv
   uv pip install pytest ruff mypy jupyter ipykernel
   ```

   **Why uv?**
   - ⚡ 10-100x faster than pip
   - 🔒 Automatic lock file (`uv.lock`) for reproducibility
   - 🎯 Drop-in replacement for pip commands
   - 🏢 Used by Dagster, FastAPI, and modern Python projects
   - 🚀 Shows cutting-edge knowledge on your portfolio

   ---

   **Alternative (if you prefer traditional approach):**

   **Using `requirements.txt`**
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

   Then: `pip install -r requirements.txt`

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

   # uv
   uv.lock

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
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"

   # OR using pip
   pip install -r requirements.txt
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
   # If using uv (recommended):
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"

   # OR if using pip:
   pip install -r requirements.txt

   # Verify you can import torch
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

   # Verify uv and other tools work
   uv --version
   ruff --version
   pytest --version
   ```

#### Deliverables Checklist

- [ ] All directories created
- [ ] `pyproject.toml` created (recommended: uv) OR `requirements.txt`
- [ ] `.gitignore` configured (including uv.lock if using uv)
- [ ] Configuration files created (`model_config.yaml`, `training_config.yaml`)
- [ ] README.md created
- [ ] All `__init__.py` files created
- [ ] Virtual environment created (`.venv/`)
- [ ] Dependencies install successfully
- [ ] Can import PyTorch

#### Evaluation Criteria

I will check:
1. ✅ Project structure matches the specification
2. ✅ All dependencies are specified with versions in `pyproject.toml` (or `requirements.txt`)
3. ✅ `.gitignore` is comprehensive (includes `.venv`, `uv.lock`)
4. ✅ Configuration files are valid YAML
5. ✅ README is clear and informative
6. ✅ Dependencies install successfully with `uv pip install -e ".[dev]"` (or pip)

#### Tips

- **Use exact versions** in requirements.txt to ensure reproducibility
- **Keep it simple** - don't add unnecessary dependencies yet
- **Test as you go** - verify each step works before moving on
- **Ask questions** if you're unsure about any step

#### Resources

- [uv Documentation](https://docs.astral.sh/uv/) ⭐ **Recommended**
- [Python Project Structure Best Practices](https://docs.python-guide.org/writing/structure/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [pyproject.toml Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

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
   - Output of `uv pip list` or `pip list` showing installed packages
   - Any issues you encountered

3. **Ready for review**: Let me know you're done!

---

### Task 2: Data Download & Exploration 🔄
**Status**: Ready to Start
**Estimated Time**: 3-4 hours
**Dependencies**: Task 1 ✅

#### Objectives
Download the FI-2010 dataset and perform exploratory data analysis to understand the data structure, distributions, and potential challenges.

#### What You Need to Do

1. **Download FI-2010 Dataset**

   The download script has been created for you at `src/data/download.py`.

   Run the download script:
   ```bash
   python src/data/download.py
   ```

   This will download 6 files to `data/raw/`:
   - **Train files**: `Train_Dst_NoAuction_DecPre_CF_7.txt`, `Train_Dst_NoAuction_DecPre_CF_8.txt`, `Train_Dst_NoAuction_DecPre_CF_9.txt`
   - **Test files**: `Test_Dst_NoAuction_DecPre_CF_7.txt`, `Test_Dst_NoAuction_DecPre_CF_8.txt`, `Test_Dst_NoAuction_DecPre_CF_9.txt`

   Verify the download:
   ```bash
   python src/data/download.py --verify
   ```

2. **Create Data Exploration Notebook**

   Create `notebooks/01_data_exploration.ipynb` and include the following analysis:

   **a) Load and Inspect Data**
   ```python
   from src.data.download import FI2010Downloader

   downloader = FI2010Downloader()
   X_train, y_train = downloader.load_data(split='train')
   X_test, y_test = downloader.load_data(split='test')

   print(f"Train features shape: {X_train.shape}")
   print(f"Train labels shape: {y_train.shape}")
   print(f"Test features shape: {X_test.shape}")
   print(f"Test labels shape: {y_test.shape}")
   ```

   **b) Understand Data Structure**
   - The data contains 144 features:
     - 40 price levels (20 ask prices + 20 bid prices)
     - 40 volume levels (20 ask volumes + 20 bid volumes)
     - 64 derived features
   - Labels have 5 columns for prediction horizons k=[1, 2, 3, 5, 10]
   - Label values: -1 (down), 0 (stationary), 1 (up)

   **c) Feature Distributions**
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns

   # Sample a few features and plot distributions
   fig, axes = plt.subplots(2, 3, figsize=(15, 8))
   for i, ax in enumerate(axes.flat):
       ax.hist(X_train[:, i*20], bins=50, alpha=0.7)
       ax.set_title(f'Feature {i*20}')
       ax.set_xlabel('Value')
       ax.set_ylabel('Frequency')
   plt.tight_layout()
   plt.show()

   # Check for outliers
   print("\nFeature statistics:")
   print(f"Min: {X_train.min()}")
   print(f"Max: {X_train.max()}")
   print(f"Mean: {X_train.mean():.4f}")
   print(f"Std: {X_train.std():.4f}")
   ```

   **d) Label Distribution Analysis**
   ```python
   import pandas as pd

   # Analyze label distribution for each horizon
   horizons = ['k=1', 'k=2', 'k=3', 'k=5', 'k=10']

   for i, horizon in enumerate(horizons):
       print(f"\n{horizon} Label Distribution:")
       unique, counts = np.unique(y_train[:, i], return_counts=True)
       for label, count in zip(unique, counts):
           percentage = count / len(y_train) * 100
           label_name = {-1: 'Down', 0: 'Stationary', 1: 'Up'}[label]
           print(f"  {label_name:12} ({label:2d}): {count:7d} ({percentage:5.2f}%)")

   # Visualize class distribution
   fig, axes = plt.subplots(1, 5, figsize=(20, 4))
   for i, (ax, horizon) in enumerate(zip(axes, horizons)):
       unique, counts = np.unique(y_train[:, i], return_counts=True)
       ax.bar(['Down', 'Stationary', 'Up'], counts)
       ax.set_title(f'{horizon}')
       ax.set_ylabel('Count')
   plt.suptitle('Label Distribution Across Prediction Horizons')
   plt.tight_layout()
   plt.show()
   ```

   **e) Sample Visualizations**

   **Order Book Heatmap:**
   ```python
   # Visualize a single LOB snapshot
   # Reshape first 100 features (10 levels * 10 features) into 2D
   sample_idx = 1000
   lob_snapshot = X_train[sample_idx, :100].reshape(10, 10)

   plt.figure(figsize=(10, 6))
   sns.heatmap(lob_snapshot, cmap='RdYlGn', center=0)
   plt.title('Limit Order Book Snapshot')
   plt.xlabel('Feature Index')
   plt.ylabel('Depth Level')
   plt.show()
   ```

   **Time Series Visualization:**
   ```python
   # Plot evolution of first few features over time
   window = 1000
   fig, axes = plt.subplots(3, 1, figsize=(15, 8))

   for i, ax in enumerate(axes):
       ax.plot(X_train[:window, i*10])
       ax.set_title(f'Feature {i*10} Time Series')
       ax.set_xlabel('Time Step')
       ax.set_ylabel('Value')
   plt.tight_layout()
   plt.show()
   ```

   **Correlation Analysis:**
   ```python
   # Sample features for correlation (all 144 would be too dense)
   sample_features = X_train[:5000, ::20]  # Every 20th feature
   corr_matrix = np.corrcoef(sample_features.T)

   plt.figure(figsize=(10, 8))
   sns.heatmap(corr_matrix, cmap='coolwarm', center=0,
               square=True, linewidths=0.5)
   plt.title('Feature Correlation Matrix (Sampled)')
   plt.tight_layout()
   plt.show()
   ```

3. **Document Your Findings**

   In markdown cells in your notebook, document:

   - **Data Quality Issues**: Any missing values, outliers, or anomalies?
   - **Class Imbalance**: Is there significant imbalance between Up/Down/Stationary?
   - **Feature Insights**: What patterns do you observe in the features?
   - **Challenges Identified**: What preprocessing challenges do you anticipate?
   - **Key Observations**: Any interesting patterns or surprising findings?

4. **Questions to Answer in Your Notebook**

   - What is the total number of training and test samples?
   - Are the features already normalized? (Check mean and std)
   - Which prediction horizon (k) has the most balanced classes?
   - Are there any temporal patterns in the data?
   - What is the data type and range of features?

#### Deliverables Checklist

- [ ] FI-2010 data downloaded to `data/raw/` (6 files)
- [ ] `notebooks/01_data_exploration.ipynb` created and runs without errors
- [ ] Data shape and structure documented
- [ ] Feature distributions visualized (histograms, time series)
- [ ] Label distribution analyzed for all 5 horizons
- [ ] Class imbalance identified and quantified
- [ ] Sample visualizations included:
  - [ ] Order book heatmap
  - [ ] Time series plots
  - [ ] Correlation matrix
- [ ] Key findings documented in markdown cells
- [ ] Data quality issues identified
- [ ] Preprocessing challenges noted

#### Evaluation Criteria

I will check:
1. ✅ All 6 data files successfully downloaded
2. ✅ Notebook runs end-to-end without errors
3. ✅ Clear understanding of data structure (144 features, 5 label horizons)
4. ✅ Thorough feature distribution analysis
5. ✅ Class imbalance quantified for each horizon
6. ✅ Quality visualizations that provide insights
7. ✅ Identified potential preprocessing needs (normalization, handling imbalance, etc.)
8. ✅ Documentation is clear and well-organized

#### Tips

- **Start simple**: Load a small subset of data first to test your code
- **Visualize iteratively**: Don't try to create all plots at once
- **Document as you go**: Write markdown cells explaining what you observe
- **Use the download script**: It has helper methods for loading data
- **Check data types**: Ensure features are float and labels are int
- **Sample the data**: For heavy visualizations, use a subset (e.g., first 5000 samples)

#### Resources

- **FI-2010 Dataset**: [Original Paper](https://arxiv.org/abs/1705.03233)
- **DeepLOB Paper**: [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
- **DeepLOB GitHub**: [Reference Implementation](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)
- **LOB Explanation**: [What is a Limit Order Book?](https://www.investopedia.com/terms/o/order-book.asp)

---

## Submission

When you're done:

1. **Commit your changes**
   ```bash
   git add data/raw/ notebooks/01_data_exploration.ipynb
   git commit -m "Task 2: Data download and exploration"
   git push
   ```

2. **Share**:
   - Your completed notebook
   - Key findings summary (3-5 bullet points)
   - Any challenges you faced

3. **Ready for review**: Let me know you're done!

---

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

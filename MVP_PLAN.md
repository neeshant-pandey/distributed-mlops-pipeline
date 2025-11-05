# MLOps Pipeline MVP - Limit Order Book Mid-Price Prediction

## 🎯 Project Goal
Build a **Simple but Production-Ready** MLOps pipeline for predicting mid-price movements from Limit Order Book data, demonstrating all key MLOps concepts without overwhelming complexity.

---

## 📊 Problem Statement

**Task**: Predict the direction of mid-price movement (Up/Down/Stationary) from Limit Order Book snapshots

**Use Case**: High-Frequency Trading signal generation

**Dataset**: FI-2010 (Finnish stock market data)
- 5 stocks (we'll start with 1)
- 10 days of tick data
- 10 levels of order book depth
- ~4.5 million data points per stock

**Prediction Horizons**: k=1, 2, 3, 5, 10 (we'll start with k=10)

---

## 🎨 MVP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ FI-2010 Dataset (downloaded locally)                     │   │
│  │ → data/raw/                                              │   │
│  │ → data/processed/                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Python Scripts (src/data/)                               │   │
│  │ - download.py: Get FI-2010 dataset                       │   │
│  │ - preprocess.py: Clean & normalize                       │   │
│  │ - dataset.py: PyTorch Dataset class                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ PyTorch + MLflow                                         │   │
│  │ - src/models/deeplob.py: Model architecture             │   │
│  │ - src/training/trainer.py: Training loop                │   │
│  │ - MLflow: Experiment tracking                           │   │
│  │ - Output: Trained model checkpoints                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Validation & Testing                                     │   │
│  │ - Accuracy, Precision, Recall, F1                        │   │
│  │ - Confusion Matrix                                       │   │
│  │ - Per-class metrics                                      │   │
│  │ - Inference latency                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL SERVING                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ FastAPI REST API                                         │   │
│  │ - POST /predict: Single prediction                       │   │
│  │ - POST /batch-predict: Batch predictions                │   │
│  │ - GET /health: Health check                             │   │
│  │ - GET /metrics: Prometheus metrics                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Docker Compose Services:                                 │   │
│  │ - MLflow Server (experiment tracking)                    │   │
│  │ - PostgreSQL (MLflow backend)                           │   │
│  │ - MinIO (model storage - S3 compatible)                 │   │
│  │ - Prometheus (metrics collection)                       │   │
│  │ - Grafana (dashboards)                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack (MVP)

### Core ML Stack
- **Language**: Python 3.10+
- **ML Framework**: PyTorch
- **Data Processing**: NumPy, Pandas
- **Experiment Tracking**: MLflow

### Serving Stack
- **API Framework**: FastAPI
- **Metrics**: Prometheus client
- **Serialization**: TorchScript or ONNX (optional optimization)

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **Storage**: MinIO (S3-compatible)
- **Database**: PostgreSQL (MLflow backend)
- **Monitoring**: Prometheus + Grafana

### Development Tools
- **Dependency Management**: Poetry or pip-tools
- **Code Quality**: Ruff (linting), MyPy (type checking)
- **Testing**: Pytest
- **Version Control**: Git

---

## 📁 Project Structure

```
distributed-mlops-pipeline/
├── README.md
├── SYSTEM_DESIGN.md
├── MVP_PLAN.md
├── pyproject.toml                 # Poetry dependencies
├── requirements.txt               # Or pip requirements
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # Basic CI (linting, tests)
│
├── docker/
│   ├── Dockerfile.training        # Training environment
│   ├── Dockerfile.serving         # Serving environment
│   └── docker-compose.yml         # All services
│
├── config/
│   ├── model_config.yaml          # Model hyperparameters
│   ├── training_config.yaml       # Training settings
│   └── serving_config.yaml        # API settings
│
├── data/
│   ├── raw/                       # Downloaded FI-2010 data
│   ├── processed/                 # Preprocessed data
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py            # Download FI-2010 dataset
│   │   ├── preprocess.py          # Data preprocessing
│   │   └── dataset.py             # PyTorch Dataset class
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deeplob.py             # DeepLOB architecture
│   │   └── base.py                # Base model class
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── callbacks.py           # Training callbacks
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── api.py                 # FastAPI application
│   │   ├── inference.py           # Inference logic
│   │   └── schemas.py             # Pydantic models
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Configuration loading
│       └── logging.py             # Logging setup
│
├── scripts/
│   ├── download_data.sh           # Download FI-2010
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── serve.py                   # Start API server
│
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   └── test_serving/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_results_analysis.ipynb
│
└── monitoring/
    ├── prometheus.yml
    └── grafana/
        └── dashboards/
```

---

## 🎯 MVP Feature Set

### ✅ What's Included

**Data Pipeline**
- [x] Download FI-2010 dataset
- [x] Basic preprocessing (normalization, train/val/test split)
- [x] PyTorch Dataset and DataLoader
- [x] Data quality checks (shape, null values, outliers)

**Model Training**
- [x] DeepLOB model implementation
- [x] Training loop with validation
- [x] MLflow experiment tracking (params, metrics, models)
- [x] Model checkpointing (best model, last model)
- [x] Early stopping
- [x] Learning rate scheduling

**Model Evaluation**
- [x] Multi-class classification metrics
- [x] Confusion matrix
- [x] Per-class performance
- [x] Inference latency measurement

**Model Serving**
- [x] FastAPI REST API
- [x] Model loading from MLflow
- [x] Single prediction endpoint
- [x] Batch prediction endpoint
- [x] Health checks
- [x] Prometheus metrics

**Infrastructure**
- [x] Docker Compose setup
- [x] MLflow server
- [x] PostgreSQL for MLflow backend
- [x] MinIO for artifact storage
- [x] Prometheus + Grafana

**Code Quality**
- [x] Type hints
- [x] Docstrings
- [x] Basic unit tests
- [x] Linting with Ruff

### ❌ What's NOT Included (Future Enhancements)

**Data Pipeline**
- [ ] Dagster orchestration
- [ ] Kafka streaming
- [ ] Advanced data quality (Great Expectations)
- [ ] Feature store

**Training**
- [ ] Distributed training (PyTorch DDP)
- [ ] Hyperparameter optimization (Optuna)
- [ ] Multi-stock parallel training
- [ ] Advanced model architectures (Transformers)

**Deployment**
- [ ] Kubernetes deployment
- [ ] CI/CD with automated deployment
- [ ] Canary deployments
- [ ] TorchServe (using FastAPI instead)

**Monitoring**
- [ ] Data drift detection
- [ ] Model performance monitoring
- [ ] Automated retraining triggers
- [ ] Alerting

---

## 📋 MVP Task Breakdown

### **Phase 1: Project Setup & Data (Week 1)**

#### Task 1: Project Initialization
**What you'll do:**
- Set up project structure
- Initialize Git repository
- Create `pyproject.toml` or `requirements.txt`
- Set up basic configuration files

**Deliverables:**
- [ ] Project structure matches layout above
- [ ] Dependencies file with all required packages
- [ ] `.gitignore` configured
- [ ] README with setup instructions

**Evaluation Criteria:**
- Clean, organized structure
- All dependencies specified with versions
- Can run `pip install -r requirements.txt` successfully

---

#### Task 2: Data Download & Exploration
**What you'll do:**
- Download FI-2010 dataset
- Explore the data structure
- Create a Jupyter notebook documenting your findings
- Implement `src/data/download.py`

**Deliverables:**
- [ ] FI-2010 data downloaded to `data/raw/`
- [ ] `notebooks/01_data_exploration.ipynb` with:
  - Data shape and structure
  - Feature distributions
  - Label distribution (class imbalance?)
  - Sample visualizations
- [ ] `src/data/download.py` script

**Evaluation Criteria:**
- Thorough data understanding
- Clear documentation
- Identified potential issues (imbalance, outliers, etc.)

---

#### Task 3: Data Preprocessing Pipeline
**What you'll do:**
- Implement normalization (z-score)
- Create train/val/test splits
- Handle any data quality issues
- Create PyTorch Dataset class

**Deliverables:**
- [ ] `src/data/preprocess.py` - preprocessing functions
- [ ] `src/data/dataset.py` - PyTorch Dataset class
- [ ] Processed data saved to `data/processed/`
- [ ] Unit tests for preprocessing

**Evaluation Criteria:**
- Correct normalization implementation
- Proper data splits (no leakage)
- Clean, reusable code
- Tests pass

---

### **Phase 2: Model Development (Week 2)**

#### Task 4: DeepLOB Model Implementation
**What you'll do:**
- Implement DeepLOB architecture (CNN + LSTM)
- Create model configuration
- Test forward pass

**Deliverables:**
- [ ] `src/models/deeplob.py` - model implementation
- [ ] `config/model_config.yaml` - hyperparameters
- [ ] Unit tests for model
- [ ] Documentation of architecture

**Evaluation Criteria:**
- Correct architecture (matches paper)
- Configurable hyperparameters
- Forward pass works
- Tests pass

---

#### Task 5: Training Pipeline with MLflow
**What you'll do:**
- Implement training loop
- Integrate MLflow tracking
- Add validation loop
- Implement checkpointing and early stopping

**Deliverables:**
- [ ] `src/training/trainer.py` - training logic
- [ ] `scripts/train.py` - training script
- [ ] MLflow tracking of metrics and parameters
- [ ] Model checkpointing

**Evaluation Criteria:**
- Training converges
- Metrics logged to MLflow
- Best model saved
- Can resume from checkpoint

---

#### Task 6: Model Evaluation
**What you'll do:**
- Implement evaluation metrics
- Create evaluation script
- Analyze model performance
- Document results

**Deliverables:**
- [ ] `src/training/metrics.py` - metric functions
- [ ] `scripts/evaluate.py` - evaluation script
- [ ] Confusion matrix visualization
- [ ] Performance report

**Evaluation Criteria:**
- Comprehensive metrics
- Clear performance analysis
- Identified strengths/weaknesses

---

### **Phase 3: Infrastructure Setup (Week 3)**

#### Task 7: Docker Compose Services
**What you'll do:**
- Create `docker-compose.yml`
- Set up MLflow server
- Set up PostgreSQL and MinIO
- Test end-to-end

**Deliverables:**
- [ ] `docker/docker-compose.yml`
- [ ] All services running
- [ ] MLflow accessible at http://localhost:5000
- [ ] Can log experiments to MLflow

**Evaluation Criteria:**
- All services start successfully
- Services can communicate
- Data persists across restarts

---

#### Task 8: Model Serving API
**What you'll do:**
- Implement FastAPI application
- Create prediction endpoints
- Add Prometheus metrics
- Dockerize the API

**Deliverables:**
- [ ] `src/serving/api.py` - FastAPI app
- [ ] `docker/Dockerfile.serving`
- [ ] API documentation (Swagger)
- [ ] Example requests

**Evaluation Criteria:**
- API returns correct predictions
- Low latency (<100ms)
- Prometheus metrics exposed
- Swagger docs clear

---

#### Task 9: Monitoring Setup
**What you'll do:**
- Configure Prometheus
- Create Grafana dashboards
- Set up basic alerts

**Deliverables:**
- [ ] `monitoring/prometheus.yml`
- [ ] Grafana dashboard JSON
- [ ] Dashboard shows key metrics

**Evaluation Criteria:**
- Metrics collected correctly
- Dashboard is informative
- Can track API performance

---

### **Phase 4: Polish & Documentation (Week 4)**

#### Task 10: Testing & CI
**What you'll do:**
- Write comprehensive tests
- Set up GitHub Actions CI
- Run linting and type checking

**Deliverables:**
- [ ] Tests for all modules
- [ ] `.github/workflows/ci.yml`
- [ ] CI passes

**Evaluation Criteria:**
- Test coverage >70%
- CI runs on push
- All checks pass

---

#### Task 11: Documentation & README
**What you'll do:**
- Write comprehensive README
- Add inline documentation
- Create architecture diagrams
- Write usage guide

**Deliverables:**
- [ ] Updated README.md
- [ ] Usage examples
- [ ] Troubleshooting guide

**Evaluation Criteria:**
- Someone can set up and run the project
- Clear, well-written
- All components documented

---

#### Task 12: End-to-End Demo
**What you'll do:**
- Create demo notebook
- Record training run
- Show serving API in action
- Performance analysis

**Deliverables:**
- [ ] Demo notebook or script
- [ ] Results summary
- [ ] Performance benchmarks

**Evaluation Criteria:**
- Complete end-to-end workflow
- Results are reproducible
- Clear demonstration

---

## 🎓 Learning Objectives

By completing this MVP, you will demonstrate:

### MLOps Skills
- ✅ Experiment tracking and versioning
- ✅ Model serving with REST APIs
- ✅ Containerization with Docker
- ✅ Monitoring and observability
- ✅ Infrastructure as code

### ML Engineering Skills
- ✅ PyTorch model implementation
- ✅ Data preprocessing pipelines
- ✅ Model evaluation and validation
- ✅ Production inference optimization

### Software Engineering Skills
- ✅ Clean code architecture
- ✅ Testing and CI/CD
- ✅ Documentation
- ✅ Version control

### Domain Knowledge
- ✅ Time series forecasting
- ✅ Financial data processing
- ✅ High-frequency trading concepts
- ✅ Limit order book mechanics

---

## 📊 Success Metrics

### Model Performance
- **Target Accuracy**: >60% (state-of-the-art on FI-2010 is ~65-78%)
- **Inference Latency**: <100ms for single prediction
- **Training Time**: <30 minutes for 50 epochs (on CPU)

### Engineering Quality
- **Test Coverage**: >70%
- **API Uptime**: 99%+
- **Documentation**: Complete and clear

### Learning Outcomes
- Can explain all components
- Can extend to new features
- Can deploy to production

---

## 🚀 Next Steps After MVP

Once MVP is complete, we'll enhance with:

1. **Dagster Orchestration**: Replace scripts with Dagster pipelines
2. **Distributed Training**: PyTorch DDP for multi-stock training
3. **Kubernetes Deployment**: Move from Docker Compose to K8s
4. **Kafka Integration**: Streaming data simulation
5. **Advanced Models**: Add Transformer architectures
6. **Drift Detection**: Monitor data and model drift
7. **Auto-Retraining**: Automated retraining triggers
8. **CI/CD**: Full deployment automation

---

## 📚 Resources

### Papers
- **DeepLOB**: [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
- **FI-2010 Dataset**: [Benchmark Dataset for Mid-Price Forecasting](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)
- **LOBFrame**: [GitHub Repo](https://github.com/FinancialComputingUCL/LOBFrame)

### Documentation
- [PyTorch Docs](https://pytorch.org/docs/)
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Docker Compose Docs](https://docs.docker.com/compose/)

### Tutorials
- [DeepLOB Official Implementation](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)
- [PyTorch Time Series Tutorial](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)

---

## ✅ Ready to Start?

Your **first task** is waiting in the next message! 🚀

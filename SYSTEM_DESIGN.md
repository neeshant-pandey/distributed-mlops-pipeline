# Distributed MLOps Pipeline - System Design

## Table of Contents
1. [High-Level System Design](#high-level-system-design)
2. [Low-Level Component Design](#low-level-component-design)
3. [Technology Stack](#technology-stack)
4. [Data Flow](#data-flow)
5. [Scalability & Performance](#scalability--performance)
6. [Monitoring & Observability](#monitoring--observability)

---

## High-Level System Design

### Overview
A production-grade distributed MLOps pipeline supporting the complete ML lifecycle from data ingestion through model deployment and monitoring, with automated retraining capabilities.

### Architecture Diagram (Conceptual)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                    │
│  (Databases, APIs, File Systems, Streaming Sources - Kafka)             │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Kafka        │  │ Batch Jobs   │  │ API          │                  │
│  │ Consumers    │  │              │  │ Connectors   │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ELT PIPELINE (Dagster)                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Data Quality Checks → Extract → Load → Transform                 │   │
│  │ - Schema validation  - Outlier detection  - Feature engineering  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  Storage: Data Lake (S3/MinIO) + Feature Store                          │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MLOPS PIPELINE (Dagster)                               │
│  ┌───────────────┐  ┌────────────────┐  ┌──────────────┐              │
│  │ Training      │→ │ Validation     │→ │ Registration │              │
│  │ - Distributed │  │ - Metrics eval │  │ - Model      │              │
│  │ - PyTorch DDP │  │ - A/B testing  │  │   versioning │              │
│  │ - Checkpoints │  │ - Champion/    │  │ - Metadata   │              │
│  │               │  │   Challenger   │  │              │              │
│  └───────────────┘  └────────────────┘  └──────────────┘              │
│  Orchestration: Dagster on Kubernetes                                   │
│  Experiment Tracking: MLflow                                            │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     CI/CD PIPELINE                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ GitHub Actions / GitLab CI                                       │   │
│  │ ├─ Code Quality (linting, type checking, testing)               │   │
│  │ ├─ Docker Image Build & Push                                    │   │
│  │ ├─ Model Testing (unit, integration, performance)               │   │
│  │ └─ Kubernetes Deployment (Helm/Kustomize)                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODEL SERVING LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ TorchServe   │  │ FastAPI      │  │ Load         │                  │
│  │ (PyTorch)    │  │ (Custom)     │  │ Balancer     │                  │
│  │              │  │              │  │              │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│  Deployment: Kubernetes with HPA (Horizontal Pod Autoscaling)           │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MONITORING & OBSERVABILITY                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Model Performance Monitoring                                     │   │
│  │ ├─ Prediction Logging (Kafka → Storage)                         │   │
│  │ ├─ Metrics Collection (Prometheus)                              │   │
│  │ ├─ Data Drift Detection (Evidently AI / custom)                 │   │
│  │ ├─ Model Drift Detection                                        │   │
│  │ └─ Alerting (trigger retraining)                                │   │
│  │                                                                  │   │
│  │ Infrastructure Monitoring                                        │   │
│  │ ├─ Prometheus + Grafana                                         │   │
│  │ └─ ELK Stack (logs)                                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **Data Ingestion Layer**
- **Purpose**: Collect data from multiple sources in batch and streaming modes
- **Technologies**: Kafka, Python ETL scripts
- **Key Features**:
  - Real-time streaming via Kafka
  - Batch ingestion for historical data
  - Schema validation at ingestion

#### 2. **ELT Pipeline (Dagster)**
- **Purpose**: Extract, Load, Transform data with quality checks
- **Technologies**: Dagster, Python, DuckDB/Spark for transformations
- **Key Features**:
  - Data quality checks (Great Expectations)
  - Feature engineering
  - Data versioning
  - Lineage tracking

#### 3. **MLOps Pipeline (Dagster)**
- **Purpose**: Automated model training, validation, and registration
- **Technologies**: Dagster, PyTorch, MLflow, Kubernetes
- **Key Features**:
  - Distributed training (PyTorch DDP, DeepSpeed)
  - Hyperparameter optimization
  - Model validation & A/B testing
  - Model registry with versioning

#### 4. **CI/CD Pipeline**
- **Purpose**: Automated testing and deployment
- **Technologies**: GitHub Actions, Docker, Kubernetes, Helm
- **Key Features**:
  - Code quality gates
  - Model testing (accuracy, latency, resource usage)
  - Containerization
  - Progressive rollouts (canary, blue-green)

#### 5. **Model Serving**
- **Purpose**: Serve predictions at scale
- **Technologies**: TorchServe, FastAPI, Kubernetes
- **Key Features**:
  - Auto-scaling based on load
  - A/B testing support
  - Model versioning in production
  - Low-latency inference

#### 6. **Monitoring & Observability**
- **Purpose**: Track model and system health
- **Technologies**: Prometheus, Grafana, Kafka, Evidently AI
- **Key Features**:
  - Data drift detection
  - Model performance tracking
  - Automated retraining triggers
  - Alert management

---

## Low-Level Component Design

### 1. Data Ingestion Service

#### Architecture
```python
# Component Structure
src/
├── ingestion/
│   ├── kafka_consumer.py       # Streaming data ingestion
│   ├── batch_loader.py         # Batch data loading
│   ├── schema_validator.py     # Pydantic models for validation
│   └── storage_writer.py       # Write to data lake
```

#### Key Classes & Responsibilities

**KafkaConsumer**
- Subscribe to configured topics
- Deserialize messages (Avro/JSON)
- Validate against schema
- Write to data lake (partitioned by date)
- Handle backpressure and failures

**BatchLoader**
- Support multiple source types (DB, S3, APIs)
- Incremental loading support
- Checkpointing for resumability
- Parallel loading for performance

**SchemaValidator**
- Pydantic models for type safety
- Custom validation rules
- Schema evolution handling
- Error reporting and logging

#### Data Flow
```
Kafka Topic → Consumer → Validation → Data Lake (S3/MinIO)
                                    ↓
                              Dead Letter Queue (failed records)
```

#### Configuration
```yaml
# config/ingestion.yaml
kafka:
  bootstrap_servers: ["kafka-1:9092", "kafka-2:9092"]
  topics: ["raw-events", "user-interactions"]
  consumer_group: "mlops-ingestion"
  auto_offset_reset: "earliest"

storage:
  type: "s3"  # or minio
  bucket: "mlops-data-lake"
  path_template: "raw/{topic}/{date}/{hour}/"
```

### 2. ELT Pipeline (Dagster)

#### Architecture
```python
src/
├── elt/
│   ├── assets/                 # Dagster assets
│   │   ├── data_quality.py     # Quality checks
│   │   ├── feature_engineering.py
│   │   └── data_versioning.py
│   ├── resources/              # Dagster resources
│   │   ├── data_lake.py
│   │   ├── feature_store.py
│   │   └── dq_engine.py
│   ├── sensors/                # Trigger on new data
│   └── schedules/              # Time-based triggers
```

#### Dagster Assets Design

**Data Quality Asset**
```python
@asset(
    partitions_def=DailyPartitionsDefinition(start_date="2024-01-01"),
    group_name="data_quality"
)
def validated_raw_data(
    context: AssetExecutionContext,
    raw_data: DataFrame,
    data_quality_engine: DataQualityEngine
) -> DataFrame:
    """
    Validates raw data against quality rules
    - Schema validation
    - Null checks
    - Range validation
    - Outlier detection
    """
    results = data_quality_engine.validate(raw_data)

    if results.failed:
        context.log.error(f"Quality checks failed: {results.errors}")
        # Store failed records for analysis
        # Optionally halt pipeline or continue with warnings

    return results.valid_data
```

**Feature Engineering Asset**
```python
@asset(
    deps=[validated_raw_data],
    group_name="features"
)
def engineered_features(
    context: AssetExecutionContext,
    validated_raw_data: DataFrame,
    feature_store: FeatureStore
) -> DataFrame:
    """
    Feature engineering transformations
    - Aggregations (rolling, windowed)
    - Encoding (categorical, embeddings)
    - Normalization/Scaling
    - Feature crosses
    """
    features = transform_features(validated_raw_data)

    # Version features in feature store
    feature_store.write_features(
        features,
        version=context.run.run_id,
        partition=context.partition_key
    )

    return features
```

#### Data Quality Framework
- **Great Expectations** integration
- Custom validators for domain-specific checks
- Automatic profiling and drift detection
- Configurable thresholds and actions

### 3. MLOps Training Pipeline (Dagster + PyTorch)

#### Architecture
```python
src/
├── training/
│   ├── assets/
│   │   ├── data_preparation.py
│   │   ├── model_training.py
│   │   ├── model_validation.py
│   │   └── model_registration.py
│   ├── distributed/
│   │   ├── ddp_trainer.py      # PyTorch DDP
│   │   ├── deepspeed_trainer.py
│   │   └── ray_trainer.py      # Alternative
│   ├── models/
│   │   ├── base_model.py
│   │   └── custom_models/
│   ├── config/
│   │   └── training_config.py  # Hydra configs
│   └── utils/
│       ├── checkpointing.py
│       └── metrics.py
```

#### Distributed Training Design

**PyTorch DDP Trainer**
```python
class DistributedTrainer:
    """
    Distributed training coordinator using PyTorch DDP
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_distributed()

    def setup_distributed(self):
        """Initialize distributed training environment"""
        torch.distributed.init_process_group(
            backend="nccl",  # GPU
            init_method="env://",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"])
        )

    def train(self, model, train_loader, val_loader):
        """
        Main training loop with:
        - Gradient accumulation
        - Mixed precision (AMP)
        - Gradient clipping
        - Learning rate scheduling
        - Checkpointing
        - MLflow logging
        """
        model = DDP(model, device_ids=[self.local_rank])
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch(model, train_loader, scaler)
            val_metrics = self.validate(model, val_loader)

            # Log to MLflow (only rank 0)
            if self.is_main_process():
                mlflow.log_metrics(
                    {**train_metrics, **val_metrics},
                    step=epoch
                )

            # Checkpointing
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(model, epoch, val_metrics)
```

**Kubernetes Training Job**
```yaml
# k8s/pytorch-training-job.yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: mlops-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: mlops/training:latest
            resources:
              limits:
                nvidia.com/gpu: 1
            env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow:5000"
    Worker:
      replicas: 3
      template:
        spec:
          containers:
          - name: pytorch
            image: mlops/training:latest
            resources:
              limits:
                nvidia.com/gpu: 1
```

#### Dagster Training Asset
```python
@asset(
    deps=[engineered_features],
    config_schema={
        "model_type": str,
        "num_workers": int,
        "gpu_per_worker": int
    }
)
def trained_model(
    context: AssetExecutionContext,
    k8s_client: KubernetesClient,
    mlflow_client: MlflowClient
) -> str:
    """
    Launch distributed training job on Kubernetes
    Returns: MLflow run_id
    """
    # Create training job
    job_spec = create_pytorch_job(
        model_type=context.op_config["model_type"],
        num_workers=context.op_config["num_workers"],
        feature_version=context.partition_key
    )

    # Submit to Kubernetes
    job = k8s_client.create_job(job_spec)

    # Wait for completion and monitor
    run_id = monitor_training_job(job, context.log)

    return run_id
```

### 4. Model Validation & Registration

#### Validation Framework
```python
class ModelValidator:
    """
    Comprehensive model validation
    """
    def validate(self, model, test_data, baseline_model=None):
        validations = {
            "accuracy_metrics": self.check_accuracy(model, test_data),
            "performance_metrics": self.check_performance(model),
            "fairness_metrics": self.check_fairness(model, test_data),
            "robustness": self.check_robustness(model, test_data)
        }

        # Champion/Challenger comparison
        if baseline_model:
            validations["comparison"] = self.compare_models(
                model, baseline_model, test_data
            )

        return ValidationReport(validations)

    def check_accuracy(self, model, test_data):
        """Standard ML metrics"""
        return {
            "accuracy": compute_accuracy(model, test_data),
            "precision": compute_precision(model, test_data),
            "recall": compute_recall(model, test_data),
            "f1": compute_f1(model, test_data),
            "auc_roc": compute_auc(model, test_data)
        }

    def check_performance(self, model):
        """Latency and resource metrics"""
        return {
            "inference_latency_p50": measure_latency(model, percentile=50),
            "inference_latency_p95": measure_latency(model, percentile=95),
            "inference_latency_p99": measure_latency(model, percentile=99),
            "memory_usage": measure_memory(model),
            "throughput": measure_throughput(model)
        }
```

#### Model Registry
```python
class ModelRegistry:
    """
    MLflow-based model registry with metadata
    """
    def register_model(
        self,
        model_uri: str,
        name: str,
        validation_report: ValidationReport,
        tags: Dict[str, str]
    ):
        """
        Register model with comprehensive metadata
        """
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=name,
            tags={
                **tags,
                "validation_passed": str(validation_report.passed),
                "accuracy": str(validation_report.metrics["accuracy"]),
                "latency_p95": str(validation_report.metrics["latency_p95"])
            }
        )

        # Store validation report
        self._store_validation_report(model_version, validation_report)

        return model_version

    def promote_to_production(
        self,
        model_name: str,
        version: str,
        deployment_strategy: str = "canary"
    ):
        """
        Promote model to production stage
        """
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=(deployment_strategy == "replace")
        )
```

### 5. CI/CD Pipeline

#### GitHub Actions Workflow
```yaml
# .github/workflows/ml-cicd.yaml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      - name: Lint with ruff
        run: ruff check src/
      - name: Type check with mypy
        run: mypy src/
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml

  model-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run model unit tests
        run: pytest tests/models/
      - name: Run model integration tests
        run: pytest tests/integration/
      - name: Performance benchmarks
        run: python scripts/benchmark_models.py

  build-and-push:
    needs: [code-quality, model-tests]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker build -t mlops/training:${{ github.sha }} -f docker/training.Dockerfile .
          docker build -t mlops/serving:${{ github.sha }} -f docker/serving.Dockerfile .
      - name: Push to registry
        run: |
          docker push mlops/training:${{ github.sha }}
          docker push mlops/serving:${{ github.sha }}

  deploy-staging:
    needs: build-and-push
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          kubectl apply -f k8s/staging/
          kubectl set image deployment/model-serving \
            model-serving=mlops/serving:${{ github.sha }}

  deploy-production:
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Canary deployment
        run: |
          # Deploy canary (10% traffic)
          kubectl apply -f k8s/production/canary.yaml
          # Monitor metrics for 10 minutes
          python scripts/monitor_canary.py --duration 600
          # If successful, proceed with full rollout
          kubectl apply -f k8s/production/
```

### 6. Model Serving Architecture

#### TorchServe Deployment
```python
# serving/torchserve_handler.py
class MLOpsHandler(BaseHandler):
    """
    Custom TorchServe handler with:
    - Preprocessing
    - Inference
    - Postprocessing
    - Logging predictions for monitoring
    """
    def __init__(self):
        super().__init__()
        self.kafka_producer = None

    def initialize(self, context):
        """Load model and initialize resources"""
        self.model = self.load_model(context)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Initialize Kafka for prediction logging
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        )

    def preprocess(self, data):
        """Transform input data"""
        # Apply same preprocessing as training
        return preprocess_input(data)

    def inference(self, data):
        """Run model inference"""
        with torch.no_grad():
            predictions = self.model(data)
        return predictions

    def postprocess(self, inference_output):
        """Format predictions and log"""
        predictions = format_predictions(inference_output)

        # Log predictions to Kafka for monitoring
        self.log_predictions(predictions)

        return predictions

    def log_predictions(self, predictions):
        """Send predictions to Kafka for drift monitoring"""
        self.kafka_producer.send(
            "model-predictions",
            value={
                "timestamp": datetime.utcnow().isoformat(),
                "model_version": self.model_version,
                "predictions": predictions,
                "metadata": self.get_metadata()
            }
        )
```

#### Kubernetes Serving Deployment
```yaml
# k8s/serving/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
        version: v1
    spec:
      containers:
      - name: torchserve
        image: mlops/serving:latest
        ports:
        - containerPort: 8080  # Inference
        - containerPort: 8081  # Management
        - containerPort: 8082  # Metrics
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_NAME
          value: "mlops-model"
        - name: MODEL_VERSION
          valueFrom:
            configMapKeyRef:
              name: model-config
              key: version
        livenessProbe:
          httpGet:
            path: /ping
            port: 8080
          initialDelaySeconds: 120
        readinessProbe:
          httpGet:
            path: /ping
            port: 8080
          initialDelaySeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: model-serving
spec:
  selector:
    app: model-serving
  ports:
  - port: 8080
    targetPort: 8080
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

### 7. Monitoring & Observability System

#### Architecture
```python
src/
├── monitoring/
│   ├── data_drift/
│   │   ├── detector.py
│   │   └── analyzers.py
│   ├── model_drift/
│   │   ├── performance_monitor.py
│   │   └── drift_detector.py
│   ├── metrics/
│   │   ├── collectors.py
│   │   └── exporters.py
│   └── alerts/
│       ├── rules.py
│       └── notifiers.py
```

#### Data Drift Detection
```python
class DataDriftDetector:
    """
    Detect distribution shifts in input data
    """
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.feature_distributions = self._compute_distributions(reference_data)

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> DriftReport:
        """
        Compare current data against reference using:
        - Kolmogorov-Smirnov test (continuous features)
        - Chi-square test (categorical features)
        - Population Stability Index (PSI)
        """
        drift_results = {}

        for feature in self.reference_data.columns:
            if is_continuous(feature):
                drift_results[feature] = self._ks_test(
                    self.reference_data[feature],
                    current_data[feature]
                )
            else:
                drift_results[feature] = self._chi_square_test(
                    self.reference_data[feature],
                    current_data[feature]
                )

        # Calculate PSI for all features
        psi_scores = self._calculate_psi(current_data)

        return DriftReport(
            drift_detected=any(r.p_value < threshold for r in drift_results.values()),
            feature_drift=drift_results,
            psi_scores=psi_scores,
            recommendation=self._get_recommendation(drift_results, psi_scores)
        )

    def _get_recommendation(self, drift_results, psi_scores):
        """Determine if retraining is needed"""
        significant_drift = sum(
            1 for r in drift_results.values() if r.p_value < 0.05
        )

        if significant_drift > len(drift_results) * 0.3:
            return "RETRAIN_IMMEDIATELY"
        elif significant_drift > len(drift_results) * 0.1:
            return "SCHEDULE_RETRAINING"
        else:
            return "CONTINUE_MONITORING"
```

#### Model Performance Monitoring
```python
class ModelPerformanceMonitor:
    """
    Monitor model predictions and performance in production
    """
    def __init__(self, kafka_consumer: KafkaConsumer):
        self.consumer = kafka_consumer
        self.metrics_window = deque(maxlen=10000)

    def monitor(self):
        """
        Continuously monitor predictions from Kafka
        """
        for message in self.consumer:
            prediction_data = json.loads(message.value)

            # Collect metrics
            self.metrics_window.append(prediction_data)

            # Compute rolling metrics
            if len(self.metrics_window) >= 1000:
                metrics = self._compute_metrics()

                # Export to Prometheus
                self._export_metrics(metrics)

                # Check for anomalies
                alerts = self._check_alerts(metrics)
                if alerts:
                    self._send_alerts(alerts)

    def _compute_metrics(self):
        """Compute performance metrics from predictions"""
        df = pd.DataFrame(self.metrics_window)

        return {
            "prediction_distribution": df["prediction"].describe(),
            "confidence_stats": df["confidence"].describe(),
            "feature_statistics": {
                col: df[col].describe()
                for col in df.columns if col.startswith("feature_")
            },
            "throughput": len(df) / self._window_duration_seconds(),
            "avg_latency": df["inference_latency"].mean()
        }

    def _check_alerts(self, metrics):
        """Check if metrics violate thresholds"""
        alerts = []

        # Check prediction distribution shift
        if self._prediction_drift_detected(metrics):
            alerts.append(Alert(
                level="WARNING",
                message="Prediction distribution shift detected",
                action="Review model performance"
            ))

        # Check latency degradation
        if metrics["avg_latency"] > self.latency_threshold:
            alerts.append(Alert(
                level="CRITICAL",
                message=f"High latency: {metrics['avg_latency']}ms",
                action="Scale up serving replicas"
            ))

        return alerts
```

#### Prometheus Metrics Export
```python
from prometheus_client import Gauge, Histogram, Counter

# Define metrics
prediction_distribution = Histogram(
    'model_prediction_distribution',
    'Distribution of model predictions',
    buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

inference_latency = Histogram(
    'model_inference_latency_seconds',
    'Model inference latency',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

data_drift_score = Gauge(
    'model_data_drift_score',
    'Data drift PSI score',
    ['feature']
)

prediction_count = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_version']
)

# Grafana dashboard queries
"""
Dashboard Panels:

1. Prediction Throughput
   Query: rate(model_predictions_total[5m])

2. Latency P95
   Query: histogram_quantile(0.95, rate(model_inference_latency_seconds_bucket[5m]))

3. Data Drift Alert
   Query: model_data_drift_score > 0.25

4. Prediction Distribution
   Query: rate(model_prediction_distribution_bucket[5m])
"""
```

#### Alerting Rules
```yaml
# prometheus/alert-rules.yaml
groups:
  - name: model_alerts
    interval: 30s
    rules:
      - alert: HighDataDrift
        expr: model_data_drift_score > 0.25
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High data drift detected"
          description: "Data drift score {{ $value }} exceeds threshold"
          action: "Consider retraining the model"

      - alert: ModelLatencyHigh
        expr: histogram_quantile(0.95, rate(model_inference_latency_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Model inference latency is high"
          description: "P95 latency is {{ $value }}s"
          action: "Scale up serving pods"

      - alert: PredictionDistributionShift
        expr: abs(delta(model_prediction_distribution_bucket[1h])) > 0.2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Prediction distribution has shifted"
          description: "Significant change in prediction patterns"
          action: "Investigate model behavior"
```

---

## Technology Stack

### Core Technologies
| Component | Technology | Reason |
|-----------|-----------|---------|
| **Orchestration** | Dagster | Asset-based, native data lineage, Python-first |
| **Streaming** | Kafka | Industry standard, high throughput, durability |
| **ML Framework** | PyTorch | Requested in JD, excellent distributed training |
| **Containerization** | Docker | Required, industry standard |
| **Orchestration** | Kubernetes | Requested in JD, cloud-native scaling |
| **Experiment Tracking** | MLflow | Model registry, experiment tracking, deployment |
| **Model Serving** | TorchServe | Native PyTorch support, production-ready |
| **Monitoring** | Prometheus + Grafana | Industry standard, rich ecosystem |
| **Data Quality** | Great Expectations | Comprehensive validation framework |
| **CI/CD** | GitHub Actions | Native GitHub integration, flexible |

### Infrastructure Stack
```yaml
Development:
  - Python 3.10+
  - Poetry (dependency management)
  - Pre-commit hooks
  - Ruff (linting)
  - MyPy (type checking)
  - Pytest (testing)

Data Layer:
  - MinIO (S3-compatible object storage)
  - PostgreSQL (metadata, MLflow backend)
  - Redis (caching)

Compute:
  - Kubernetes (container orchestration)
  - Docker (containerization)
  - CUDA (GPU support)

Monitoring:
  - Prometheus (metrics)
  - Grafana (dashboards)
  - ELK Stack (logs)
  - Evidently AI (ML monitoring)
```

---

## Data Flow

### End-to-End Flow

```
1. DATA INGESTION
   Raw Data → Kafka → Consumer → Validation → Data Lake (S3/MinIO)

2. ELT PIPELINE (Dagster)
   Data Lake → Quality Checks → Feature Engineering → Feature Store

3. MODEL TRAINING (Triggered by schedule/sensor)
   Feature Store → Distributed Training (K8s) → Model Checkpoints → MLflow

4. MODEL VALIDATION
   Trained Model → Validation Suite → Comparison with Champion → Registry

5. MODEL DEPLOYMENT (CI/CD)
   Model Registry → Docker Build → K8s Deployment → Canary Release

6. MODEL SERVING
   Client Request → Load Balancer → TorchServe → Prediction → Response
                                              ↓
                                        Log to Kafka

7. MONITORING
   Prediction Logs → Drift Detection → Alerts → Retraining Trigger
```

### Retraining Trigger Flow
```
Monitoring → Drift Detection → Alert Manager → Dagster Sensor → Training Pipeline
```

---

## Scalability & Performance

### Horizontal Scaling
- **Data Ingestion**: Kafka consumer groups (parallel processing)
- **Training**: Kubernetes job scaling (multiple nodes)
- **Serving**: HPA based on CPU/QPS metrics
- **Monitoring**: Distributed consumers for prediction logs

### Performance Optimizations
1. **Training**
   - Mixed precision training (AMP)
   - Gradient accumulation
   - Distributed data parallel (DDP)
   - Model parallelism for large models
   - Efficient data loading (num_workers, prefetch)

2. **Serving**
   - Model quantization (INT8)
   - ONNX runtime optimization
   - Batch inference
   - Model caching
   - GPU sharing

3. **Data Pipeline**
   - Partition pruning
   - Columnar storage (Parquet)
   - Data caching
   - Incremental processing

---

## Monitoring & Observability

### Three Pillars

1. **Metrics** (Prometheus)
   - System metrics: CPU, memory, GPU utilization
   - Application metrics: throughput, latency, error rate
   - ML metrics: drift scores, prediction distribution

2. **Logs** (ELK)
   - Application logs
   - Audit logs
   - Error tracking

3. **Traces** (OpenTelemetry)
   - Request tracing
   - Distributed tracing across services
   - Performance profiling

### Dashboards
- **System Health**: Resource utilization, pod status
- **Model Performance**: Latency, throughput, error rate
- **ML Metrics**: Drift scores, prediction distribution
- **Pipeline Status**: DAG runs, asset materialization

---

## Next Steps

This design covers:
✅ All job requirements (ELT, MLOps, CI/CD, Monitoring)
✅ All preferred technologies (PyTorch, Kubernetes, Dagster, Kafka)
✅ Production-grade patterns
✅ Scalability and reliability

### Questions for Refinement:

1. **Use Case**: What type of ML problem? (CV, NLP, tabular, etc.)
2. **Scale**: Expected data volume and request throughput?
3. **Infrastructure**: Cloud provider (AWS, GCP, Azure) or on-prem?
4. **Budget**: Resource constraints for compute?
5. **Timeline**: Which components to prioritize first?

Would you like me to:
- Dive deeper into any specific component?
- Start implementing the foundation (project structure, configs)?
- Create detailed API contracts between components?
- Design specific data schemas?

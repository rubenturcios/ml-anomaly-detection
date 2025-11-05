# AWS SageMaker ML Deployment Pipeline

A production ML system built on AWS infrastructure demonstrating end-to-end machine learning deployment using SageMaker, AWS CDK, and serverless architecture. This project showcases dual-model deployment with real-time inference capabilities for event and custom predictions.

## Overview

This project implements a complete MLOps pipeline on AWS, featuring automated model training, deployment, and inference using AWS SageMaker and AWS CDK for infrastructure as code. The system deploys two independent ML models (event and custom predictors) with custom inference endpoints and Lambda integration for serverless predictions.

## Project Structure
```
.
├── README.md
├── app.py
├── cdk.json
├── config.py
├── data/
│   ├── event_predicted.csv
│   ├── event_training_data.csv
│   ├── custom_predicted.csv
│   ├── custom_training_data.csv
│   └── test_df.csv
├── ml_stack/
│   ├── __init__.py
│   ├── glue_stack.py
│   └── ml_stack.py
├── model/
│   ├── event_estimator.joblib
│   ├── event_estimator.tar.gz
│   ├── event_model.joblib
│   ├── event_model.tar.gz
│   ├── custom_estimator.joblib
│   ├── custom_estimator.tar.gz
│   ├── custom_model.joblib
│   └── custom_model.tar.gz
├── notebooks/
│   ├── explainer.ipynb
│   ├── model.ipynb
│   └── model_2.ipynb
├── requirements-dev.txt
├── requirements.txt
├── src/
│   ├── Dockerfile
│   ├── deploy.py
│   ├── endpoint_code/
│   │   ├── event_estimator.py
│   │   ├── event_inference.py
│   │   ├── custom_estimator.py
│   │   ├── custom_inference.py
│   │   └── requirements.txt
│   ├── lambda_handler.py
│   └── utils.py
└── tests/
    └── unit/
        └── test_ml_stack.py
```

## Key Features

### Machine Learning
- **Dual Model Architecture**: Separate event and custom prediction models
- **Model Persistence**: Joblib serialization with tar.gz packaging for SageMaker
- **Custom Estimators**: Tailored estimator classes for each prediction task
- **Model Explainability**: Dedicated notebook for model interpretation and analysis
- **Training Data Management**: Organized datasets with prediction outputs

### AWS Infrastructure (CDK)
- **Infrastructure as Code**: Complete AWS infrastructure defined in CDK
- **SageMaker Integration**: Automated model deployment to SageMaker endpoints
- **AWS Glue**: Data processing and ETL pipeline stack
- **Lambda Functions**: Serverless inference invocation
- **Custom Docker Images**: Containerized inference code for SageMaker

### MLOps & Engineering
- **Automated Deployment**: One-command infrastructure and model deployment
- **Custom Inference**: Specialized inference logic for each model type
- **Version Control**: Model versioning through tar.gz artifacts
- **Testing Suite**: Unit tests for stack validation
- **Configuration Management**: Centralized config for environment settings

## Technical Stack

- **Cloud Platform**: AWS
- **ML Service**: Amazon SageMaker
- **IaC**: AWS CDK (Python)
- **Data Processing**: AWS Glue
- **Serverless Compute**: AWS Lambda
- **ML Framework**: scikit-learn (joblib serialization)
- **Container**: Docker
- **Language**: Python 3.8+

## Architecture
```
┌─────────────────┐
│   Data Source   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   AWS Glue      │  (ETL Pipeline)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         SageMaker Endpoints         │
│  ┌─────────────┐  ┌──────────────┐ │
│  │ Event Model │  │custom Model│ │
│  └──────┬──────┘  └──────┬───────┘ │
└─────────┼─────────────────┼─────────┘
          │                 │
          └────────┬────────┘
                   ▼
          ┌────────────────┐
          │ Lambda Handler │
          └────────────────┘
```

## Getting Started

### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured with credentials
- AWS CDK installed (`npm install -g aws-cdk`)
- Python 3.8 or higher
- Docker (for building custom inference containers)

### Installation

1. **Clone the repository**
```bash
   git clone <repository-url>
   cd <project-directory>
```

2. **Set up Python environment**
```bash
   python -m venv .venv
   
   # Windows
   source.bat
   
   # Linux/Mac
   source .venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
```

4. **Configure AWS credentials**
```bash
   aws configure
```

### Deployment

1. **Bootstrap CDK** (first time only)
```bash
   cdk bootstrap
```

2. **Review infrastructure changes**
```bash
   cdk diff
```

3. **Deploy the stack**
```bash
   cdk deploy
```

4. **Deploy models to SageMaker**
```bash
   python src/deploy.py
```

## Usage

### Model Training

Explore and train models using the provided notebooks:
```bash
jupyter notebook notebooks/model.ipynb
```

### Making Predictions

#### Via Lambda Function
```python
import boto3
import json

lambda_client = boto3.client('lambda')

payload = {
    'model': 'event',  # or 'custom'
    'data': {
        # your input features
    }
}

response = lambda_client.invoke(
    FunctionName='<your-lambda-function-name>',
    Payload=json.dumps(payload)
)
```

#### Direct SageMaker Endpoint
```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

response = runtime.invoke_endpoint(
    EndpointName='<endpoint-name>',
    ContentType='application/json',
    Body=json.dumps(data)
)
```

## Project Components

### CDK Stacks

- **ml_stack.py**: Core ML infrastructure including SageMaker endpoints
- **glue_stack.py**: Data processing pipeline with AWS Glue
- **app.py**: Main CDK application entry point

### Model Components

- **event_estimator.py**: Custom estimator for event predictions
- **event_inference.py**: Inference logic for event model
- **custom_estimator.py**: Custom estimator for custom predictions
- **custom_inference.py**: Inference logic for custom model

### Utilities

- **deploy.py**: Automated model deployment script
- **lambda_handler.py**: Serverless inference handler
- **utils.py**: Common utilities and helper functions
- **config.py**: Environment and configuration management

## Model Details

### Event Prediction Model
- **Purpose**: [Describe what the event model predicts]
- **Features**: [Key features used]
- **Performance**: [Add metrics]

### custom Prediction Model
- **Purpose**: [Describe what the custom model predicts]
- **Features**: [Key features used]
- **Performance**: [Add metrics]

## Development

### Running Tests
```bash
pytest tests/
```

### Model Experimentation

Use the notebooks for iterative development:
- `model.ipynb`: Primary model development
- `model_2.ipynb`: Alternative model experiments
- `explainer.ipynb`: Model interpretability and feature importance

### Building Custom Docker Images
```bash
cd src
docker build -t sagemaker-inference .
```

## Project Highlights

This project demonstrates:

✅ **MLOps Best Practices**: Automated deployment, version control, testing  
✅ **AWS Cloud Architecture**: SageMaker, Lambda, Glue integration  
✅ **Infrastructure as Code**: Complete AWS CDK implementation  
✅ **Scalable Inference**: Real-time predictions via SageMaker endpoints  
✅ **Serverless Integration**: Lambda for cost-effective inference  
✅ **Custom Inference Logic**: Tailored inference code for specific use cases  
✅ **Model Versioning**: Proper artifact management and versioning  
✅ **Production Ready**: Containerized deployment with monitoring  

## Cost Optimization

- Lambda functions for occasional inference (pay-per-use)
- SageMaker endpoints can be configured for auto-scaling
- Glue jobs run on-demand for ETL processing

## Monitoring & Maintenance

- SageMaker endpoint metrics available in CloudWatch
- Lambda invocation logs and errors in CloudWatch Logs
- Model performance tracking through prediction outputs

## Testing
```bash
# Run unit tests
pytest tests/unit/

# Test specific stack
pytest tests/unit/test_ml_stack.py
```

## Cleanup

To avoid ongoing AWS charges:
```bash
cdk destroy
```

## Acknowledgments

- AWS SageMaker documentation
- AWS CDK examples
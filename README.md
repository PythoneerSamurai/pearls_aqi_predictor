# 🌍 AQI Prediction System

> **A 100% Serverless, End-to-End Air Quality Index Prediction Application for Rawalpindi**

![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Pipeline Components](#pipeline-components)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 📌 Overview

The **AQI Prediction System** is a fully automated, serverless machine learning application that forecasts hourly Air Quality Index (AQI) values for Rawalpindi, Pakistan. The system integrates data ingestion, feature engineering, model training, deployment, and interactive visualization in a seamless, production-ready pipeline.

### Key Highlights

- 🤖 **Daily Model Retraining**: Continuous adaptation to evolving pollution patterns
- 📊 **72-Hour Forecasts**: Hourly granularity with daily aggregation
- ⚡ **100% Serverless**: Leveraging GitHub Actions and Hopsworks for scalability
- 🎨 **Interactive Dashboard**: Real-time visualization with Streamlit
- 📈 **High Accuracy**: XGBoost achieves R² = 0.982 with MAE of 1.30 AQI points
- 🔄 **Automated CI/CD**: Fully orchestrated pipelines with zero manual intervention

## ✨ Features

### Data Pipeline
- **Hourly data ingestion + Historical Data Backfill** from Open-Meteo API
- **Automated feature engineering** with temporal, seasonal, and domain-specific features
- **Feature store integration** with Hopsworks for scalable data management
- **Duplicate prevention** using datetime-based primary keys

### Model Training
- **Five ensemble models** for robust predictions:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor (SVR)
  - K-Nearest Neighbors (KNN)
  - XGBoost Regressor
- **Daily retraining** with automatic hyperparameter tuning
- **Model versioning** with Hopsworks Model Registry
- **KSERVE deployment** for production inference

### Inference & Forecasting
- **72-hour hourly predictions** (configurable up to 5 days)
- **Consistent feature engineering** across training and inference
- **Ensemble mean predictions** for robust forecasts
- **On-demand inference** triggered by frontend requests

### Interactive Frontend
- **Streamlit-based dashboard** for public-facing predictions
- **Model selection interface** to compare individual models
- **Interactive Plotly charts** with AQI category coloring
- **Daily summary cards** with health risk assessment
- **Real-time updates** with fresh forecast generation

### Automation & Monitoring
- **GitHub Actions workflows** for pipeline orchestration
- **Secure secret management** for API credentials
- **Continuous data accumulation** for improved long-term trends
- **Automated deployment** with zero downtime

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Hourly Data Backfill (One-time) + Data Ingestion (Hourly)        │
│         GitHub Actions → Open-Meteo API → Weather & Pollutants   │
└─────────────────────┬────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────────┐
│              Feature Engineering & Storage                       │
│      Temporal, Seasonal, Wind Decomposition → Hopsworks          │
└────────────────────┬─────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────────┐
│             Model Training Pipeline (Daily)                      │
│    Feature Retrieval → Model Training → Hyperparameter Tuning    │
└────────────────────┬─────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────────┐
│           Model Registry & Deployment (Hopsworks)                │
│     Version Management → KSERVE Deployment → Active Models       │
└────────────────────┬─────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────────┐
│              Inference Pipeline (On-Demand)                      │
│    Fetch Forecasts → Feature Engineering → Model Prediction      │
└────────────────────┬─────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────────┐
│         Interactive Dashboard (Streamlit Frontend)               │
│  User Interface → Model Selection → Visualization → Predictions  │
└──────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Data Source** | Open-Meteo API |
| **Data Storage** | Hopsworks Feature Store |
| **ML Frameworks** | Scikit-learn, XGBoost, Pandas, NumPy |
| **Model Registry** | Hopsworks Model Registry |
| **Model Serving** | KSERVE |
| **Frontend** | Streamlit, Plotly |
| **Automation** | GitHub Actions |
| **Explainability** | SHAP |
| **Language** | Python 3.8+ |

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- GitHub account with Actions enabled
- Hopsworks account (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PythoneerSamurai/pearls_aqi_predictor.git
   cd pearls_aqi_predictor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Hopsworks**
   - Create a free account at [Hopsworks](https://www.hopsworks.ai/)
   - Generate an API key
   - Add it as a GitHub Secret named `HOPSWORKS_API_KEY`

### Running Locally

#### Data Pipeline
```bash
python dataset_pipeline.py
```

#### Training Pipeline
```bash
python training_pipeline.py
```

#### Inference & Dashboard
```bash
streamlit run inference_app.py
```

## 📊 Pipeline Components

### Data Pipeline (`dataset_pipeline.py`)
Performs historical data backfill once + Runs hourly via GitHub Actions workflow `data_pipeline.yml`

**Functionality:**
- Fetches hourly weather and pollutant data from Open-Meteo API
- Applies feature engineering transformations
- Stores features and targets in Hopsworks feature group
- Prevents duplicates using datetime primary key

**Schedule:** Every hour at minute 5 (cron: `5 * * * *`)

### Training Pipeline (`training_pipeline.py`)
Runs daily via GitHub Actions workflow `training_pipeline.yml`

**Functionality:**
- Retrieves data from `aqi_hourly_features` feature group
- Splits data into 90% train, 10% test sets
- Trains five regression models with standard hyperparameters
- Evaluates models using R² and MAE metrics
- Manages model versioning and KSERVE deployment
- Deploys best-performing model (XGBoost by default)

**Schedule:** Daily at configured time

### Inference Pipeline
Triggered on-demand by the Streamlit frontend

**Functionality:**
- Fetches 72-hour forecast data from Open-Meteo
- Applies identical feature engineering logic
- Generates hourly predictions using deployed model
- Calculates daily AQI as mean of hourly predictions
- Returns ensemble predictions if available

### Streamlit Dashboard (`inference_app.py`)
Interactive web application for visualization

**Features:**
- Model selection dropdown (individual models or ensemble)
- Interactive Plotly charts with hourly forecasts
- AQI category color coding (Good, Moderate, Unhealthy, etc.)
- Daily summary cards with health assessments
- Real-time prediction generation

**Live Demo:** [haroons-pearls-aqi-predictor.streamlit.app](https://haroons-pearls-aqi-predictor.streamlit.app/)

## 📈 Model Performance

### Evaluation Results (Test Set - 10% of Data)

| Model | R² Score | MAE (AQI Points) |
|-------|----------|------------------|
| **XGBoost** | **0.982** | **1.30** |
| Random Forest | 0.981 | 2.13 |
| Gradient Boosting | 0.903 | 6.61 |
| SVR | 0.570 | 13.28 |
| KNN | 0.240 | 18.25 |

### Key Insights

- **Tree-based ensemble methods** (XGBoost, Random Forest) significantly outperform linear and distance-based models
- **XGBoost achieves state-of-the-art performance** with minimal prediction error
- **Feature engineering is highly effective**, with pollutants and temporal features showing strong predictive power
- **SHAP analysis reveals**:
  - PM2.5 and PM10 are the most influential features
  - Clear linear relationships between pollutants and AQI
  - Model predictions based on physically meaningful features

## 📁 Project Structure

```
pearls_aqi_predictor/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .github/
│   └── workflows/
│       ├── data_pipeline.yml          # Hourly data ingestion workflow
│       └── training_pipeline.yml      # Daily model training workflow
├── dataset_pipeline.py                # Data ingestion & feature engineering
├── training_pipeline.py               # Model training & deployment
├── inference_app.py                   # Streamlit dashboard
├── xgboost_predictor.py              # Custom XGBoost inference script
├── notebooks/
│   └── eda_analysis.ipynb            # Exploratory data analysis
└── docs/
    └── document.tex                   # Project report (LaTeX)
```

## ⚙️ Configuration

### Target Location
- **City**: Rawalpindi, Pakistan
- **Latitude**: 33.5973
- **Longitude**: 73.0479

### Data Features

#### Weather Features (Open-Meteo)
- Temperature, Humidity, Wind Speed, Wind Direction
- Atmospheric Pressure, Precipitation

#### Pollutants (Open-Meteo)
- PM2.5, PM10, CO, NO2, SO2, Ozone

#### Engineered Features
- **Temporal**: Hour, Day of Week, Month, Day of Year, Weekend Flag, Rush Hour Flag
- **Seasonal**: Winter (0), Spring (1), Summer (2), Autumn (3)
- **Wind**: U-component (east-west), V-component (north-south)
- **Derived**: Stagnation Flag (wind speed < 2 m/s), Temperature-Humidity Product

### Hyperparameters (Default)

All models trained with standard hyperparameters:
- `n_estimators`: 100
- `random_state`: 42
- `n_jobs`: -1 (parallel processing)

## 🔐 Environment Variables & Secrets

Configure the following GitHub Secrets for CI/CD:

| Secret | Description |
|--------|-------------|
| `HOPSWORKS_API_KEY` | Hopsworks API key for feature store access |
| `HOPSWORKS_PROJECT_NAME` | Hopsworks project name (optional) |

## 🚢 Deployment

### GitHub Actions Workflows

#### Data Pipeline Workflow
```yaml
# Runs every hour at minute 5
schedule:
  - cron: '5 * * * *'
```

#### Training Pipeline Workflow
```yaml
# Runs daily at configured time (default: midnight UTC)
schedule:
  - cron: '0 0 * * *'
```

### Streamlit Deployment

Deploy the Streamlit app to Streamlit Cloud:

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repository
4. Configure secrets in Streamlit Cloud settings

## 🎯 Prediction Strategy

The system uses **hourly-level predictions** instead of direct daily forecasts:

1. Models predict **hourly AQI values** for the next 72 hours
2. Daily AQI is calculated as the **mean of 24 hourly predictions**
3. This approach captures **intra-day pollution spikes** and is more robust
4. Aligns with standard AQI calculation methodology

## 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) analysis provides transparency:

- **Most Important Features**: PM2.5, PM10, Temperature, Humidity
- **Feature Impact**: Clear separation between high and low feature values
- **Prediction Justification**: Each prediction is traceable to input features

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Limited historical data (92 days from Open-Meteo) | Continuous hourly data gathering for accumulation over time |
| Hopsworks auto-incrementing model versions | Delete-and-redeploy strategy ensuring only version 1 exists |
| XGBoost deployment complexity | Custom predictor script (`xgboost_predictor.py`) uploaded to Hopsworks |

## 🔮 Future Enhancements

- [ ] Alert system for hazardous AQI levels (SMS/Email notifications)
- [ ] Deep learning models (LSTM) for time-series forecasting
- [ ] Multi-city expansion with location-based APIs
- [ ] Mobile app integration
- [ ] Extended 7-day forecasts
- [ ] Confidence intervals for predictions
- [ ] Anomaly detection for data quality monitoring

## 📊 EDA Findings

Key observations from exploratory data analysis:

- **Target Distribution**: AQI primarily in "Moderate" (0-50) and "Unhealthy for Sensitive Groups" (101-150) categories
- **Feature Correlations**: PM2.5, PM10, SO2 show strongest positive correlation with AQI
- **Temporal Patterns**: Distinct diurnal cycle with peaks in evening (increased emissions)
- **Seasonal Trends**: Winter months exhibit 20-30% higher AQI (temperature inversions)
- **Climate Impact**: Temperature shows moderate positive correlation; humidity shows negative correlation

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Haroon Rashid**

- GitHub: [@PythoneerSamurai](https://github.com/PythoneerSamurai)

## 🙏 Acknowledgments

- [Open-Meteo](https://open-meteo.com/) - Free weather & air quality data
- [Hopsworks](https://www.hopsworks.ai/) - Feature store & model registry
- [Streamlit](https://streamlit.io/) - Interactive dashboard framework
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting library
- [SHAP](https://shap.readthedocs.io/) - Model explainability

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact the project maintainer

---

**Last Updated**: 2026-05-24 | **Status**: Active & Maintained

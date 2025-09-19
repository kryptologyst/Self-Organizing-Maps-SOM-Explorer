# Changelog

All notable changes to the Self-Organizing Maps Explorer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-09

### Added
- Modern SOM implementation with comprehensive features
- Interactive Streamlit web interface
- Multiple dataset support (real-world and synthetic)
- Advanced visualization capabilities:
  - Interactive Plotly charts
  - U-Matrix (Unified Distance Matrix)
  - Activation response maps
  - Feature weight visualizations
  - Training progress monitoring
- Comprehensive evaluation metrics:
  - Quantization error
  - Topographic error
  - Silhouette score
  - Calinski-Harabasz index
  - Davies-Bouldin score
- K-means comparison functionality
- Model persistence and configuration export
- Mock database with 10 different datasets
- Professional project structure with documentation

### Enhanced
- Updated from basic MiniSom usage to comprehensive ML pipeline
- Replaced matplotlib-only visualization with interactive Plotly charts
- Added multiple preprocessing options (StandardScaler, MinMaxScaler, RobustScaler)
- Implemented proper error handling and validation
- Added type hints and comprehensive docstrings

### Technical Improvements
- Modern Python packaging with setup.py
- Comprehensive requirements.txt with latest library versions
- Professional README with detailed documentation
- Contributing guidelines and license
- Git ignore file for clean repository
- Modular code structure for maintainability

### Datasets
- **Real-world**: Iris, Wine, Breast Cancer, Digits
- **Synthetic**: Blobs, Circles, Moons, Classification, Time Series, Customer Segmentation

### Dependencies
- numpy>=1.24.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- plotly>=5.15.0
- scikit-learn>=1.3.0
- minisom>=2.3.0
- streamlit>=1.25.0
- joblib>=1.3.0
- kaleido>=0.2.1

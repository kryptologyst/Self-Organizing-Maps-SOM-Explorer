# Self-Organizing Maps (SOM) Explorer

A modern, interactive implementation of Self-Organizing Maps with comprehensive visualization and analysis tools.

## Features

- **Modern ML Implementation**: Built with latest scikit-learn, plotly, and streamlit
- **Interactive Web Interface**: Comprehensive Streamlit dashboard for exploration
- **Multiple Datasets**: Real-world and synthetic datasets for testing
- **Advanced Visualizations**: U-matrix, activation maps, and interactive plots
- **Comprehensive Metrics**: Clustering evaluation and K-means comparison
- **Model Persistence**: Save and load trained models
- **Professional Structure**: Clean, modular code ready for production

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd 0078_Self-organizing_maps

# Install dependencies
pip install -r requirements.txt
```

### Run the Interactive App

```bash
streamlit run streamlit_app.py
```

### Basic Usage

```python
from som_model import ModernSOM, SOMConfig
from data_generator import MockDatabase
import numpy as np

# Load data
db = MockDatabase()
X, y, feature_names, target_names = db.get_dataset('iris')

# Configure and train SOM
config = SOMConfig(grid_size=(10, 10), num_iterations=1000)
som = ModernSOM(config)
metrics = som.train(X, scaler_type='standard')

# Create visualizations
fig = som.create_interactive_visualization(X, y, feature_names)
fig.show()

# Save model
som.save_model('my_som_model.joblib')
```

## Available Datasets

| Dataset | Description | Samples | Features | Classes |
|---------|-------------|---------|----------|---------|
| **iris** | Classic iris flower dataset | 150 | 4 | 3 |
| **wine** | Wine recognition dataset | 178 | 13 | 3 |
| **breast_cancer** | Breast cancer diagnostic | 569 | 30 | 2 |
| **digits** | Handwritten digits (8x8) | 1797 | 64 | 10 |
| **blobs** | Synthetic blob clusters | Configurable | Configurable | Configurable |
| **circles** | Concentric circles | Configurable | 2 | 2 |
| **moons** | Two interleaving half circles | Configurable | 2 | 2 |
| **classification** | Random classification data | Configurable | Configurable | Configurable |
| **time_series** | Synthetic time series patterns | Configurable | 5 | 3 |
| **customer** | Customer segmentation data | Configurable | 6 | 4 |

## Architecture

### Core Components

- **`som_model.py`**: Modern SOM implementation with comprehensive features
- **`data_generator.py`**: Mock database and dataset generation utilities
- **`streamlit_app.py`**: Interactive web interface
- **`0078.py`**: Original simple implementation (legacy)

### Key Classes

#### `ModernSOM`
Enhanced SOM implementation with:
- Multiple preprocessing options (StandardScaler, MinMaxScaler, RobustScaler)
- Comprehensive training monitoring
- Advanced evaluation metrics
- Interactive visualizations
- Model persistence

#### `SOMConfig`
Configuration dataclass for SOM parameters:
```python
@dataclass
class SOMConfig:
    grid_size: Tuple[int, int] = (10, 10)
    sigma: float = 1.0
    learning_rate: float = 0.5
    num_iterations: int = 1000
    neighborhood_function: str = 'gaussian'
    topology: str = 'rectangular'
    activation_distance: str = 'euclidean'
    random_seed: Optional[int] = 42
```

#### `MockDatabase`
Dataset management system providing:
- Multiple real-world datasets
- Synthetic data generation
- Dataset information and statistics
- Pandas DataFrame integration

## Visualizations

### Interactive Plots
- **SOM Grid**: Data points mapped to winning neurons
- **U-Matrix**: Unified distance matrix showing cluster boundaries
- **Activation Response**: Neuron activation frequency
- **Feature Weights**: Weight visualization for each feature
- **Training Progress**: Quantization error over time

### Evaluation Metrics
- **Quantization Error**: Average distance to Best Matching Unit (BMU)
- **Topographic Error**: Measure of topology preservation
- **Silhouette Score**: Cluster separation quality
- **Calinski-Harabasz Index**: Cluster validity measure
- **Davies-Bouldin Score**: Cluster compactness and separation

## 🔧 Configuration Options

### SOM Parameters
- **Grid Size**: Dimensions of the SOM grid (e.g., 10x10)
- **Sigma**: Neighborhood radius for learning
- **Learning Rate**: Rate of weight updates
- **Iterations**: Number of training iterations
- **Neighborhood Function**: gaussian, mexican_hat, bubble, triangle
- **Topology**: rectangular, hexagonal
- **Distance Metric**: euclidean, manhattan, chebyshev

### Data Preprocessing
- **Standard Scaling**: Zero mean, unit variance
- **Min-Max Scaling**: Scale to [0, 1] range
- **Robust Scaling**: Use median and IQR

## Use Cases

### 1. Data Exploration
- Visualize high-dimensional data in 2D
- Identify clusters and patterns
- Understand feature relationships

### 2. Clustering Analysis
- Compare with K-means clustering
- Evaluate cluster quality
- Determine optimal number of clusters

### 3. Dimensionality Reduction
- Project high-dimensional data to 2D grid
- Preserve topological relationships
- Interactive exploration of data structure

### 4. Anomaly Detection
- Identify outliers through activation patterns
- Monitor quantization errors
- Detect unusual data points

## Technical Details

### Dependencies
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive visualizations
- **scikit-learn**: ML utilities and metrics
- **minisom**: Core SOM implementation
- **streamlit**: Web interface
- **joblib**: Model persistence

### Performance Considerations
- Efficient numpy operations
- Vectorized computations
- Memory-optimized for large datasets
- Configurable batch processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MiniSom**: Excellent SOM implementation by Giuseppe Vettigli
- **Scikit-learn**: Comprehensive ML library
- **Plotly**: Interactive visualization library
- **Streamlit**: Rapid web app development

## References

- Kohonen, T. (1982). Self-organized formation of topologically correct feature maps
- Vesanto, J., & Alhoniemi, E. (2000). Clustering of the self-organizing map
- Ultsch, A., & Siemon, H. P. (1990). Kohonen's self organizing feature maps for exploratory data analysis


# Self-Organizing-Maps-SOM-Explorer

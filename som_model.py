"""
Modern Self-Organizing Maps Implementation
Enhanced with latest ML techniques and comprehensive functionality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import joblib
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SOMConfig:
    """Configuration class for SOM parameters"""
    grid_size: Tuple[int, int] = (10, 10)
    sigma: float = 1.0
    learning_rate: float = 0.5
    num_iterations: int = 1000
    neighborhood_function: str = 'gaussian'
    topology: str = 'rectangular'
    activation_distance: str = 'euclidean'
    random_seed: Optional[int] = 42


class ModernSOM:
    """
    Enhanced Self-Organizing Map implementation with modern ML practices
    """
    
    def __init__(self, config: SOMConfig = None):
        self.config = config or SOMConfig()
        self.som = None
        self.scaler = None
        self.is_trained = False
        self.training_history = []
        self.cluster_labels = None
        self.metrics = {}
        
    def preprocess_data(self, data: np.ndarray, scaler_type: str = 'standard') -> np.ndarray:
        """
        Preprocess data with various scaling options
        """
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if scaler_type not in scalers:
            raise ValueError(f"Scaler type must be one of {list(scalers.keys())}")
            
        self.scaler = scalers[scaler_type]
        return self.scaler.fit_transform(data)
    
    def initialize_som(self, input_data: np.ndarray):
        """Initialize SOM with given data"""
        if self.config.random_seed:
            np.random.seed(self.config.random_seed)
            
        self.som = MiniSom(
            x=self.config.grid_size[0],
            y=self.config.grid_size[1],
            input_len=input_data.shape[1],
            sigma=self.config.sigma,
            learning_rate=self.config.learning_rate,
            neighborhood_function=self.config.neighborhood_function,
            topology=self.config.topology,
            activation_distance=self.config.activation_distance,
            random_seed=self.config.random_seed
        )
        
        # Initialize weights
        self.som.pca_weights_init(input_data)
    
    def train(self, data: np.ndarray, scaler_type: str = 'standard', 
              verbose: bool = True) -> Dict:
        """
        Train the SOM with comprehensive monitoring
        """
        # Preprocess data
        processed_data = self.preprocess_data(data, scaler_type)
        
        # Initialize SOM
        self.initialize_som(processed_data)
        
        # Train with monitoring
        if verbose:
            print(f"Training SOM with {self.config.num_iterations} iterations...")
            
        # Track quantization error during training
        quantization_errors = []
        
        for i in range(0, self.config.num_iterations, max(1, self.config.num_iterations // 10)):
            self.som.train_batch(processed_data, num_iteration=max(1, self.config.num_iterations // 10))
            qe = self.som.quantization_error(processed_data)
            quantization_errors.append(qe)
            
            if verbose and i % (self.config.num_iterations // 5) == 0:
                print(f"Iteration {i}: Quantization Error = {qe:.4f}")
        
        self.is_trained = True
        self.training_history = quantization_errors
        
        # Calculate final metrics
        self._calculate_metrics(processed_data)
        
        return self.metrics
    
    def _calculate_metrics(self, data: np.ndarray):
        """Calculate comprehensive evaluation metrics"""
        if not self.is_trained:
            return
            
        # Get winner coordinates for each data point
        winner_coordinates = np.array([self.som.winner(x) for x in data])
        
        # Convert 2D coordinates to 1D labels for clustering metrics
        self.cluster_labels = winner_coordinates[:, 0] * self.config.grid_size[1] + winner_coordinates[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'quantization_error': self.som.quantization_error(data),
            'topographic_error': self.som.topographic_error(data),
            'silhouette_score': silhouette_score(data, self.cluster_labels) if len(np.unique(self.cluster_labels)) > 1 else 0,
            'calinski_harabasz_score': calinski_harabasz_score(data, self.cluster_labels) if len(np.unique(self.cluster_labels)) > 1 else 0,
            'davies_bouldin_score': davies_bouldin_score(data, self.cluster_labels) if len(np.unique(self.cluster_labels)) > 1 else 0,
            'n_clusters': len(np.unique(self.cluster_labels))
        }
    
    def predict_cluster(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data"""
        if not self.is_trained:
            raise ValueError("SOM must be trained before prediction")
            
        processed_data = self.scaler.transform(data)
        winner_coordinates = np.array([self.som.winner(x) for x in processed_data])
        return winner_coordinates[:, 0] * self.config.grid_size[1] + winner_coordinates[:, 1]
    
    def get_distance_map(self) -> np.ndarray:
        """Get U-matrix (unified distance matrix)"""
        if not self.is_trained:
            raise ValueError("SOM must be trained before getting distance map")
        return self.som.distance_map()
    
    def get_activation_response(self, data: np.ndarray) -> np.ndarray:
        """Get activation response for each neuron"""
        if not self.is_trained:
            raise ValueError("SOM must be trained before getting activation response")
            
        processed_data = self.scaler.transform(data)
        activation_map = np.zeros(self.config.grid_size)
        
        for x in processed_data:
            winner = self.som.winner(x)
            activation_map[winner] += 1
            
        return activation_map
    
    def create_interactive_visualization(self, data: np.ndarray, 
                                      labels: Optional[np.ndarray] = None,
                                      feature_names: Optional[List[str]] = None) -> go.Figure:
        """Create interactive Plotly visualization"""
        if not self.is_trained:
            raise ValueError("SOM must be trained before visualization")
        
        processed_data = self.scaler.transform(data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('SOM Grid with Data Points', 'U-Matrix (Distance Map)', 
                          'Activation Response', 'Feature Weights'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # 1. SOM Grid with data points
        winner_coordinates = np.array([self.som.winner(x) for x in processed_data])
        
        scatter_data = go.Scatter(
            x=winner_coordinates[:, 0],
            y=winner_coordinates[:, 1],
            mode='markers',
            marker=dict(
                color=labels if labels is not None else 'blue',
                colorscale='viridis',
                size=8,
                opacity=0.7
            ),
            text=[f"Point {i}" for i in range(len(data))],
            name='Data Points'
        )
        fig.add_trace(scatter_data, row=1, col=1)
        
        # 2. U-Matrix
        distance_map = self.get_distance_map()
        fig.add_trace(
            go.Heatmap(z=distance_map, colorscale='viridis', name='Distance Map'),
            row=1, col=2
        )
        
        # 3. Activation Response
        activation_map = self.get_activation_response(data)
        fig.add_trace(
            go.Heatmap(z=activation_map, colorscale='blues', name='Activation'),
            row=2, col=1
        )
        
        # 4. Feature weights (first feature as example)
        if processed_data.shape[1] > 0:
            weights = np.array([self.som.get_weights()[i, j, 0] 
                              for i in range(self.config.grid_size[0]) 
                              for j in range(self.config.grid_size[1])]).reshape(self.config.grid_size)
            fig.add_trace(
                go.Heatmap(z=weights, colorscale='rdbu', name='Feature 0 Weights'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Self-Organizing Map Analysis",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_training_plot(self) -> go.Figure:
        """Create training progress visualization"""
        if not self.training_history:
            raise ValueError("No training history available")
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.training_history,
            mode='lines+markers',
            name='Quantization Error',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="SOM Training Progress",
            xaxis_title="Training Checkpoint",
            yaxis_title="Quantization Error",
            template="plotly_white"
        )
        
        return fig
    
    def save_model(self, filepath: str):
        """Save trained SOM model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'som_weights': self.som.get_weights().tolist(),
            'config': asdict(self.config),
            'metrics': self.metrics,
            'training_history': self.training_history,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained SOM model"""
        model_data = joblib.load(filepath)
        
        self.config = SOMConfig(**model_data['config'])
        self.metrics = model_data['metrics']
        self.training_history = model_data['training_history']
        self.scaler = model_data['scaler']
        
        # Reconstruct SOM
        weights = np.array(model_data['som_weights'])
        input_len = weights.shape[2]
        
        self.som = MiniSom(
            x=self.config.grid_size[0],
            y=self.config.grid_size[1],
            input_len=input_len,
            sigma=self.config.sigma,
            learning_rate=self.config.learning_rate,
            neighborhood_function=self.config.neighborhood_function,
            topology=self.config.topology,
            activation_distance=self.config.activation_distance,
            random_seed=self.config.random_seed
        )
        
        # Set weights
        for i in range(self.config.grid_size[0]):
            for j in range(self.config.grid_size[1]):
                self.som._weights[i, j] = weights[i, j]
                
        self.is_trained = True
    
    def compare_with_kmeans(self, data: np.ndarray, n_clusters: int = None) -> Dict:
        """Compare SOM clustering with K-means"""
        if not self.is_trained:
            raise ValueError("SOM must be trained before comparison")
            
        processed_data = self.scaler.transform(data)
        
        if n_clusters is None:
            n_clusters = min(self.metrics['n_clusters'], len(np.unique(self.cluster_labels)))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_seed)
        kmeans_labels = kmeans.fit_predict(processed_data)
        
        # Calculate metrics for both
        comparison = {
            'som_silhouette': self.metrics['silhouette_score'],
            'kmeans_silhouette': silhouette_score(processed_data, kmeans_labels),
            'som_calinski_harabasz': self.metrics['calinski_harabasz_score'],
            'kmeans_calinski_harabasz': calinski_harabasz_score(processed_data, kmeans_labels),
            'som_davies_bouldin': self.metrics['davies_bouldin_score'],
            'kmeans_davies_bouldin': davies_bouldin_score(processed_data, kmeans_labels),
            'som_clusters': self.metrics['n_clusters'],
            'kmeans_clusters': n_clusters
        }
        
        return comparison

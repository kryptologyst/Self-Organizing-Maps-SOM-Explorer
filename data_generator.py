"""
Mock Database and Dataset Generator for SOM Testing
Provides various synthetic and real datasets for comprehensive testing
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_blobs, make_circles, make_moons, make_classification
)
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class DatasetGenerator:
    """Generate various datasets for SOM testing and demonstration"""
    
    @staticmethod
    def get_iris_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load Iris dataset"""
        iris = load_iris()
        return iris.data, iris.target, iris.feature_names, iris.target_names
    
    @staticmethod
    def get_wine_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load Wine dataset"""
        wine = load_wine()
        return wine.data, wine.target, wine.feature_names, wine.target_names
    
    @staticmethod
    def get_breast_cancer_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load Breast Cancer dataset"""
        cancer = load_breast_cancer()
        return cancer.data, cancer.target, cancer.feature_names, cancer.target_names
    
    @staticmethod
    def get_digits_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load Digits dataset (8x8 images)"""
        digits = load_digits()
        return digits.data, digits.target, [f'pixel_{i}' for i in range(64)], [str(i) for i in range(10)]
    
    @staticmethod
    def generate_blobs(n_samples: int = 300, n_features: int = 4, 
                      n_centers: int = 3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate blob clusters"""
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_centers,
            random_state=random_state,
            cluster_std=1.5
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        target_names = [f'cluster_{i}' for i in range(n_centers)]
        return X, y, feature_names, target_names
    
    @staticmethod
    def generate_circles(n_samples: int = 300, noise: float = 0.1, 
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate concentric circles"""
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state,
            factor=0.6
        )
        feature_names = ['x_coordinate', 'y_coordinate']
        target_names = ['inner_circle', 'outer_circle']
        return X, y, feature_names, target_names
    
    @staticmethod
    def generate_moons(n_samples: int = 300, noise: float = 0.1, 
                      random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate two interleaving half circles"""
        X, y = make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state
        )
        feature_names = ['x_coordinate', 'y_coordinate']
        target_names = ['moon_1', 'moon_2']
        return X, y, feature_names, target_names
    
    @staticmethod
    def generate_classification_data(n_samples: int = 300, n_features: int = 8,
                                   n_informative: int = 4, n_redundant: int = 2,
                                   n_classes: int = 3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate random classification dataset"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            random_state=random_state
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
        target_names = [f'class_{i}' for i in range(n_classes)]
        return X, y, feature_names, target_names
    
    @staticmethod
    def generate_time_series_data(n_samples: int = 200, n_features: int = 5,
                                 random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate synthetic time series data"""
        np.random.seed(random_state)
        
        # Generate time-based features
        t = np.linspace(0, 4*np.pi, n_samples)
        
        # Create different patterns
        patterns = []
        labels = []
        
        # Sine wave pattern
        sine_data = np.column_stack([
            np.sin(t) + 0.1 * np.random.randn(n_samples),
            np.cos(t) + 0.1 * np.random.randn(n_samples),
            np.sin(2*t) + 0.1 * np.random.randn(n_samples),
            np.cos(2*t) + 0.1 * np.random.randn(n_samples),
            t/max(t) + 0.1 * np.random.randn(n_samples)
        ])
        patterns.append(sine_data[:n_samples//3])
        labels.extend([0] * (n_samples//3))
        
        # Exponential pattern
        exp_data = np.column_stack([
            np.exp(-t/4) + 0.1 * np.random.randn(n_samples),
            np.exp(-t/2) + 0.1 * np.random.randn(n_samples),
            np.log(t + 1) + 0.1 * np.random.randn(n_samples),
            t**0.5 + 0.1 * np.random.randn(n_samples),
            np.ones(n_samples) + 0.1 * np.random.randn(n_samples)
        ])
        patterns.append(exp_data[:n_samples//3])
        labels.extend([1] * (n_samples//3))
        
        # Random walk pattern
        random_walk = np.cumsum(np.random.randn(n_samples, n_features) * 0.1, axis=0)
        patterns.append(random_walk[:n_samples - 2*(n_samples//3)])
        labels.extend([2] * (n_samples - 2*(n_samples//3)))
        
        X = np.vstack(patterns)
        y = np.array(labels)
        
        feature_names = ['sine_component', 'cosine_component', 'harmonic', 'phase', 'trend']
        target_names = ['periodic', 'exponential', 'random_walk']
        
        return X, y, feature_names, target_names
    
    @staticmethod
    def generate_customer_data(n_samples: int = 500, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate synthetic customer segmentation data"""
        np.random.seed(random_state)
        
        # Define customer segments
        segments = {
            'high_value': {'age': (35, 55), 'income': (70000, 120000), 'spending': (0.3, 0.6)},
            'young_professional': {'age': (25, 35), 'income': (40000, 80000), 'spending': (0.2, 0.4)},
            'budget_conscious': {'age': (30, 60), 'income': (25000, 50000), 'spending': (0.1, 0.25)},
            'luxury_buyer': {'age': (40, 65), 'income': (100000, 200000), 'spending': (0.4, 0.8)}
        }
        
        data = []
        labels = []
        
        samples_per_segment = n_samples // len(segments)
        
        for i, (segment_name, params) in enumerate(segments.items()):
            for _ in range(samples_per_segment):
                age = np.random.uniform(*params['age'])
                income = np.random.uniform(*params['income'])
                spending_ratio = np.random.uniform(*params['spending'])
                spending_score = spending_ratio * 100
                
                # Add some correlation and noise
                loyalty_score = 50 + spending_ratio * 30 + np.random.normal(0, 10)
                loyalty_score = np.clip(loyalty_score, 0, 100)
                
                frequency = spending_ratio * 20 + np.random.normal(0, 3)
                frequency = np.clip(frequency, 1, 25)
                
                recency = np.random.exponential(30) if spending_ratio > 0.3 else np.random.exponential(60)
                recency = np.clip(recency, 1, 365)
                
                data.append([age, income, spending_score, loyalty_score, frequency, recency])
                labels.append(i)
        
        X = np.array(data)
        y = np.array(labels)
        
        feature_names = ['age', 'annual_income', 'spending_score', 'loyalty_score', 'purchase_frequency', 'days_since_last_purchase']
        target_names = list(segments.keys())
        
        return X, y, feature_names, target_names


class MockDatabase:
    """Mock database interface for dataset management"""
    
    def __init__(self):
        self.generator = DatasetGenerator()
        self.available_datasets = {
            'iris': 'Classic iris flower dataset',
            'wine': 'Wine recognition dataset',
            'breast_cancer': 'Breast cancer diagnostic dataset',
            'digits': 'Handwritten digits dataset',
            'blobs': 'Synthetic blob clusters',
            'circles': 'Concentric circles dataset',
            'moons': 'Two interleaving half circles',
            'classification': 'Random classification dataset',
            'time_series': 'Synthetic time series patterns',
            'customer': 'Customer segmentation dataset'
        }
    
    def list_datasets(self) -> Dict[str, str]:
        """List all available datasets"""
        return self.available_datasets
    
    def get_dataset(self, name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Get dataset by name"""
        if name not in self.available_datasets:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(self.available_datasets.keys())}")
        
        dataset_methods = {
            'iris': self.generator.get_iris_data,
            'wine': self.generator.get_wine_data,
            'breast_cancer': self.generator.get_breast_cancer_data,
            'digits': self.generator.get_digits_data,
            'blobs': self.generator.generate_blobs,
            'circles': self.generator.generate_circles,
            'moons': self.generator.generate_moons,
            'classification': self.generator.generate_classification_data,
            'time_series': self.generator.generate_time_series_data,
            'customer': self.generator.generate_customer_data
        }
        
        return dataset_methods[name](**kwargs)
    
    def get_dataset_info(self, name: str) -> Dict:
        """Get detailed information about a dataset"""
        if name not in self.available_datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        X, y, feature_names, target_names = self.get_dataset(name)
        
        return {
            'name': name,
            'description': self.available_datasets[name],
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'feature_names': feature_names,
            'target_names': target_names,
            'data_shape': X.shape,
            'target_distribution': dict(zip(*np.unique(y, return_counts=True)))
        }
    
    def create_dataframe(self, name: str, **kwargs) -> pd.DataFrame:
        """Create pandas DataFrame from dataset"""
        X, y, feature_names, target_names = self.get_dataset(name, **kwargs)
        
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df['target_name'] = [target_names[i] for i in y]
        
        return df

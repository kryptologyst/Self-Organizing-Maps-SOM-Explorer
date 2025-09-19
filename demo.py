"""
Demonstration script for the Modern SOM implementation
Shows basic usage and creates sample visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from som_model import ModernSOM, SOMConfig
from data_generator import MockDatabase

def main():
    """Run SOM demonstration"""
    print("🧠 Self-Organizing Maps Demonstration")
    print("=" * 50)
    
    # Initialize database
    db = MockDatabase()
    
    # Load Iris dataset
    print("\n📊 Loading Iris dataset...")
    X, y, feature_names, target_names = db.get_dataset('iris')
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    
    # Configure SOM
    config = SOMConfig(
        grid_size=(8, 8),
        sigma=1.0,
        learning_rate=0.5,
        num_iterations=500,
        random_seed=42
    )
    
    # Train SOM
    print("\n🚀 Training SOM...")
    som = ModernSOM(config)
    metrics = som.train(X, scaler_type='standard', verbose=True)
    
    # Display metrics
    print("\n📈 Training Results:")
    print(f"Quantization Error: {metrics['quantization_error']:.4f}")
    print(f"Topographic Error: {metrics['topographic_error']:.4f}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Number of Clusters: {metrics['n_clusters']}")
    
    # Create basic visualization
    print("\n📊 Creating visualization...")
    
    # Get winner coordinates
    winner_coordinates = np.array([som.som.winner(x) for x in som.scaler.transform(X)])
    
    # Create matplotlib plot
    plt.figure(figsize=(12, 5))
    
    # Plot 1: SOM grid with data points
    plt.subplot(1, 2, 1)
    colors = ['red', 'green', 'blue']
    for i, target in enumerate(np.unique(y)):
        mask = y == target
        plt.scatter(winner_coordinates[mask, 0], winner_coordinates[mask, 1], 
                   c=colors[i], label=target_names[target], alpha=0.7, s=50)
    
    plt.title('SOM Grid - Iris Dataset')
    plt.xlabel('SOM X Coordinate')
    plt.ylabel('SOM Y Coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distance map (U-matrix)
    plt.subplot(1, 2, 2)
    distance_map = som.get_distance_map()
    plt.imshow(distance_map, cmap='viridis', origin='lower')
    plt.colorbar(label='Distance')
    plt.title('U-Matrix (Distance Map)')
    plt.xlabel('SOM X Coordinate')
    plt.ylabel('SOM Y Coordinate')
    
    plt.tight_layout()
    plt.savefig('som_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compare with K-means
    print("\n🔄 Comparing with K-means...")
    comparison = som.compare_with_kmeans(X, n_clusters=3)
    
    print("Comparison Results:")
    print(f"SOM Silhouette Score: {comparison['som_silhouette']:.4f}")
    print(f"K-means Silhouette Score: {comparison['kmeans_silhouette']:.4f}")
    print(f"SOM Calinski-Harabasz: {comparison['som_calinski_harabasz']:.2f}")
    print(f"K-means Calinski-Harabasz: {comparison['kmeans_calinski_harabasz']:.2f}")
    
    # Save model
    print("\n💾 Saving model...")
    som.save_model('demo_som_model.joblib')
    print("Model saved as 'demo_som_model.joblib'")
    
    print("\n✅ Demonstration completed!")
    print("Run 'streamlit run streamlit_app.py' for interactive interface")

if __name__ == "__main__":
    main()

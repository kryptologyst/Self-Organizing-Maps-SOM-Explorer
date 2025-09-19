"""
Interactive Streamlit UI for Self-Organizing Maps
Modern web interface with comprehensive visualization and analysis tools
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from som_model import ModernSOM, SOMConfig
from data_generator import MockDatabase
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Self-Organizing Maps Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'som_model' not in st.session_state:
    st.session_state.som_model = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

# Initialize database
@st.cache_resource
def get_database():
    return MockDatabase()

db = get_database()

# Main header
st.markdown('<h1 class="main-header">🧠 Self-Organizing Maps Explorer</h1>', unsafe_allow_html=True)
st.markdown("**Explore high-dimensional data through interactive Self-Organizing Maps with modern ML techniques**")

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Dataset selection
st.sidebar.subheader("📊 Dataset Selection")
dataset_options = list(db.list_datasets().keys())
selected_dataset = st.sidebar.selectbox(
    "Choose Dataset",
    options=dataset_options,
    help="Select from various real and synthetic datasets"
)

# Dataset parameters for synthetic data
dataset_params = {}
if selected_dataset in ['blobs', 'circles', 'moons', 'classification', 'time_series', 'customer']:
    st.sidebar.subheader("Dataset Parameters")
    if selected_dataset == 'blobs':
        dataset_params['n_samples'] = st.sidebar.slider("Number of samples", 100, 1000, 300)
        dataset_params['n_features'] = st.sidebar.slider("Number of features", 2, 10, 4)
        dataset_params['n_centers'] = st.sidebar.slider("Number of clusters", 2, 8, 3)
    elif selected_dataset in ['circles', 'moons']:
        dataset_params['n_samples'] = st.sidebar.slider("Number of samples", 100, 1000, 300)
        dataset_params['noise'] = st.sidebar.slider("Noise level", 0.0, 0.3, 0.1)
    elif selected_dataset == 'classification':
        dataset_params['n_samples'] = st.sidebar.slider("Number of samples", 100, 1000, 300)
        dataset_params['n_features'] = st.sidebar.slider("Number of features", 4, 20, 8)
        dataset_params['n_classes'] = st.sidebar.slider("Number of classes", 2, 5, 3)
    elif selected_dataset in ['time_series', 'customer']:
        dataset_params['n_samples'] = st.sidebar.slider("Number of samples", 200, 1000, 500)

# SOM Configuration
st.sidebar.subheader("🧠 SOM Parameters")
grid_x = st.sidebar.slider("Grid Width", 5, 20, 10)
grid_y = st.sidebar.slider("Grid Height", 5, 20, 10)
sigma = st.sidebar.slider("Sigma (neighborhood)", 0.5, 3.0, 1.0, 0.1)
learning_rate = st.sidebar.slider("Learning Rate", 0.1, 1.0, 0.5, 0.05)
num_iterations = st.sidebar.slider("Training Iterations", 100, 2000, 1000, 100)
scaler_type = st.sidebar.selectbox("Data Scaling", ['standard', 'minmax', 'robust'])

# Load data button
if st.sidebar.button("🔄 Load Dataset", type="primary"):
    with st.spinner("Loading dataset..."):
        try:
            X, y, feature_names, target_names = db.get_dataset(selected_dataset, **dataset_params)
            st.session_state.current_data = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'target_names': target_names,
                'dataset_name': selected_dataset
            }
            st.session_state.training_complete = False
            st.success(f"✅ Loaded {selected_dataset} dataset: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")

# Main content area
if st.session_state.current_data is not None:
    data = st.session_state.current_data
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🧠 Train SOM", "📈 Visualizations", "📋 Analysis", "💾 Model Management"])
    
    with tab1:
        st.header("📊 Dataset Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Dataset info
            st.subheader("Dataset Information")
            info = db.get_dataset_info(selected_dataset)
            
            info_cols = st.columns(4)
            with info_cols[0]:
                st.metric("Samples", info['n_samples'])
            with info_cols[1]:
                st.metric("Features", info['n_features'])
            with info_cols[2]:
                st.metric("Classes", info['n_classes'])
            with info_cols[3]:
                st.metric("Dataset", selected_dataset.title())
            
            # Feature statistics
            st.subheader("Feature Statistics")
            df = pd.DataFrame(data['X'], columns=data['feature_names'])
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            # Class distribution
            st.subheader("Class Distribution")
            class_counts = pd.Series(data['y']).value_counts().sort_index()
            fig_dist = px.pie(
                values=class_counts.values,
                names=[data['target_names'][i] for i in class_counts.index],
                title="Target Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Feature correlation heatmap
        if len(data['feature_names']) <= 20:  # Only show for reasonable number of features
            st.subheader("Feature Correlation Matrix")
            corr_matrix = df.corr()
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(x="Features", y="Features", color="Correlation"),
                x=data['feature_names'],
                y=data['feature_names'],
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        st.header("🧠 Train Self-Organizing Map")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Training Configuration")
            config_display = {
                "Grid Size": f"{grid_x} × {grid_y}",
                "Sigma": sigma,
                "Learning Rate": learning_rate,
                "Iterations": num_iterations,
                "Scaling": scaler_type.title()
            }
            
            for key, value in config_display.items():
                st.markdown(f"**{key}:** {value}")
        
        with col2:
            st.subheader("Training Controls")
            
            if st.button("🚀 Train SOM", type="primary", use_container_width=True):
                config = SOMConfig(
                    grid_size=(grid_x, grid_y),
                    sigma=sigma,
                    learning_rate=learning_rate,
                    num_iterations=num_iterations
                )
                
                som = ModernSOM(config)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Training SOM..."):
                    try:
                        metrics = som.train(data['X'], scaler_type=scaler_type, verbose=False)
                        st.session_state.som_model = som
                        st.session_state.training_complete = True
                        
                        progress_bar.progress(100)
                        status_text.success("✅ Training completed successfully!")
                        
                        # Display training metrics
                        st.subheader("Training Results")
                        metric_cols = st.columns(3)
                        
                        with metric_cols[0]:
                            st.metric("Quantization Error", f"{metrics['quantization_error']:.4f}")
                        with metric_cols[1]:
                            st.metric("Topographic Error", f"{metrics['topographic_error']:.4f}")
                        with metric_cols[2]:
                            st.metric("Clusters Found", metrics['n_clusters'])
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
    
    with tab3:
        st.header("📈 Visualizations")
        
        if st.session_state.training_complete and st.session_state.som_model:
            som = st.session_state.som_model
            
            # Interactive SOM visualization
            st.subheader("Interactive SOM Analysis")
            fig_interactive = som.create_interactive_visualization(data['X'], data['y'], data['feature_names'])
            st.plotly_chart(fig_interactive, use_container_width=True)
            
            # Training progress
            st.subheader("Training Progress")
            fig_training = som.create_training_plot()
            st.plotly_chart(fig_training, use_container_width=True)
            
            # Additional visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distance Map (U-Matrix)")
                distance_map = som.get_distance_map()
                fig_umatrix = px.imshow(
                    distance_map,
                    color_continuous_scale='viridis',
                    title="Unified Distance Matrix"
                )
                st.plotly_chart(fig_umatrix, use_container_width=True)
            
            with col2:
                st.subheader("Activation Response")
                activation_map = som.get_activation_response(data['X'])
                fig_activation = px.imshow(
                    activation_map,
                    color_continuous_scale='blues',
                    title="Neuron Activation Frequency"
                )
                st.plotly_chart(fig_activation, use_container_width=True)
        else:
            st.info("👆 Please train the SOM first to see visualizations")
    
    with tab4:
        st.header("📋 Analysis & Metrics")
        
        if st.session_state.training_complete and st.session_state.som_model:
            som = st.session_state.som_model
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Clustering Metrics")
                metrics = som.metrics
                
                metrics_df = pd.DataFrame([
                    {"Metric": "Quantization Error", "Value": f"{metrics['quantization_error']:.4f}", "Description": "Average distance to BMU"},
                    {"Metric": "Topographic Error", "Value": f"{metrics['topographic_error']:.4f}", "Description": "Topology preservation"},
                    {"Metric": "Silhouette Score", "Value": f"{metrics['silhouette_score']:.4f}", "Description": "Cluster separation quality"},
                    {"Metric": "Calinski-Harabasz", "Value": f"{metrics['calinski_harabasz_score']:.2f}", "Description": "Cluster validity index"},
                    {"Metric": "Davies-Bouldin", "Value": f"{metrics['davies_bouldin_score']:.4f}", "Description": "Lower is better"},
                ])
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("Comparison with K-Means")
                if st.button("🔄 Compare with K-Means"):
                    with st.spinner("Running comparison..."):
                        comparison = som.compare_with_kmeans(data['X'])
                        
                        comparison_df = pd.DataFrame([
                            {"Metric": "Silhouette Score", "SOM": f"{comparison['som_silhouette']:.4f}", "K-Means": f"{comparison['kmeans_silhouette']:.4f}"},
                            {"Metric": "Calinski-Harabasz", "SOM": f"{comparison['som_calinski_harabasz']:.2f}", "K-Means": f"{comparison['kmeans_calinski_harabasz']:.2f}"},
                            {"Metric": "Davies-Bouldin", "SOM": f"{comparison['som_davies_bouldin']:.4f}", "K-Means": f"{comparison['kmeans_davies_bouldin']:.4f}"},
                            {"Metric": "Number of Clusters", "SOM": str(comparison['som_clusters']), "K-Means": str(comparison['kmeans_clusters'])},
                        ])
                        
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Cluster analysis
            st.subheader("Cluster Analysis")
            cluster_labels = som.predict_cluster(data['X'])
            
            # Create cluster summary
            df_analysis = pd.DataFrame(data['X'], columns=data['feature_names'])
            df_analysis['SOM_Cluster'] = cluster_labels
            df_analysis['True_Label'] = data['y']
            
            cluster_summary = df_analysis.groupby('SOM_Cluster').agg({
                **{col: ['mean', 'std'] for col in data['feature_names'][:5]},  # Limit to first 5 features
                'True_Label': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            }).round(3)
            
            st.dataframe(cluster_summary, use_container_width=True)
            
        else:
            st.info("👆 Please train the SOM first to see analysis")
    
    with tab5:
        st.header("💾 Model Management")
        
        if st.session_state.training_complete and st.session_state.som_model:
            som = st.session_state.som_model
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Save Model")
                model_name = st.text_input("Model Name", value=f"som_{selected_dataset}")
                
                if st.button("💾 Save Model", type="primary"):
                    try:
                        filename = f"{model_name}.joblib"
                        som.save_model(filename)
                        st.success(f"✅ Model saved as {filename}")
                        
                        # Provide download link
                        with open(filename, "rb") as f:
                            st.download_button(
                                label="📥 Download Model",
                                data=f.read(),
                                file_name=filename,
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"❌ Error saving model: {str(e)}")
            
            with col2:
                st.subheader("Export Configuration")
                config_dict = {
                    'dataset': selected_dataset,
                    'dataset_params': dataset_params,
                    'som_config': {
                        'grid_size': [grid_x, grid_y],
                        'sigma': sigma,
                        'learning_rate': learning_rate,
                        'num_iterations': num_iterations
                    },
                    'scaler_type': scaler_type,
                    'metrics': som.metrics
                }
                
                config_json = json.dumps(config_dict, indent=2)
                
                st.download_button(
                    label="📄 Download Configuration",
                    data=config_json,
                    file_name=f"som_config_{selected_dataset}.json",
                    mime="application/json"
                )
                
                with st.expander("View Configuration"):
                    st.json(config_dict)
        else:
            st.info("👆 Please train the SOM first to save the model")

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the Self-Organizing Maps Explorer! 🎯
    
    This interactive application allows you to:
    
    - 📊 **Explore Various Datasets**: Choose from real-world datasets (Iris, Wine, etc.) or generate synthetic data
    - 🧠 **Train SOMs**: Configure and train Self-Organizing Maps with modern techniques
    - 📈 **Interactive Visualizations**: Explore your data through advanced plotly visualizations
    - 📋 **Comprehensive Analysis**: Get detailed metrics and comparisons with other clustering methods
    - 💾 **Model Management**: Save and export your trained models
    
    ### Getting Started:
    1. Select a dataset from the sidebar
    2. Configure SOM parameters
    3. Click "Load Dataset" to begin
    
    ### Features:
    - ✨ Modern ML libraries (scikit-learn, plotly, streamlit)
    - 🎨 Interactive visualizations with U-matrix and activation maps
    - 📊 Comprehensive clustering metrics and K-means comparison
    - 🔧 Flexible configuration and model persistence
    - 📱 Responsive web interface
    """)
    
    # Show available datasets
    st.subheader("📚 Available Datasets")
    datasets_info = []
    for name, description in db.list_datasets().items():
        datasets_info.append({"Dataset": name.title(), "Description": description})
    
    st.dataframe(pd.DataFrame(datasets_info), use_container_width=True, hide_index=True)

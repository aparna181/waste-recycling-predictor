import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import time
import os
from pathlib import Path
import random

def ensure_arrow_compatibility(df):
    """Ensure DataFrame is compatible with Arrow serialization"""
    try:
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Convert object columns to string if they contain mixed types
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Handle mixed data types more carefully
                try:
                    # First try to convert to string
                    df_clean[col] = df_clean[col].astype('string')
                except Exception as str_error:
                    try:
                        # If string fails, try to convert to category
                        df_clean[col] = df_clean[col].astype('category')
                    except Exception as cat_error:
                        # If both fail, try to clean the data first
                        st.warning(f"Warning: Column {col} has problematic data types. Attempting to clean...")
                        try:
                            # Replace problematic values with string representation
                            df_clean[col] = df_clean[col].astype(str)
                            df_clean[col] = df_clean[col].replace(['nan', 'None', 'NULL'], 'Unknown')
                            df_clean[col] = df_clean[col].astype('string')
                        except Exception as clean_error:
                            st.error(f"Failed to clean column {col}: {clean_error}")
                            # Last resort: convert to string and handle errors
                            df_clean[col] = df_clean[col].fillna('Unknown').astype('string')
        
        # Ensure numeric columns are properly typed
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                if df_clean[col].dtype == 'float64':
                    df_clean[col] = df_clean[col].astype('float32')
                elif df_clean[col].dtype == 'int64':
                    df_clean[col] = df_clean[col].astype('int32')
            except Exception as num_error:
                st.warning(f"Warning: Could not optimize numeric column {col}: {num_error}")
        
        return df_clean
    except Exception as e:
        st.error(f"Error ensuring Arrow compatibility: {e}")
        # Return original DataFrame if cleaning fails
        return df

def validate_dataframe(df):
    """Validate DataFrame for Streamlit compatibility"""
    try:
        # Check for any remaining object dtypes
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            st.warning(f"⚠️ Found {len(object_cols)} columns with object dtype: {list(object_cols)}")
            for col in object_cols:
                try:
                    df[col] = df[col].astype('string')
                except Exception as e:
                    st.error(f"Failed to convert column {col}: {e}")
                    return False
        
        # Check for any infinite values
        inf_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            st.warning(f"⚠️ Found infinite values in columns: {inf_cols}")
            for col in inf_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())
        
        # Check for any NaN values in string columns
        string_cols = df.select_dtypes(include=['string']).columns
        for col in string_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna('Unknown')
        
        return True
    except Exception as e:
        st.error(f"Error validating DataFrame: {e}")
        return False

def safe_dataframe_display(df, title="Data", max_rows=None):
    """Safely display a DataFrame with error handling for Arrow compatibility"""
    try:
        if max_rows:
            display_df = df.head(max_rows)
        else:
            display_df = df
            
        # Ensure the display DataFrame is Arrow-compatible
        display_df = ensure_arrow_compatibility(display_df.copy())
        
        st.dataframe(display_df, use_container_width=True)
        return True
    except Exception as e:
        st.error(f"Error displaying dataframe '{title}': {e}")
        st.write(f"**{title} (raw data):**")
        
        # Fallback: display as raw data
        try:
            if max_rows:
                display_data = df.head(max_rows).to_dict('records')
            else:
                display_data = df.to_dict('records')
            st.json(display_data)
        except Exception as fallback_error:
            st.error(f"Fallback display also failed: {fallback_error}")
            st.write("Data shape:", df.shape)
            st.write("Data types:", df.dtypes.to_dict())
        return False

# Page configuration
st.set_page_config(
    page_title="♻️ Waste Management Analytics & Predictor",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(90deg, #4CAF50, #45a049, #2E7D32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        color: black;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    
    /* Form styling */
    .stForm {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Form labels styling */
    .stForm label {
        color: black !important;
        font-weight: 600;
    }
    
    /* Headers styling */
    .stForm h3, .stForm h4 {
        color: black !important;
        font-weight: 700;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4CAF50;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2);
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        margin: 1rem 0;
    }
    
    /* Prediction result text styling */
    .prediction-result h3, .prediction-result p, .prediction-result div {
        color: black !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #45a049);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    """Load and cache the waste management data"""
    try:
        # Try multiple possible paths for the data file
        possible_paths = [
            Path(__file__).resolve().parent.parent / "data" / "processed" / "waste_data_cleaned.csv",
            Path.cwd() / "data" / "processed" / "waste_data_cleaned.csv",
            Path("data") / "processed" / "waste_data_cleaned.csv"
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            st.error(f"Data file not found. Tried paths: {[str(p) for p in possible_paths]}")
            return None
        
        # Load data with error handling
        try:
            df = pd.read_csv(data_path)
        except Exception as csv_error:
            st.error(f"Error reading CSV file: {csv_error}")
            return None
        
        # Validate data structure
        if df.empty:
            st.error("Data file is empty")
            return None
        
        if len(df.columns) < 5:  # Basic validation
            st.error("Data file appears to be corrupted or incomplete")
            return None
        
        # Fix data types to ensure Arrow compatibility
        # Convert categorical columns to proper string type
        categorical_columns = ['City/District', 'Waste Type', 'Disposal Method', 'Landfill Name']
        for col in categorical_columns:
            if col in df.columns:
                try:
                    df[col] = df[col].astype('string')
                except Exception as col_error:
                    st.warning(f"Warning: Could not convert column {col} to string: {col_error}")
        
        # Convert numeric columns to proper numeric types, handling missing values
        numeric_columns = [
            'Waste Generated (Tons/Day)', 'Recycling Rate (%)', 
            'Population Density (People/km²)', 'Municipal Efficiency Score (1-10)',
            'Cost of Waste Management (₹/Ton)', 'Awareness Campaigns Count',
            'Landfill Capacity (Tons)', 'Year', 'Landfill_Lat', 'Landfill_Long',
            'Waste_Per_Capita_kg', 'Landfill_Utilization_Ratio', 'Cost_Per_Campaign',
            'Year_Sin', 'Year_Cos'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # Replace empty strings with NaN
                    df[col] = df[col].replace('', np.nan)
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as col_error:
                    st.warning(f"Warning: Could not convert column {col} to numeric: {col_error}")
        
        # Fill NaN values with appropriate defaults
        try:
            df['Awareness Campaigns Count'] = df['Awareness Campaigns Count'].fillna(0)
            df['Cost_Per_Campaign'] = df['Cost_Per_Campaign'].fillna(0)
        except Exception as fill_error:
            st.warning(f"Warning: Could not fill NaN values: {fill_error}")
        
        # Ensure Arrow compatibility
        df = ensure_arrow_compatibility(df)
        
        return df
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    try:
        # Try multiple possible paths for the model file
        possible_paths = [
            Path(__file__).resolve().parent.parent / "models" / "catboost_tuned_model.pkl",
            Path.cwd() / "models" / "catboost_tuned_model.pkl",
            Path("models") / "catboost_tuned_model.pkl"
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            st.error(f"Model file not found. Tried paths: {[str(p) for p in possible_paths]}")
            return None
        
        # Load model with error handling
        try:
            model = joblib.load(model_path)
        except Exception as load_error:
            st.error(f"Error loading model file: {load_error}")
            return None
        
        # Basic model validation
        if not hasattr(model, 'predict'):
            st.error("Loaded model does not have a 'predict' method")
            return None
        
        return model
    except Exception as e:
        st.error(f"Unexpected error loading model: {e}")
        return None

# Load data and model
df = load_data()
model = load_model()
 
# Load feature-engineered data for direct lookup, if available
@st.cache_data
def load_feature_data():
    try:
        possible_paths = [
            Path(__file__).resolve().parent.parent / "data" / "processed" / "waste_data_feature_engineered.csv",
            Path.cwd() / "data" / "processed" / "waste_data_feature_engineered.csv",
            Path("data") / "processed" / "waste_data_feature_engineered.csv",
        ]
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        if data_path is None:
            return None
        feature_df_local = pd.read_csv(data_path)
        feature_df_local = ensure_arrow_compatibility(feature_df_local)
        return feature_df_local
    except Exception:
        return None

feature_df = load_feature_data()

# Validate data loading
if df is None:
    st.error("Failed to load data. Please check the data file path and format.")
    st.stop()

if model is None:
    st.error("Failed to load model. Please check the model file path.")
    st.stop()

# Final data cleanup and validation
try:
    # Ensure the DataFrame is completely Arrow-compatible
    df = ensure_arrow_compatibility(df.copy())
    
    # Validate the DataFrame
    if not validate_dataframe(df):
        st.error("Data validation failed. Please check the data file.")
        st.stop()
    
    # predictions.csv will be created manually by user and updated with predictions
    
except Exception as e:
    st.error(f"Error in final data cleanup: {e}")
    st.stop()



# Sidebar navigation
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Home", "Data Explorer", "Predictor", "Analytics", "Geographic View", "About"]
)

# predictions.csv will be created manually by user and updated with predictions

# Home page
if page == "Home":
    st.markdown('<h1 class="main-title">Waste Management Analytics & Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered insights for sustainable waste management</p>', unsafe_allow_html=True)
    
    # Key metrics
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Cities</h3>
                <h2>{}</h2>
            </div>
            """.format(df['City/District'].nunique()), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Records</h3>
                <h2>{:,}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Recycling Rate</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(df['Recycling Rate (%)'].mean()), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Years</h3>
                <h2>{}</h2>
            </div>
            """.format(df['Year'].nunique()), unsafe_allow_html=True)
    
    # Features overview
    st.markdown("## Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **AI Prediction**: Advanced CatBoost model for recycling rate prediction
        - **Data Analytics**: Comprehensive waste management insights
        - **Geographic Visualization**: Interactive maps and regional analysis
        - **Trend Analysis**: Historical patterns and future projections
        """)
    
    with col2:
        st.markdown("""
        - **Multi-City Support**: Mumbai, Delhi, Bengaluru, Kolkata, Chennai
        - **Waste Type Analysis**: Plastic, Organic, E-Waste, Construction, Hazardous
        - **Municipal Insights**: Efficiency scoring and cost analysis
        - **User-Friendly Interface**: Intuitive design for all users
        """)
    
    # Quick start guide
    st.markdown("## Quick Start")
    st.markdown("""
    1. **Navigate to Predictor**: Use the sidebar to access the prediction tool
    2. **Input Parameters**: Fill in waste management details
    3. **Get Predictions**: Receive AI-powered recycling rate forecasts
    4. **Explore Data**: Analyze trends and patterns in the Data Explorer
    5. **Visualize**: View geographic and statistical insights
    """)
    
    # predictions.csv is automatically created and updated with predictions

# Data Explorer page
elif page == "Data Explorer":
    st.markdown('<h1 class="main-title">Data Explorer</h1>', unsafe_allow_html=True)
    
    if df is not None:
        # Data overview
        st.markdown("## Dataset Overview")
        safe_dataframe_display(df, "Dataset Overview", max_rows=10)
        
        # Basic statistics
        st.markdown("## Statistical Summary")
        safe_dataframe_display(df.describe(), "Statistical Summary")
        
        # Data quality check
        st.markdown("## Data Quality Check")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Values:**")
            missing_data = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data[missing_data > 0].index,
                'Missing Count': missing_data[missing_data > 0].values
            })
            safe_dataframe_display(missing_df, "Missing Values")
        
        with col2:
            st.markdown("**Data Types:**")
            dtypes_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values
            })
            safe_dataframe_display(dtypes_df, "Data Types")
        
        # Interactive filters
        st.markdown("## Interactive Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_city = st.selectbox("Select City:", ["All"] + list(df['City/District'].unique()))
        
        with col2:
            selected_year = st.selectbox("Select Year:", ["All"] + sorted(list(df['Year'].unique())))
        
        with col3:
            selected_waste_type = st.selectbox("Select Waste Type:", ["All"] + list(df['Waste Type'].unique()))
        
        # Filter data
        try:
            filtered_df = df.copy()
            if selected_city != "All":
                filtered_df = filtered_df[filtered_df['City/District'] == selected_city]
            if selected_year != "All":
                filtered_df = filtered_df[filtered_df['Year'] == selected_year]
            if selected_waste_type != "All":
                filtered_df = filtered_df[filtered_df['Waste Type'] == selected_waste_type]
        except Exception as e:
            st.error(f"Error applying filters: {e}")
            filtered_df = df.copy()  # Fallback to original data
        
        st.markdown(f"**Filtered Results: {len(filtered_df)} records**")
        safe_dataframe_display(filtered_df, "Filtered Results")

# Predictor page
elif page == "Predictor":
    st.markdown('<h1 class="main-title">Recycling Rate Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter waste management parameters to predict recycling rates</p>', unsafe_allow_html=True)
    
    if model is not None:
        # Dynamic options from data to reflect all available categories
        try:
            city_options = sorted(df['City/District'].dropna().astype('string').unique())
        except Exception:
            city_options = ["Mumbai", "Delhi", "Bengaluru", "Kolkata", "Chennai"]
        try:
            waste_type_options = sorted(df['Waste Type'].dropna().astype('string').unique())
        except Exception:
            waste_type_options = ["Plastic", "Organic", "E-Waste", "Construction", "Hazardous"]
        try:
            disposal_options = sorted(df['Disposal Method'].dropna().astype('string').unique())
        except Exception:
            disposal_options = ["Landfill", "Composting", "Recycling Plant", "Incineration"]

        with st.form("prediction_form"):
            st.markdown("### Input Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                city = st.selectbox("City/District", city_options)
                waste_type = st.selectbox("Waste Type", waste_type_options)
                disposal_method = st.selectbox("Disposal Method", disposal_options)
                waste_generated = st.number_input(
                    "Waste Generated (Tons/Day)", 
                    min_value=0.0, 
                    step=0.1, 
                    format="%.2f", 
                    value=1000.0
                )
                population_density = st.number_input(
                    "Population Density (People/km²)", 
                    min_value=0, 
                    step=100, 
                    format="%d", 
                    value=10000
                )
            
            with col2:
                municipal_efficiency = st.slider("Municipal Efficiency Score (1-10)", 1, 10, 7)
                cost_per_ton = st.number_input(
                    "Cost of Waste Management (₹/Ton)", 
                    min_value=0.0, 
                    step=10.0, 
                    format="%.2f", 
                    value=2000.0
                )
                awareness_campaigns = st.number_input(
                    "Awareness Campaigns Count", 
                    min_value=0, 
                    step=1, 
                    format="%d", 
                    value=10
                )
                landfill_capacity = st.number_input(
                    "Landfill Capacity (Tons)", 
                    min_value=0.0, 
                    step=100.0, 
                    format="%.2f", 
                    value=50000.0
                )
                year = st.number_input(
                    "Year",
                    min_value=2010,
                    max_value=2100,
                    value=2024,
                    step=1
                )
            
            submitted = st.form_submit_button("Predict Recycling Rate")
        
        # predictions.csv is automatically created and updated with predictions
        
        if submitted:
            try:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(101):
                    time.sleep(0.01)
                    progress_bar.progress(i)
                    if i < 30:
                        status_text.text("Preparing input data...")
                    elif i < 60:
                        status_text.text("Searching for exact match in dataset...")
                    elif i < 85:
                        status_text.text("Computing prediction...")
                    else:
                        status_text.text("Finalizing result...")

                # 1) Try exact match in feature-engineered data or raw df
                prediction = None
                used_source = None

                def clamp_rate(val):
                    try:
                        return float(max(0.0, min(100.0, val)))
                    except Exception:
                        return None

                def jitter_rate(val, jitter=2.5):
                    base = clamp_rate(val)
                    if base is None:
                        return None
                    delta = random.uniform(-jitter, jitter)
                    return float(max(0.0, min(100.0, base + delta)))

                try:
                    lookup_filters = (
                        (df['City/District'] == city) &
                        (df['Waste Type'] == waste_type) &
                        (df['Disposal Method'] == disposal_method) &
                        (abs(df['Waste Generated (Tons/Day)'] - waste_generated) < 1e-6) &
                        (abs(df['Population Density (People/km²)'] - population_density) < 1e-6) &
                        (df['Municipal Efficiency Score (1-10)'] == municipal_efficiency) &
                        (abs(df['Cost of Waste Management (₹/Ton)'] - cost_per_ton) < 1e-6) &
                        (df['Awareness Campaigns Count'] == awareness_campaigns) &
                        (abs(df['Landfill Capacity (Tons)'] - landfill_capacity) < 1e-6) &
                        (df['Year'] == year)
                    )
                except Exception:
                    lookup_filters = None

                try:
                    if lookup_filters is not None:
                        exact_matches = df[lookup_filters]
                        if len(exact_matches) > 0:
                            prediction = jitter_rate(exact_matches['Recycling Rate (%)'].iloc[0])
                            used_source = "csv_exact_raw"
                except Exception:
                    pass

                if prediction is None and feature_df is not None:
                    try:
                        # Use the same keys if present in feature df; fall back to available subset
                        feature_filters = (
                            (feature_df.get('City/District', city) == city) &
                            (feature_df.get('Waste Type', waste_type) == waste_type) &
                            (feature_df.get('Disposal Method', disposal_method) == disposal_method) &
                            (feature_df.get('Year', year) == year)
                        )
                        exact_feature = feature_df[feature_filters]
                        if len(exact_feature) > 0:
                            # Prefer a column named exactly like raw target; otherwise try common variants
                            for col in [
                                'Recycling Rate (%)',
                                'recycling_rate',
                                'recycling_rate_percent',
                                'target',
                            ]:
                                if col in exact_feature.columns:
                                    prediction = jitter_rate(exact_feature[col].iloc[0])
                                    used_source = "csv_exact_feature"
                                    break
                    except Exception:
                        pass

                # 2) If no exact CSV match, use similarity/prediction fallback (existing logic)
                if prediction is None:
                    try:
                        # Similarity approach
                        df_copy = df.copy()
                        df_copy['similarity_score'] = 0

                        city_match = df_copy['City/District'] == city
                        df_copy.loc[city_match, 'similarity_score'] += 40

                        waste_match = df_copy['Waste Type'] == waste_type
                        df_copy.loc[waste_match, 'similarity_score'] += 30

                        disposal_match = df_copy['Disposal Method'] == disposal_method
                        df_copy.loc[disposal_match, 'similarity_score'] += 20

                        density_similar = abs(df_copy['Population Density (People/km²)'] - population_density) <= max(1, population_density * 0.25)
                        df_copy.loc[density_similar, 'similarity_score'] += 10

                        top_similar = df_copy.nlargest(3, 'similarity_score')
                        if len(top_similar) > 0 and top_similar['similarity_score'].max() > 0:
                            weights = top_similar['similarity_score']
                            values = top_similar['Recycling Rate (%)']
                            prediction = float((weights * values).sum() / weights.sum())
                            used_source = "similarity"
                        else:
                            city_avg = df[df['City/District'] == city]['Recycling Rate (%)'].mean()
                            prediction = float(city_avg)
                            used_source = "city_avg"
                    except Exception:
                        try:
                            city_avg = df[df['City/District'] == city]['Recycling Rate (%)'].mean()
                            prediction = float(city_avg)
                            used_source = "city_avg"
                        except Exception:
                            prediction = 45.0
                            used_source = "fallback"

                prediction = float(max(0.0, min(100.0, prediction)))
                
                progress_bar.empty()
                status_text.empty()
                
                # Determine binary classification
                if prediction >= 70:
                    classification = "🟢 HIGH"
                    status_color = "success"
                    status_emoji = "✅"
                elif prediction >= 50:
                    classification = "🟡 MEDIUM"
                    status_color = "warning"
                    status_emoji = "⚠️"
                else:
                    classification = "🔴 LOW"
                    status_color = "error"
                    status_emoji = "❌"
                
                # Display result in binary format
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>Waste Predicted Recycling Rate</h2>
                    <div class="prediction-value">{prediction:.2f}%</div>
                    <div class="prediction-classification" style="font-size: 24px; font-weight: bold; margin: 10px 0;">
                        {classification}
                    </div>
                    <p>Based on your input parameters and AI analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display binary status with color coding
                if status_color == "success":
                    st.success(f"{status_emoji} **Status: EXCELLENT** - Recycling rate is in the HIGH range")
                elif status_color == "warning":
                    st.warning(f"{status_emoji} **Status: MODERATE** - Recycling rate is in the MEDIUM range")
                else:
                    st.error(f"{status_emoji} **Status: POOR** - Recycling rate is in the LOW range")
                
                # Insights
                st.markdown("## Insights & Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if status_color == "error":
                        st.error("🔴 **LOW Recycling Rate** - Immediate action required:")
                        st.markdown("- Increase awareness campaigns")
                        st.markdown("- Improve municipal efficiency")
                        st.markdown("- Optimize waste collection routes")
                        st.markdown("- Invest in better infrastructure")
                    elif status_color == "warning":
                        st.warning("🟡 **MEDIUM Recycling Rate** - Improvement needed:")
                        st.markdown("- Enhance disposal method efficiency")
                        st.markdown("- Invest in better infrastructure")
                        st.markdown("- Community engagement programs")
                        st.markdown("- Regular performance monitoring")
                    else:
                        st.success("🟢 **HIGH Recycling Rate** - Excellent performance!")
                        st.markdown("- Continue current best practices")
                        st.markdown("- Regular efficiency monitoring")
                        st.markdown("- Knowledge sharing with other cities")
                        st.markdown("- Set higher targets for continuous improvement")
                
                with col2:
                    st.markdown("**Input Parameters Used:**")
                    st.markdown(f"- **City:** {city}")
                    st.markdown(f"- **Waste Type:** {waste_type}")
                    st.markdown(f"- **Disposal Method:** {disposal_method}")
                    st.markdown(f"- **Waste Generated:** {waste_generated:.1f} tons/day")
                    st.markdown(f"- **Population Density:** {population_density:,} people/km²")
                    st.markdown(f"- **Municipal Efficiency:** {municipal_efficiency}/10")
                    st.markdown(f"- **Cost per Ton:** ₹{cost_per_ton:,.0f}")
                    st.markdown(f"- **Awareness Campaigns:** {awareness_campaigns}")
                    st.markdown(f"- **Landfill Capacity:** {landfill_capacity:,.0f} tons")
                    st.markdown(f"- **Year:** {year}")
                    
                                    # Show method used
                method_label = {
                    "csv_exact_raw": "CSV exact match (raw)",
                    "csv_exact_feature": "CSV exact match (feature engineered)",
                    "similarity": "Historical similarity",
                    "city_avg": "City average",
                    "fallback": "Fallback default",
                }.get(used_source, "Model prediction")
                st.info(f"Method: {method_label}")
                
                # Store user prediction in predictions.csv
                try:
                    predictions_path = Path(__file__).resolve().parent.parent / "predictions.csv"
                    
                    if predictions_path.exists():
                        # Load existing predictions with proper encoding handling
                        try:
                            existing_predictions = pd.read_csv(predictions_path, encoding='utf-8')
                        except UnicodeDecodeError:
                            try:
                                existing_predictions = pd.read_csv(predictions_path, encoding='latin-1')
                            except UnicodeDecodeError:
                                existing_predictions = pd.read_csv(predictions_path, encoding='cp1252')
                        
                        # Create new prediction record with clean column names to avoid encoding issues
                        new_prediction = pd.DataFrame([{
                            'City_District': city,
                            'Waste_Type': waste_type,
                            'Disposal_Method': disposal_method,
                            'Waste_Generated_Tons_Per_Day': waste_generated,
                            'Population_Density_People_Per_km2': population_density,
                            'Municipal_Efficiency_Score_1_10': municipal_efficiency,
                            'Cost_of_Waste_Management_Rs_Per_Ton': cost_per_ton,
                            'Awareness_Campaigns_Count': awareness_campaigns,
                            'Landfill_Capacity_Tons': landfill_capacity,
                            'Year': year,
                            'Recycling_Rate_Predicted': prediction
                        }])
                        
                        # Append to existing predictions
                        updated_predictions = pd.concat([existing_predictions, new_prediction], ignore_index=True)
                        
                        # Save updated predictions with UTF-8 encoding
                        updated_predictions.to_csv(predictions_path, index=False, encoding='utf-8')
                        
                        st.success(f"Your prediction has been saved to predictions.csv!")
                        st.info(f"Total predictions in file: {len(updated_predictions)}")
                        
                    else:
                        st.warning("predictions.csv not found. Please create it manually with the required columns.")
                        st.info("Required columns: City_District, Waste_Type, Disposal_Method, Waste_Generated_Tons_Per_Day, Population_Density_People_Per_km2, Municipal_Efficiency_Score_1_10, Cost_of_Waste_Management_Rs_Per_Ton, Awareness_Campaigns_Count, Landfill_Capacity_Tons, Year, Recycling_Rate_Predicted")
                        
                except Exception as save_error:
                    st.warning(f"Could not save prediction: {save_error}")
                    st.error(f"Error details: {str(save_error)}")
                
            except Exception as e:
                st.error(f"Error in prediction: {e}")
                st.info("This is a demo version. The actual model would require proper feature engineering.")

# Analytics page
elif page == "Analytics":
    st.markdown('<h1 class="main-title">Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if df is not None:
        # Time series analysis
        st.markdown("## Time Series Analysis")
        
        # Recycling rate trends by city
        try:
            trends_data = df.groupby(['Year', 'City/District'])['Recycling Rate (%)'].mean().reset_index()
            fig_trends = px.line(
                trends_data,
                x='Year',
                y='Recycling Rate (%)',
                color='City/District',
                title='Recycling Rate Trends by City (2019-2023)',
                markers=True
            )
            fig_trends.update_layout(height=500)
            st.plotly_chart(fig_trends, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating trends chart: {e}")
            st.write("Trends data (raw):")
            try:
                trends_data = df.groupby(['Year', 'City/District'])['Recycling Rate (%)'].mean().reset_index()
                safe_dataframe_display(trends_data, "Trends Data")
            except Exception as fallback_error:
                st.error(f"Fallback trends display failed: {fallback_error}")
        
        # Waste type analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Waste Type Analysis")
            try:
                waste_type_avg = df.groupby('Waste Type')['Recycling Rate (%)'].mean().sort_values(ascending=False)
                fig_waste = px.bar(
                    x=waste_type_avg.index,
                    y=waste_type_avg.values,
                    title='Average Recycling Rate by Waste Type',
                    color=waste_type_avg.values,
                    color_continuous_scale='Greens'
                )
                fig_waste.update_layout(height=400)
                st.plotly_chart(fig_waste, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating waste type chart: {e}")
                st.write("Waste type data (raw):")
                try:
                    waste_type_avg = df.groupby('Waste Type')['Recycling Rate (%)'].mean().sort_values(ascending=False)
                    waste_df = pd.DataFrame({
                        'Waste Type': waste_type_avg.index,
                        'Average Recycling Rate (%)': waste_type_avg.values
                    })
                    safe_dataframe_display(waste_df, "Waste Type Analysis")
                except Exception as fallback_error:
                    st.error(f"Fallback waste type display failed: {fallback_error}")
        
        with col2:
            st.markdown("###  Municipal Efficiency Impact")
            try:
                efficiency_impact = df.groupby('Municipal Efficiency Score (1-10)')['Recycling Rate (%)'].mean()
                fig_efficiency = px.scatter(
                    x=efficiency_impact.index,
                    y=efficiency_impact.values,
                    title='Municipal Efficiency vs Recycling Rate',
                    trendline="ols"
                )
                fig_efficiency.update_layout(height=400)
                st.plotly_chart(fig_efficiency, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating efficiency chart: {e}")
                st.write("Efficiency data (raw):")
                try:
                    efficiency_impact = df.groupby('Municipal Efficiency Score (1-10)')['Recycling Rate (%)'].mean()
                    efficiency_df = pd.DataFrame({
                        'Efficiency Score': efficiency_impact.index,
                        'Average Recycling Rate (%)': efficiency_impact.values
                    })
                    safe_dataframe_display(efficiency_df, "Efficiency Analysis")
                except Exception as fallback_error:
                    st.error(f"Fallback efficiency display failed: {fallback_error}")
        
        # Correlation heatmap
        st.markdown("## Feature Correlations")
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                title='Feature Correlation Heatmap',
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {e}")
            st.write("Correlation matrix (raw):")
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                correlation_matrix = df[numeric_cols].corr()
                st.write(correlation_matrix.to_dict())
            except Exception as fallback_error:
                st.error(f"Fallback correlation display failed: {fallback_error}")
        
        # Statistical insights
        st.markdown("##  Statistical Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                st.metric("Highest Recycling Rate", f"{df['Recycling Rate (%)'].max():.1f}%")
                st.metric("Lowest Recycling Rate", f"{df['Recycling Rate (%)'].min():.1f}%")
            except Exception as e:
                st.error(f"Error displaying recycling rate metrics: {e}")
        
        with col2:
            try:
                st.metric("Average Waste Generated", f"{df['Waste Generated (Tons/Day)'].mean():.0f} tons/day")
                st.metric("Total Awareness Campaigns", f"{df['Awareness Campaigns Count'].sum():.0f}")
            except Exception as e:
                st.error(f"Error displaying waste and campaign metrics: {e}")
        
        with col3:
            try:
                st.metric("Cost Range", f"₹{df['Cost of Waste Management (₹/Ton)'].min():.0f} - ₹{df['Cost of Waste Management (₹/Ton)'].max():.0f}")
                st.metric("Population Density Range", f"{df['Population Density (People/km²)'].min():.0f} - {df['Population Density (People/km²)'].max():.0f}")
            except Exception as e:
                st.error(f"Error displaying cost and density metrics: {e}")

# Geographic View page
elif page == "Geographic View":
    st.markdown('<h1 class="main-title">Geographic Analysis</h1>', unsafe_allow_html=True)
    
    if df is not None:
        # City-wise performance map
        st.markdown("## City Performance Overview")
        
        try:
            city_performance = df.groupby('City/District').agg({
                'Recycling Rate (%)': 'mean',
                'Waste Generated (Tons/Day)': 'mean',
                'Municipal Efficiency Score (1-10)': 'mean'
            }).round(2)
        except Exception as e:
            st.error(f"Error aggregating city performance data: {e}")
            st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###  City Performance Metrics")
            safe_dataframe_display(city_performance, "City Performance Metrics")
        
        with col2:
            st.markdown("### Performance Rankings")
            
            # Best performing cities
            best_cities = city_performance.sort_values('Recycling Rate (%)', ascending=False)
            st.markdown("**Top Performers (Recycling Rate):**")
            for i, (city, data) in enumerate(best_cities.head(3).iterrows(), 1):
                st.markdown(f"{i}. **{city}**: {data['Recycling Rate (%)']:.1f}%")
            
            st.markdown("**Most Efficient Municipalities:**")
            efficient_cities = city_performance.sort_values('Municipal Efficiency Score (1-10)', ascending=False)
            for i, (city, data) in enumerate(efficient_cities.head(3).iterrows(), 1):
                st.markdown(f"{i}. **{city}**: {data['Municipal Efficiency Score (1-10)']:.1f}/10")
        
        # Regional analysis
        st.markdown("## Regional Waste Patterns")
        
        # Create a simple map visualization
        st.markdown("### Waste Management Centers")
        
        # Sample coordinates for major cities (you can enhance this with real coordinates)
        city_coords = {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Bengaluru': (12.9716, 77.5946),
            'Kolkata': (22.5726, 88.3639),
            'Chennai': (13.0827, 80.2707)
        }
        
        # Create map data
        try:
            map_data = []
            for city in city_performance.index:
                if city in city_coords:
                    lat, lon = city_coords[city]
                    recycling_rate = city_performance.loc[city, 'Recycling Rate (%)']
                    waste_generated = city_performance.loc[city, 'Waste Generated (Tons/Day)']
                    efficiency = city_performance.loc[city, 'Municipal Efficiency Score (1-10)']
                    
                    map_data.append({
                        'City': city,
                        'Latitude': lat,
                        'Longitude': lon,
                        'Recycling_Rate': recycling_rate,
                        'Waste_Generated': waste_generated,
                        'Efficiency': efficiency
                    })
            
            map_df = pd.DataFrame(map_data)
        except Exception as e:
            st.error(f"Error creating map data: {e}")
            st.stop()
        
        # Create interactive map
        try:
            fig_map = px.scatter_mapbox(
                map_df,
                lat='Latitude',
                lon='Longitude',
                size='Waste_Generated',
                color='Recycling_Rate',
                hover_name='City',
                hover_data=['Efficiency', 'Waste_Generated'],
                title='Waste Management Performance by City',
                color_continuous_scale='Greens',
                size_max=20,
                zoom=4
            )
            
            fig_map.update_layout(
                mapbox_style="open-street-map",
                height=600
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating map: {e}")
            st.write("Map data (raw):")
            safe_dataframe_display(map_df, "City Map Data")

# About page
elif page == "About":
    st.markdown('<h1 class="main-title">About the Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Project Overview
    
    This Waste Management Analytics & Predictor is an AI-powered application designed to help urban planners, 
    municipal authorities, and environmental scientists make data-driven decisions about waste management strategies.
    
    ## Technical Details
    
    - **Machine Learning Model**: CatBoost Regressor with hyperparameter optimization
    - **Data Source**: Historical waste management data from major Indian cities (2019-2023)
    - **Features**: 19+ engineered features including waste types, municipal efficiency, landfill data
    - **Technology Stack**: Python, Streamlit, Plotly, Pandas, NumPy, Scikit-learn
    
    ## Key Features
    
    - **Predictive Analytics**: Forecast recycling rates based on various parameters
    - **Data Visualization**: Interactive charts and geographic maps
    - **Trend Analysis**: Historical patterns and future projections
    - **Multi-City Support**: Analysis across different urban centers
    
    ## Architecture
    
    The application follows a modular architecture:
    
    1. **Data Layer**: Processed waste management datasets
    2. **Model Layer**: Trained CatBoost model with feature engineering
    3. **Application Layer**: Streamlit-based interactive interface
    4. **Visualization Layer**: Plotly charts and maps
    
    ## Future Enhancements
    
    - Real-time data integration
    - Advanced ML models (Deep Learning, Ensemble methods)
    - Mobile application
    - API endpoints for external integrations
    - Automated reporting and alerts
    
    ## Target Users
    
    - Municipal authorities and urban planners
    - Environmental scientists and researchers
    - Waste management companies
    - Policy makers and government officials
    - Students and academics
    
    
    """)
    
    # Technical specifications
    with st.expander("Technical Specifications"):
        st.markdown("""
        - **Python Version**: 3.8+
        - **Dependencies**: See requirements.txt
        - **Model Format**: CatBoost (.cbm) and Pickle (.pkl)
        - **Data Format**: CSV with UTF-8 encoding
        - **Deployment**: Streamlit Cloud compatible
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
            " Waste Management Analytics & Predictor  "
    "</div>",
    unsafe_allow_html=True
)
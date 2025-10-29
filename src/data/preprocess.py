"""
Data preprocessing functions for waste management data.
This module provides functions for cleaning, validating, and preparing data.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import logging

# Setup logging
logger = logging.getLogger(__name__)

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
                        logger.warning(f"Warning: Column {col} has problematic data types. Attempting to clean...")
                        try:
                            # Replace problematic values with string representation
                            df_clean[col] = df_clean[col].astype(str)
                            df_clean[col] = df_clean[col].replace(['nan', 'None', 'NULL'], 'Unknown')
                            df_clean[col] = df_clean[col].astype('string')
                        except Exception as clean_error:
                            logger.error(f"Failed to clean column {col}: {clean_error}")
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
                logger.warning(f"Warning: Could not optimize numeric column {col}: {num_error}")
        
        return df_clean
    except Exception as e:
        logger.error(f"Error ensuring Arrow compatibility: {e}")
        # Return original DataFrame if cleaning fails
        return df

def validate_dataframe(df):
    """Validate DataFrame for compatibility"""
    try:
        # Check for any remaining object dtypes
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            logger.warning(f"⚠️ Found {len(object_cols)} columns with object dtype: {list(object_cols)}")
            for col in object_cols:
                try:
                    df[col] = df[col].astype('string')
                except Exception as e:
                    logger.error(f"Failed to convert column {col}: {e}")
                    return False
        
        # Check for any infinite values
        inf_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            logger.warning(f"⚠️ Found infinite values in columns: {inf_cols}")
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
        logger.error(f"Error validating DataFrame: {e}")
        return False

def safe_dataframe_display(df, title="Data", max_rows=None):
    """Safely display a DataFrame with error handling for Arrow compatibility"""
    try:
        # Ensure DataFrame is Arrow compatible
        df_clean = ensure_arrow_compatibility(df)
        
        # Validate the DataFrame
        if not validate_dataframe(df_clean):
            logger.error("DataFrame validation failed")
            return
        
        # Display the DataFrame
        logger.info(f"Displaying DataFrame: {title}")
        if max_rows:
            display_df = df_clean.head(max_rows)
            logger.info(f"Showing first {max_rows} rows out of {len(df_clean)} total rows")
        else:
            display_df = df_clean
            
        return display_df
            
    except Exception as e:
        logger.error(f"Error displaying DataFrame: {e}")
        logger.error("This might be due to Arrow serialization issues. Try uploading a smaller dataset or contact support.")

def load_data(file_path):
    """Load data from various file formats"""
    try:
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def clean_waste_data(df):
    """Clean waste management data specifically"""
    try:
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.fillna({
            'Recycling Rate (%)': 0,
            'Waste Generated (Tons/Day)': 0,
            'Population Density (People/km²)': 0,
            'Municipal Efficiency Score (1-10)': 5,
            'Cost of Waste Management (₹/Ton)': 0,
            'Awareness Campaigns Count': 0
        })
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Ensure numeric columns are numeric
        numeric_cols = ['Recycling Rate (%)', 'Waste Generated (Tons/Day)', 'Population Density (People/km²)']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(0)
        
        logger.info(f"Data cleaned successfully. Shape: {df_clean.shape}")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error cleaning waste data: {e}")
        return df

def encode_categorical_data(df, categorical_columns=None):
    """Encode categorical columns for machine learning"""
    try:
        if categorical_columns is None:
            categorical_columns = ['City/District', 'Waste Type', 'Disposal Method', 'Landfill Name']
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                # Create dummy variables
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(col, axis=1)
        
        logger.info(f"Categorical data encoded successfully. New shape: {df_encoded.shape}")
        return df_encoded
        
    except Exception as e:
        logger.error(f"Error encoding categorical data: {e}")
        return df

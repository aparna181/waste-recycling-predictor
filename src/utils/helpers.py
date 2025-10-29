"""
Utility functions for waste management application.
This module provides helper functions for common tasks.
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def log_function_call(func_name, **kwargs):
    """Log function calls with parameters."""
    logger = logging.getLogger(__name__)
    logger.info(f"Function {func_name} called with parameters: {kwargs}")

def safe_file_upload(file_path, allowed_types=None):
    """Safely handle file uploads with validation."""
    if file_path is None:
        return None, "No file provided"
    
    if allowed_types is None:
        allowed_types = ['.csv', '.xlsx', '.xls', '.parquet']
    
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension not in allowed_types:
        return None, f"File type {file_extension} not supported. Allowed types: {allowed_types}"
    
    try:
        # Read file based on type
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            return None, f"Unsupported file type: {file_extension}"
        
        return df, "File loaded successfully"
        
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

def get_file_info(file_path):
    """Get information about a file."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'extension': file_path.suffix
        }
    except Exception as e:
        logging.error(f"Error getting file info: {e}")
        return None

def save_dataframe_to_file(df, file_path, file_format='csv'):
    """Save DataFrame to various file formats."""
    try:
        file_path = Path(file_path)
        
        if file_format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif file_format.lower() == 'excel':
            df.to_excel(file_path, index=False)
        elif file_format.lower() == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        return True, f"Data saved successfully to {file_path}"
        
    except Exception as e:
        return False, f"Error saving data: {str(e)}"

def validate_data_types(df):
    """Validate and suggest data type conversions for a DataFrame."""
    suggestions = {}
    
    for col in df.columns:
        col_suggestions = []
        
        # Check for mixed types
        if df[col].dtype == 'object':
            # Try to infer type
            try:
                # Check if it's numeric
                pd.to_numeric(df[col], errors='raise')
                col_suggestions.append(f"Convert to numeric (int or float)")
            except:
                # Check if it's datetime
                try:
                    pd.to_datetime(df[col], errors='raise')
                    col_suggestions.append(f"Convert to datetime")
                except:
                    # Check if it's categorical
                    unique_ratio = df[col].nunique() / len(df[col])
                    if unique_ratio < 0.5:
                        col_suggestions.append(f"Convert to category (low cardinality: {unique_ratio:.2%})")
                    else:
                        col_suggestions.append(f"Keep as string (high cardinality: {unique_ratio:.2%})")
        
        # Check for memory optimization
        if df[col].dtype == 'int64':
            if df[col].min() >= -32768 and df[col].max() <= 32767:
                col_suggestions.append("Convert to int16 for memory optimization")
            elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                col_suggestions.append("Convert to int32 for memory optimization")
        
        if df[col].dtype == 'float64':
            col_suggestions.append("Convert to float32 for memory optimization")
        
        if col_suggestions:
            suggestions[col] = col_suggestions
    
    return suggestions

def get_memory_usage(df):
    """Get memory usage information for a DataFrame."""
    try:
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        return {
            'total_memory_mb': total_memory / 1024 / 1024,
            'total_memory_kb': total_memory / 1024,
            'total_memory_bytes': total_memory,
            'per_column': {col: size / 1024 / 1024 for col, size in memory_usage.items()}
        }
    except Exception as e:
        logging.error(f"Error calculating memory usage: {e}")
        return None

def create_summary_statistics(df):
    """Create comprehensive summary statistics for a DataFrame."""
    try:
        summary = {
            'shape': df.shape,
            'memory_usage': get_memory_usage(df),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_summary': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
            'categorical_summary': {}
        }
        
        # Add categorical column summaries
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_count': df[col].nunique(),
                'most_common': df[col].value_counts().head(5).to_dict(),
                'least_common': df[col].value_counts().tail(5).to_dict()
            }
        
        return summary
        
    except Exception as e:
        logging.error(f"Error creating summary statistics: {e}")
        return None

def format_number(value, decimal_places=2):
    """Format numbers for display."""
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        
        if isinstance(value, (int, float)):
            if value == 0:
                return "0"
            elif abs(value) < 0.01:
                return f"{value:.2e}"
            elif abs(value) >= 1000000:
                return f"{value/1000000:.{decimal_places}f}M"
            elif abs(value) >= 1000:
                return f"{value/1000:.{decimal_places}f}K"
            else:
                return f"{value:.{decimal_places}f}"
        else:
            return str(value)
    except:
        return str(value)

def safe_divide(numerator, denominator, default_value=0):
    """Safely divide two numbers, handling division by zero."""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default_value
        return numerator / denominator
    except:
        return default_value

def create_data_quality_report(df):
    """Create a comprehensive data quality report."""
    try:
        report = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': get_memory_usage(df)['total_memory_mb'] if get_memory_usage(df) else 0
            },
            'data_types': df.dtypes.value_counts().to_dict(),
            'missing_data': {
                'total_missing': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'columns_with_missing': df.columns[df.isnull().any()].tolist()
            },
            'duplicates': {
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
            },
            'outliers': {}
        }
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            report['outliers'][col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100
            }
        
        return report
        
    except Exception as e:
        logging.error(f"Error creating data quality report: {e}")
        return None

def export_report_to_json(report, file_path):
    """Export a report to JSON format."""
    try:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert the report
        converted_report = convert_numpy_types(report)
        
        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(converted_report, f, indent=2, default=str)
        
        logging.info(f"Report exported successfully to {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error exporting report: {e}")
        return False

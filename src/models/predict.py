"""
Prediction functions for waste management models.
This module handles making predictions using trained models.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import catboost as cb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

class WasteManagementPredictor:
    """A class to handle predictions using trained waste management models."""
    
    def __init__(self, model_path=None):
        self.model = None
        self.feature_columns = None
        self.target_column = None
        self.model_type = None
        self.task = None
        self.scaler = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model for prediction."""
        try:
            model_path = Path(model_path)
            
            # Try to determine model type from file extension
            if model_path.suffix == '.cbm':
                self.model_type = 'catboost'
                # We'll determine task when loading info
            elif model_path.suffix == '.pkl':
                self.model_type = 'sklearn'
                # We'll determine task when loading info
            else:
                logger.error(f"Unsupported model format: {model_path.suffix}")
                return False
            
            # Load model info first
            info_path = model_path.with_suffix('.info.pkl')
            if info_path.exists():
                model_info = joblib.load(info_path)
                self.feature_columns = model_info.get('feature_columns')
                self.target_column = model_info.get('target_column')
                self.model_type = model_info.get('model_type', self.model_type)
                self.task = model_info.get('task', 'regression')
                self.scaler = model_info.get('scaler')
            else:
                logger.warning("Model info file not found, using default settings")
                self.task = 'regression'
            
            # Load the actual model
            if self.model_type == 'catboost':
                if self.task == 'regression':
                    self.model = cb.CatBoostRegressor()
                else:
                    self.model = cb.CatBoostClassifier()
                self.model.load_model(str(model_path))
            else:
                self.model = joblib.load(model_path)
            
            logger.info("Model loaded successfully for prediction")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for prediction: {e}")
            return False
    
    def prepare_features_for_prediction(self, df):
        """Prepare input data for prediction by selecting and formatting features."""
        if not self.feature_columns:
            logger.error("No feature columns defined. Please load a model first.")
            return None
        
        try:
            # Select only the required features
            if isinstance(df, pd.DataFrame):
                # If DataFrame, select columns
                if all(col in df.columns for col in self.feature_columns):
                    X = df[self.feature_columns].copy()
                else:
                    missing_cols = [col for col in self.feature_columns if col not in df.columns]
                    logger.error(f"Missing required columns: {missing_cols}")
                    return None
            else:
                # If single row data, create DataFrame
                if len(df) != len(self.feature_columns):
                    logger.error(f"Expected {len(self.feature_columns)} features, got {len(df)}")
                    return None
                X = pd.DataFrame([df], columns=self.feature_columns)
            
            # Handle missing values
            X = X.fillna(0)
            
            # Ensure numeric types
            for col in X.columns:
                if X[col].dtype not in ['int64', 'float64']:
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        X[col] = X[col].fillna(0)
                    except:
                        logger.error(f"Could not convert column {col} to numeric")
                        return None
            
            # Scale features if scaler is available
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            return X
            
        except Exception as e:
            logger.error(f"Error preparing features for prediction: {e}")
            return None
    
    def predict(self, input_data):
        """Make predictions using the loaded model."""
        if not self.model:
            logger.error("No model loaded. Please load a model first.")
            return None
        
        try:
            # Prepare features
            X = self.prepare_features_for_prediction(input_data)
            if X is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(X)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def predict_batch(self, df):
        """Make batch predictions on a DataFrame."""
        if not self.model:
            logger.error("No model loaded. Please load a model first.")
            return None
        
        try:
            # Prepare features
            X = self.prepare_features_for_prediction(df)
            if X is None:
                return None
            
            # Make predictions
            predictions = self.predict(X)
            
            if predictions is not None:
                # Add predictions to DataFrame
                result_df = df.copy()
                result_df[f'{self.target_column}_predicted'] = predictions
                
                return result_df
            
            return None
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            return None
    
    def predict_proba(self, input_data):
        """Get prediction probabilities for classification tasks."""
        if not self.model or self.task != 'classification':
            logger.error("Probability predictions only available for classification models")
            return None
        
        try:
            # Prepare features
            X = self.prepare_features_for_prediction(input_data)
            if X is None:
                return None
            
            # Get probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
                return probabilities
            else:
                logger.warning("Model does not support probability predictions")
                return None
                
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {e}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from the model."""
        if not self.model:
            logger.error("No model loaded. Please load a model first.")
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_names = self.feature_columns
            elif hasattr(self.model, 'get_feature_importance'):
                # CatBoost models
                importance = self.model.get_feature_importance()
                feature_names = self.feature_columns
            else:
                logger.warning("Feature importance not available for this model type")
                return None
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None
    
    def explain_prediction(self, input_data, feature_names=None):
        """Explain a single prediction."""
        if not self.model:
            logger.error("No model loaded. Please load a model first.")
            return None
        
        try:
            # For now, return basic prediction info
            # In a full implementation, you could integrate SHAP or LIME
            X = self.prepare_features_for_prediction(input_data)
            if X is None:
                return None
            
            prediction = self.predict(X)
            if prediction is None:
                return None
            
            # Create explanation DataFrame
            if isinstance(input_data, pd.DataFrame):
                input_row = input_data.iloc[0] if len(input_data) == 1 else input_data
            else:
                input_row = pd.Series(input_data, index=self.feature_columns)
            
            explanation_df = pd.DataFrame({
                'feature': self.feature_columns,
                'value': [input_row[col] for col in self.feature_columns]
            })
            
            return {
                'prediction': prediction[0] if len(prediction) == 1 else prediction,
                'features': explanation_df
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return None
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if not self.model:
            return None
        
        info = {
            'model_type': self.model_type,
            'task': self.task,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'target_column': self.target_column,
            'has_scaler': self.scaler is not None
        }
        
        return info

"""
Model training logic for waste management prediction.
This module handles model training, evaluation, and saving.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import catboost as cb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

class WasteManagementModel:
    """A class to handle waste management model training and evaluation."""
    
    def __init__(self, model_type='random_forest', task='regression'):
        self.model_type = model_type
        self.task = task
        self.model = None
        self.feature_columns = None
        self.target_column = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, df, target_column, exclude_columns=None):
        """Prepare features for training by selecting relevant columns."""
        if exclude_columns is None:
            exclude_columns = []
        
        # Get all columns except target and excluded
        feature_columns = [col for col in df.columns 
                          if col != target_column and col not in exclude_columns]
        
        # Ensure all features are numeric
        numeric_features = []
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            else:
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_features.append(col)
                except:
                    logger.warning(f"Could not convert column {col} to numeric, excluding from features")
        
        self.feature_columns = numeric_features
        self.target_column = target_column
        
        logger.info(f"Prepared {len(numeric_features)} features for training")
        return numeric_features
    
    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train a Random Forest model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with any provided parameters
        default_params.update(kwargs)
        
        if self.task == 'regression':
            model = RandomForestRegressor(**default_params)
        else:
            model = RandomForestClassifier(**default_params)
        
        # Train model
        model.fit(X_train, y_train)
        
        return model
    
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train a CatBoost model."""
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE' if self.task == 'regression' else 'MultiClass',
            'verbose': 100,
            'random_state': 42
        }
        
        # Update with any provided parameters
        default_params.update(kwargs)
        
        # Initialize model
        if self.task == 'regression':
            model = cb.CatBoostRegressor(**default_params)
        else:
            model = cb.CatBoostClassifier(**default_params)
        
        # Train model
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def train_linear_model(self, X_train, y_train, **kwargs):
        """Train a linear model."""
        if self.task == 'regression':
            model = LinearRegression(**kwargs)
        else:
            model = LogisticRegression(random_state=42, **kwargs)
        
        model.fit(X_train, y_train)
        return model
    
    def train_model(self, df, target_column, test_size=0.2, random_state=42, **model_params):
        """Train the model with the given data."""
        try:
            # Prepare features
            feature_columns = self.prepare_features(df, target_column)
            
            if not feature_columns:
                logger.error("No valid features found for training")
                return False
            
            # Prepare data
            X = df[feature_columns].fillna(0)
            y = df[target_column].fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model based on type
            if self.model_type == 'random_forest':
                self.model = self.train_random_forest(X_train_scaled, y_train, **model_params)
            elif self.model_type == 'catboost':
                self.model = self.train_catboost(X_train_scaled, y_train, X_test_scaled, y_test, **model_params)
            elif self.model_type == 'linear':
                self.model = self.train_linear_model(X_train_scaled, y_train, **model_params)
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return False
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            if self.task == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logger.info("Model training completed!")
                logger.info(f"Model Performance:")
                logger.info(f"- RMSE: {rmse:.4f}")
                logger.info(f"- MAE: {mae:.4f}")
                logger.info(f"- R² Score: {r2:.4f}")
            else:
                # Classification metrics
                report = classification_report(y_test, y_pred)
                logger.info("Model training completed!")
                logger.info(f"Classification Report:\n{report}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
    
    def cross_validate(self, df, target_column, cv=5):
        """Perform cross-validation on the model."""
        try:
            if not self.is_trained:
                logger.error("Model must be trained before cross-validation")
                return None
            
            feature_columns = self.prepare_features(df, target_column)
            X = df[feature_columns].fillna(0)
            y = df[target_column].fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Perform cross-validation
            if self.task == 'regression':
                scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='r2')
            else:
                scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
            
            logger.info(f"Cross-validation scores: {scores}")
            logger.info(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            return scores
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {e}")
            return None
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        try:
            if self.model is not None and self.is_trained:
                # Save model
                if self.model_type == 'catboost':
                    self.model.save_model(str(filepath))
                else:
                    joblib.dump(self.model, filepath)
                
                # Save feature information and scaler
                model_info = {
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'model_type': self.model_type,
                    'task': self.task,
                    'scaler': self.scaler
                }
                info_path = Path(filepath).with_suffix('.info.pkl')
                joblib.dump(model_info, info_path)
                
                logger.info(f"Model saved successfully to {filepath}")
                return True
            else:
                logger.error("No trained model to save")
                return False
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        try:
            if self.model_type == 'catboost':
                if self.task == 'regression':
                    self.model = cb.CatBoostRegressor()
                else:
                    self.model = cb.CatBoostClassifier()
                self.model.load_model(str(filepath))
            else:
                self.model = joblib.load(filepath)
            
            # Load model info
            info_path = Path(filepath).with_suffix('.info.pkl')
            if info_path.exists():
                model_info = joblib.load(info_path)
                self.feature_columns = model_info.get('feature_columns')
                self.target_column = model_info.get('target_column')
                self.scaler = model_info.get('scaler')
                self.is_trained = True
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

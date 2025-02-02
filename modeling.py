"""
Statistical modeling module for market analysis and prediction.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from trading_types import (
    Symbol, Config, TradingError, ValidationError
)

class BaseModel:
    """Base class for statistical models"""
    
    def __init__(self, config: Config):
        """Initialize base model
        
        Args:
            config: System configuration
        """
        self.config = config
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> None:
        """Fit model to data
        
        Args:
            X: Input data matrix
            
        Raises:
            ValidationError: If input data is invalid
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions
        
        Args:
            X: Input data matrix
            
        Returns:
            Model predictions
            
        Raises:
            ValidationError: If model is not fitted or input is invalid
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def _validate_input(self, X: np.ndarray, min_samples: int = 2) -> None:
        """Validate input data
        
        Args:
            X: Input data matrix
            min_samples: Minimum required samples
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(X, np.ndarray):
            raise ValidationError("Input must be numpy array")
            
        if len(X.shape) != 2:
            raise ValidationError("Input must be 2-dimensional")
            
        if X.shape[0] < min_samples:
            raise ValidationError(f"At least {min_samples} samples required")

class PCAModel(BaseModel):
    """Principal Component Analysis model"""
    
    def __init__(
        self,
        config: Config,
        n_components: int = 3,
        standardize: bool = True
    ):
        """Initialize PCA model
        
        Args:
            config: System configuration
            n_components: Number of components to keep
            standardize: Whether to standardize input data
        """
        super().__init__(config)
        self.n_components = n_components
        self.standardize = standardize
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler() if standardize else None
        
    def fit(self, X: np.ndarray) -> None:
        """Fit PCA model
        
        Args:
            X: Input data matrix [samples, features]
            
        Raises:
            ValidationError: If input data is invalid
        """
        try:
            self._validate_input(X)
            
            # Standardize if requested
            if self.standardize:
                X = self.scaler.fit_transform(X)
            
            # Fit PCA
            self.pca.fit(X)
            self.is_fitted = True
            
        except Exception as e:
            raise ValidationError(f"PCA fitting failed: {str(e)}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to principal component space
        
        Args:
            X: Input data matrix
            
        Returns:
            Transformed data
            
        Raises:
            ValidationError: If model is not fitted or input is invalid
        """
        if not self.is_fitted:
            raise ValidationError("Model must be fitted before transform")
            
        try:
            self._validate_input(X)
            
            # Standardize if requested
            if self.standardize:
                X = self.scaler.transform(X)
            
            # Transform data
            return self.pca.transform(X)
            
        except Exception as e:
            raise ValidationError(f"PCA transform failed: {str(e)}")
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data back to original space
        
        Args:
            X: Data in principal component space
            
        Returns:
            Data in original space
            
        Raises:
            ValidationError: If model is not fitted or input is invalid
        """
        if not self.is_fitted:
            raise ValidationError("Model must be fitted before inverse transform")
            
        try:
            # Inverse transform
            X_orig = self.pca.inverse_transform(X)
            
            # Inverse standardize if needed
            if self.standardize:
                X_orig = self.scaler.inverse_transform(X_orig)
                
            return X_orig
            
        except Exception as e:
            raise ValidationError(f"PCA inverse transform failed: {str(e)}")
    
    @property
    def components(self) -> np.ndarray:
        """Get principal components
        
        Returns:
            Principal component vectors
            
        Raises:
            ValidationError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValidationError("Model must be fitted first")
        return self.pca.components_
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratios
        
        Returns:
            Explained variance ratios
            
        Raises:
            ValidationError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValidationError("Model must be fitted first")
        return self.pca.explained_variance_ratio_

class RollingPCA(PCAModel):
    """Rolling window PCA model"""
    
    def __init__(
        self,
        config: Config,
        window: int = 252,
        n_components: int = 3,
        standardize: bool = True
    ):
        """Initialize Rolling PCA model
        
        Args:
            config: System configuration
            window: Rolling window size
            n_components: Number of components to keep
            standardize: Whether to standardize input data
        """
        super().__init__(config, n_components, standardize)
        self.window = window
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using rolling PCA
        
        Args:
            X: Input data matrix
            
        Returns:
            Transformed data
            
        Raises:
            ValidationError: If input data is invalid
        """
        try:
            self._validate_input(X)
            
            # Initialize output array
            n_samples = X.shape[0]
            transformed = np.zeros((n_samples, self.n_components))
            
            # Apply rolling PCA
            for i in range(self.window, n_samples + 1):
                window_data = X[i-self.window:i]
                
                # Fit and transform window
                if self.standardize:
                    window_data = self.scaler.fit_transform(window_data)
                self.pca.fit(window_data)
                
                if i < n_samples:
                    next_sample = X[i:i+1]
                    if self.standardize:
                        next_sample = self.scaler.transform(next_sample)
                    transformed[i] = self.pca.transform(next_sample)
            
            return transformed
            
        except Exception as e:
            raise ValidationError(f"Rolling PCA transform failed: {str(e)}")

class DynamicPCA(PCAModel):
    """Dynamic PCA with adaptive number of components"""
    
    def __init__(
        self,
        config: Config,
        var_threshold: float = 0.95,
        max_components: int = 10,
        standardize: bool = True
    ):
        """Initialize Dynamic PCA model
        
        Args:
            config: System configuration
            var_threshold: Minimum explained variance threshold
            max_components: Maximum number of components
            standardize: Whether to standardize input data
        """
        super().__init__(config, n_components=max_components, standardize=standardize)
        self.var_threshold = var_threshold
        self.max_components = max_components
        
    def fit(self, X: np.ndarray) -> None:
        """Fit Dynamic PCA model
        
        Args:
            X: Input data matrix
            
        Raises:
            ValidationError: If input data is invalid
        """
        try:
            self._validate_input(X)
            
            # Standardize if requested
            if self.standardize:
                X = self.scaler.fit_transform(X)
            
            # Fit PCA with max components
            self.pca.fit(X)
            
            # Find optimal number of components
            cumsum = np.cumsum(self.pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum >= self.var_threshold) + 1
            
            # Update PCA model
            self.pca = PCA(n_components=self.n_components)
            if self.standardize:
                X = self.scaler.transform(X)
            self.pca.fit(X)
            
            self.is_fitted = True
            
        except Exception as e:
            raise ValidationError(f"Dynamic PCA fitting failed: {str(e)}")
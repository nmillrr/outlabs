"""
LDL-C prediction API.

This module provides a simple interface for making LDL-C predictions
using various methods including mechanistic equations and the hybrid ML model.
"""

import os
import warnings
from typing import Optional, Dict, Any, Union
import numpy as np

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from ldlC.models import (
    calc_ldl_friedewald,
    calc_ldl_martin_hopkins,
    calc_ldl_martin_hopkins_extended,
    calc_ldl_sampson
)


# Default model path (relative to package directory)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'models',
    'best_model.joblib'
)

# Cache for loaded model
_model_cache: Dict[str, Any] = {}


def _load_hybrid_model(model_path: Optional[str] = None) -> Any:
    """
    Load the hybrid ML model from disk.
    
    Args:
        model_path: Path to the model file. If None, uses default path.
        
    Returns:
        Loaded model object.
        
    Raises:
        FileNotFoundError: If model file doesn't exist.
        ImportError: If joblib is not installed.
    """
    if not HAS_JOBLIB:
        raise ImportError("joblib is required to load the hybrid model. Install with: pip install joblib")
    
    path = model_path or _DEFAULT_MODEL_PATH
    
    # Check cache first
    if path in _model_cache:
        return _model_cache[path]
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at: {path}. "
            "Please train a model first using the training notebook."
        )
    
    model = joblib.load(path)
    _model_cache[path] = model
    return model


def _create_features_for_prediction(tc: float, hdl: float, tg: float) -> np.ndarray:
    """
    Create feature array for hybrid model prediction.
    
    Args:
        tc: Total cholesterol in mg/dL
        hdl: HDL cholesterol in mg/dL
        tg: Triglycerides in mg/dL
        
    Returns:
        Feature array matching training feature order.
    """
    non_hdl = tc - hdl
    tg_hdl_ratio = tg / hdl if hdl > 0 else np.nan
    tc_hdl_ratio = tc / hdl if hdl > 0 else np.nan
    
    # Calculate equation predictions
    try:
        ldl_friedewald = calc_ldl_friedewald(tc, hdl, tg)
    except (ValueError, Warning):
        ldl_friedewald = np.nan
    
    try:
        ldl_martin_hopkins = calc_ldl_martin_hopkins(tc, hdl, tg)
    except ValueError:
        ldl_martin_hopkins = np.nan
    
    try:
        ldl_martin_hopkins_ext = calc_ldl_martin_hopkins_extended(tc, hdl, tg)
    except ValueError:
        ldl_martin_hopkins_ext = np.nan
    
    try:
        ldl_sampson = calc_ldl_sampson(tc, hdl, tg)
    except ValueError:
        ldl_sampson = np.nan
    
    # Feature order must match training (from train.create_features)
    features = np.array([
        tc,                      # tc_mgdl
        hdl,                     # hdl_mgdl
        tg,                      # tg_mgdl
        non_hdl,                 # non_hdl_mgdl
        tg_hdl_ratio,            # tg_hdl_ratio
        tc_hdl_ratio,            # tc_hdl_ratio
        ldl_friedewald,          # ldl_friedewald
        ldl_martin_hopkins,      # ldl_martin_hopkins
        ldl_martin_hopkins_ext,  # ldl_martin_hopkins_extended
        ldl_sampson              # ldl_sampson
    ]).reshape(1, -1)
    
    return features


def predict_ldl(
    tc: Union[float, int],
    hdl: Union[float, int],
    tg: Union[float, int],
    method: str = 'hybrid',
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict LDL cholesterol from lipid panel values.
    
    This is the main API for making LDL-C predictions. Supports multiple
    methods including the traditional equations and the hybrid ML model.
    
    Args:
        tc: Total cholesterol in mg/dL (must be positive)
        hdl: HDL cholesterol in mg/dL (must be positive)
        tg: Triglycerides in mg/dL (must be positive)
        method: Prediction method. One of:
            - 'friedewald': Traditional Friedewald equation (TG < 400)
            - 'martin_hopkins': Martin-Hopkins equation with adjustable factor
            - 'martin_hopkins_extended': Extended M-H for high TG
            - 'sampson': Sampson/NIH Equation 2
            - 'hybrid': ML model combining all equations (default)
        model_path: Path to hybrid model file. Only used when method='hybrid'.
            If None, uses default model path.
            
    Returns:
        Dictionary with:
            - ldl_pred: Predicted LDL-C in mg/dL
            - ci_lower: Lower 95% confidence bound (approximate)
            - ci_upper: Upper 95% confidence bound (approximate)
            - method: Method used for prediction
            - warning: Warning message if any (e.g., TG > 400)
            
    Raises:
        ValueError: If inputs are invalid or method is unknown.
        
    Example:
        >>> result = predict_ldl(200, 50, 150)
        >>> print(f"LDL: {result['ldl_pred']:.1f} mg/dL")
        LDL: 120.0 mg/dL
    """
    # Validate inputs
    try:
        tc = float(tc)
        hdl = float(hdl)
        tg = float(tg)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid input values: {e}")
    
    if np.isnan(tc) or np.isnan(hdl) or np.isnan(tg):
        raise ValueError("Input values cannot be NaN")
    
    if tc <= 0 or hdl <= 0 or tg < 0:
        raise ValueError("TC and HDL must be positive; TG must be non-negative")
    
    # Validate method
    valid_methods = ['friedewald', 'martin_hopkins', 'martin_hopkins_extended', 'sampson', 'hybrid']
    method_lower = method.lower()
    if method_lower not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Must be one of: {valid_methods}")
    
    # Initialize result
    result: Dict[str, Any] = {
        'ldl_pred': None,
        'ci_lower': None,
        'ci_upper': None,
        'method': method_lower,
        'warning': None
    }
    
    # Check for high TG warning
    if tg > 800:
        result['warning'] = "TG > 800 mg/dL: Direct LDL measurement recommended"
    elif tg > 400:
        result['warning'] = "TG > 400 mg/dL: Friedewald not valid; other methods may have reduced accuracy"
    
    # Calculate prediction based on method
    try:
        if method_lower == 'friedewald':
            ldl_pred = calc_ldl_friedewald(tc, hdl, tg)
            # Friedewald has ~10-12 mg/dL standard error at normal TG
            se = 11.0 if tg <= 150 else 15.0
            
        elif method_lower == 'martin_hopkins':
            ldl_pred = calc_ldl_martin_hopkins(tc, hdl, tg)
            se = 10.0 if tg <= 400 else 12.0
            
        elif method_lower == 'martin_hopkins_extended':
            ldl_pred = calc_ldl_martin_hopkins_extended(tc, hdl, tg)
            se = 10.0 if tg <= 400 else 12.0
            
        elif method_lower == 'sampson':
            ldl_pred = calc_ldl_sampson(tc, hdl, tg)
            se = 9.0 if tg <= 400 else 11.0
            
        else:  # hybrid
            model = _load_hybrid_model(model_path)
            features = _create_features_for_prediction(tc, hdl, tg)
            
            # Check for NaN features
            if np.any(np.isnan(features)):
                # Fall back to Sampson if features have NaN
                ldl_pred = calc_ldl_sampson(tc, hdl, tg)
                result['warning'] = (result.get('warning') or '') + (
                    "; Hybrid model unavailable, falling back to Sampson"
                )
                se = 10.0
            else:
                ldl_pred = float(model.predict(features)[0])
                # Hybrid model typically has lower error
                se = 8.0 if tg <= 400 else 10.0
        
        # Set prediction and approximate CI
        result['ldl_pred'] = round(ldl_pred, 1) if ldl_pred is not None and not np.isnan(ldl_pred) else None
        
        if result['ldl_pred'] is not None:
            result['ci_lower'] = round(ldl_pred - 1.96 * se, 1)
            result['ci_upper'] = round(ldl_pred + 1.96 * se, 1)
            
    except ValueError as e:
        result['ldl_pred'] = None
        result['warning'] = str(e)
    except FileNotFoundError as e:
        # Hybrid model not found, fall back to Sampson
        try:
            ldl_pred = calc_ldl_sampson(tc, hdl, tg)
            result['ldl_pred'] = round(ldl_pred, 1)
            se = 9.0
            result['ci_lower'] = round(ldl_pred - 1.96 * se, 1)
            result['ci_upper'] = round(ldl_pred + 1.96 * se, 1)
            result['method'] = 'sampson'
            result['warning'] = "Hybrid model not found; used Sampson equation"
        except ValueError as e2:
            result['ldl_pred'] = None
            result['warning'] = str(e2)
    
    return result

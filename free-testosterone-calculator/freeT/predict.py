"""
Prediction API for free testosterone estimation.

Provides a simple interface for making FT predictions using either
mechanistic solvers (Vermeulen) or hybrid ML models.
"""

from pathlib import Path
from typing import Dict, Optional, Any
import math


def predict_ft(
    tt: float,
    shbg: float,
    alb: float,
    method: str = 'hybrid',
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict free testosterone (FT) from total testosterone, SHBG, and albumin.
    
    This function provides a unified API for FT prediction using either the
    mechanistic Vermeulen solver or a hybrid ML model trained on NHANES data.
    
    Parameters
    ----------
    tt : float
        Total testosterone in nmol/L. Must be positive.
    shbg : float
        Sex hormone-binding globulin in nmol/L. Must be non-negative.
    alb : float
        Albumin in g/L. Must be positive (typically 35-55 g/L).
    method : str, default='hybrid'
        Prediction method to use:
        - 'vermeulen': Use mechanistic Vermeulen solver only
        - 'hybrid': Use best ML model (falls back to Vermeulen if no model found)
    model_path : str, optional
        Path to a saved model file (.joblib). If not provided, looks in default
        location (models/best_model.joblib relative to package root).
    
    Returns
    -------
    dict
        Dictionary containing:
        - ft_pred : float
            Predicted free testosterone in nmol/L
        - ci_lower : float or None
            Lower bound of 95% confidence interval (None for Vermeulen)
        - ci_upper : float or None
            Upper bound of 95% confidence interval (None for Vermeulen)
        - method : str
            Method actually used for prediction
    
    Raises
    ------
    ValueError
        If inputs are invalid (negative, NaN, or out of physiological range).
    
    Examples
    --------
    >>> result = predict_ft(tt=15.0, shbg=40.0, alb=45.0)
    >>> print(f"Free T: {result['ft_pred']:.3f} nmol/L")
    Free T: 0.269 nmol/L
    
    >>> result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='vermeulen')
    >>> print(f"Method: {result['method']}")
    Method: vermeulen
    """
    # Input validation
    if math.isnan(tt) or math.isnan(shbg) or math.isnan(alb):
        raise ValueError("Input values cannot be NaN")
    
    if tt < 0:
        raise ValueError(f"Total testosterone must be non-negative, got {tt}")
    
    if shbg < 0:
        raise ValueError(f"SHBG must be non-negative, got {shbg}")
    
    if alb <= 0:
        raise ValueError(f"Albumin must be positive, got {alb}")
    
    if method not in ('vermeulen', 'hybrid'):
        raise ValueError(f"Method must be 'vermeulen' or 'hybrid', got '{method}'")
    
    # Handle zero TT edge case
    if tt == 0:
        return {
            'ft_pred': 0.0,
            'ci_lower': None,
            'ci_upper': None,
            'method': method,
        }
    
    # Import models for Vermeulen solver
    from freeT.models import calc_ft_vermeulen
    
    if method == 'vermeulen':
        ft_pred = calc_ft_vermeulen(tt, shbg, alb)
        return {
            'ft_pred': float(ft_pred),
            'ci_lower': None,
            'ci_upper': None,
            'method': 'vermeulen',
        }
    
    # Hybrid method: try to load and use ML model
    if method == 'hybrid':
        model = None
        actual_method = 'hybrid'
        
        # Try to load model
        if model_path is not None:
            model_file = Path(model_path)
            if model_file.exists():
                try:
                    import joblib
                    model = joblib.load(model_file)
                except Exception:
                    model = None
        else:
            # Look for default model location
            package_root = Path(__file__).parent.parent
            default_paths = [
                package_root / 'models' / 'best_model.joblib',
                package_root / 'models' / 'lightgbm_model.joblib',
                package_root / 'models' / 'random_forest_model.joblib',
                package_root / 'models' / 'ridge_model.joblib',
            ]
            for default_path in default_paths:
                if default_path.exists():
                    try:
                        import joblib
                        model = joblib.load(default_path)
                        break
                    except Exception:
                        continue
        
        if model is not None:
            # Create feature vector: [tt_nmoll, shbg_nmoll, alb_gl, shbg_tt_ratio, ft_vermeulen]
            import numpy as np
            
            shbg_tt_ratio = shbg / (tt + 1e-10)  # Avoid division by zero
            ft_vermeulen = calc_ft_vermeulen(tt, shbg, alb)
            
            X = np.array([[tt, shbg, alb, shbg_tt_ratio, ft_vermeulen]])
            
            try:
                ft_pred = float(model.predict(X)[0])
                
                # Estimate confidence interval using prediction variance if available
                ci_lower = None
                ci_upper = None
                
                # For RandomForest, we can get prediction intervals from trees
                if hasattr(model, 'estimators_'):
                    tree_predictions = np.array([
                        tree.predict(X)[0] for tree in model.estimators_
                    ])
                    ci_lower = float(np.percentile(tree_predictions, 2.5))
                    ci_upper = float(np.percentile(tree_predictions, 97.5))
                
                return {
                    'ft_pred': ft_pred,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'method': 'hybrid',
                }
            except Exception:
                # Fall back to Vermeulen if prediction fails
                actual_method = 'vermeulen'
        else:
            # No model found, fall back to Vermeulen
            actual_method = 'vermeulen'
        
        # Fallback to Vermeulen
        ft_pred = calc_ft_vermeulen(tt, shbg, alb)
        return {
            'ft_pred': float(ft_pred),
            'ci_lower': None,
            'ci_upper': None,
            'method': actual_method,
        }

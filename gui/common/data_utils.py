"""
Data utility functions for GUI operations.

This module provides helper functions for data validation, safe statistics,
and JSON serialization.
"""

import numpy as np


def validate_array(arr):
    """
    Check if array is valid and non-empty.
    
    Parameters:
    -----------
    arr : array-like or None
        Array to validate
    
    Returns:
    --------
    bool : True if array is valid and has elements, False otherwise
    """
    return arr is not None and len(arr) > 0


def safe_mean(arr):
    """
    Safely compute mean of array, returning NaN if invalid.
    
    Parameters:
    -----------
    arr : array-like or None
        Array to compute mean of
    
    Returns:
    --------
    float : Mean value or np.nan if invalid
    """
    return float(np.mean(arr)) if validate_array(arr) else np.nan


def safe_std(arr):
    """
    Safely compute standard deviation of array, returning NaN if invalid.
    
    Parameters:
    -----------
    arr : array-like or None
        Array to compute std of
    
    Returns:
    --------
    float : Standard deviation or np.nan if invalid
    """
    return float(np.std(arr)) if validate_array(arr) else np.nan


def to_json_serializable(data):
    """
    Convert data (including numpy arrays) to JSON-serializable format.
    
    Parameters:
    -----------
    data : any
        Data to convert (can be numpy array, dict, list, etc.)
    
    Returns:
    --------
    Converted data in JSON-serializable format
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_json_serializable(item) for item in data]
    else:
        return data


def downsample_for_plot(data, max_points=2000):
    """
    Downsample large datasets for faster plotting without losing visual information.
    
    Uses uniform sampling to preserve distribution shape while reducing points.
    
    Parameters:
    -----------
    data : array-like
        Data to downsample
    max_points : int, optional
        Maximum number of points to keep (default: 2000)
    
    Returns:
    --------
    array-like : Downsampled data (or original if already small enough)
    """
    if len(data) <= max_points:
        return data
    # Use uniform sampling to preserve distribution
    indices = np.linspace(0, len(data)-1, max_points, dtype=int)
    return np.array(data)[indices]


"""
Core Utils - Shared Utility Functions
"""
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def convert_coordinates(ra_str: str, dec_str: str) -> tuple[float, float]:
    """
    Convert RA/Dec from string format to decimal degrees.
    
    Args:
        ra_str: Right Ascension (e.g., "14h 22m 36.5s")
        dec_str: Declination (e.g., "-32° 05' 12.3\"")
    
    Returns:
        Tuple of (ra_deg, dec_deg)
    """
    # TODO: Implement coordinate conversion
    raise NotImplementedError("Coordinate conversion not yet implemented")


def calculate_airmass(altitude: float) -> float:
    """
    Calculate airmass from altitude angle.
    
    Args:
        altitude: Altitude angle in degrees
    
    Returns:
        Airmass value
    """
    import math
    if altitude <= 0:
        return float('inf')
    
    # Simple plane-parallel approximation
    zenith_angle = 90.0 - altitude
    zenith_radians = math.radians(zenith_angle)
    airmass = 1.0 / math.cos(zenith_radians)
    
    return airmass


def safe_get_nested(data: Dict[str, Any], *keys, default=None) -> Any:
    """
    Safely get nested dictionary value.
    
    Args:
        data: Dictionary to search
        *keys: Nested keys to traverse
        default: Default value if key not found
    
    Returns:
        Value at nested key or default
    
    Example:
        >>> data = {'a': {'b': {'c': 42}}}
        >>> safe_get_nested(data, 'a', 'b', 'c')
        42
    """
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
            if current is None:
                return default
        else:
            return default
    return current

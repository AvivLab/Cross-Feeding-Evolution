"""
Color schemes and palettes for all GUIs.

This module provides color-blind friendly color schemes using
the Okabe-Ito qualitative palette and safe sequential colormaps.
"""

# Okabe-Ito qualitative palette (color-blind friendly)
# Reference: https://jfly.uni-koeln.de/color/
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "blue": "#0072B2",
    "green": "#009E73",
    "yellow": "#F0E442",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#999999",
    "light_gray": "#D3D3D3",
}

# Semantic series colors used across plots
SERIES_COLORS = {
    "m1_env": OKABE_ITO["blue"],
    "m2_env": OKABE_ITO["orange"],
    "m1_internal": OKABE_ITO["green"],
    "m2_internal": OKABE_ITO["purple"],
    "energy": OKABE_ITO["vermillion"],
    "population": OKABE_ITO["sky"],
    "import": OKABE_ITO["blue"],
    "export": OKABE_ITO["vermillion"],
    "production": OKABE_ITO["green"],
    "consumption": OKABE_ITO["purple"],
    "transport": OKABE_ITO["sky"],
    "net": OKABE_ITO["gray"],
    "death": OKABE_ITO["vermillion"],
    "duplication": OKABE_ITO["blue"],
    "mutation": OKABE_ITO["purple"],
    "volume": OKABE_ITO["orange"],
}

# Enzyme-specific colors (for consistency across plots)
ENZYME_COLORS = {
    "A": OKABE_ITO["blue"],      # Enzyme A (production)
    "B": OKABE_ITO["orange"],    # Enzyme B (consumption)
    "T": OKABE_ITO["green"],     # Enzyme T (transport)
}

# Task/Performance colors
TASK_COLORS = {
    "task1": OKABE_ITO["blue"],      # Task A performance
    "task2": OKABE_ITO["orange"],    # Task B performance
    "transport": OKABE_ITO["green"], # Transport
}

# Status colors
STATUS_COLORS = {
    "energy": OKABE_ITO["vermillion"],
    "metabolite": OKABE_ITO["purple"],
    "population": OKABE_ITO["sky"],
}

# Colormaps for heatmaps (color-blind friendly)
HEATMAP_COLORMAPS = {
    "default": "cividis",
    "viridis": "viridis",
    "cividis": "cividis",
}

# Default colormap for all heatmaps
DEFAULT_HEATMAP_CMAP = HEATMAP_COLORMAPS["default"]


def get_enzyme_color(enzyme_name):
    """
    Get color for a specific enzyme.
    
    Parameters:
    -----------
    enzyme_name : str
        Enzyme name ('A', 'B', or 'T')
    
    Returns:
    --------
    str : Hex color code
    """
    return ENZYME_COLORS.get(enzyme_name, OKABE_ITO["blue"])


def get_task_color(task_name):
    """
    Get color for a specific task.
    
    Parameters:
    -----------
    task_name : str
        Task identifier ('task1', 'task2', 'transport')
    
    Returns:
    --------
    str : Hex color code
    """
    return TASK_COLORS.get(task_name, OKABE_ITO["blue"])


def get_series_color(series_name, fallback="blue"):
    """
    Get color for a semantic series name.
    """
    if series_name in SERIES_COLORS:
        return SERIES_COLORS[series_name]
    return OKABE_ITO.get(fallback, OKABE_ITO["blue"])


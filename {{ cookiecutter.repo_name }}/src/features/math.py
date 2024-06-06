import numpy as np


def get_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the derivative of two arrays with normalization.

    Parameters:
    x (np.ndarray): The first array.
    y (np.ndarray): The second array.

    Returns:
    np.ndarray: The array of the derivative.
    """
    # Make sure it's an array
    x = np.array(x)
    y = np.array(y)

    # Calculate percentage differences
    dx_perc = np.diff(x) / x[:-1]
    dy_perc = np.diff(y) / y[:-1]

    # Calculate the derivative
    derivative = np.round((dy_perc / dx_perc), 4)

    # Since np.diff reduces the length by 1, prepend a NaN to align with the original length
    derivative = np.insert(derivative, 0, np.nan)

    # Convert the result back to a pandas Series
    return derivative

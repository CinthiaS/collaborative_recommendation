def calc_rmse(y_pred, y_true):
    """
    Calculate root mean squared error.

    Args:
      y_pred: float
          predict rating value
      y_true: float
          true rating value
    Returns: 
    
    rmse: float
        root mean squared error between predict rating values and true ratings values
    """
    
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def calc_mae(y_pred, y_true):
    """
    Calculate mean absolute error.

    Args:
      y_pred: float
          predict rating value
      y_true: float
          true rating value
    
    Returns: 
    
    mae: float
        mean absolute error between predict rating values and true ratings values
    """
    return np.absolute(y_pred - y_true).mean()
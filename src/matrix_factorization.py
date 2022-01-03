import numpy ad np

from numpy import dot
from numpy.linalg import norm

import metrics
import prepare_data
import utils

def scheduler(epoch, alpha):
    """
    Learning rating scheduler.
    
    Args:
      epoch: int
          actual epoch in training
      alpha: int
          learning rating

    Returns: 

    learning rate update 
    """
    
    if epoch < 2:
        return alpha
    else:
        return 0.05

def update_weights(
    pu, qi, user, item, eui, num_factors, alpha=0.1, lamb=0.02):
    """
    Update user and item matrix weights.

    Args:
      pu: np.array
          user matrix
      qi: np.array
          item matrix
      user: str
          user code
      item: str
          item code
      eui: float
          predict error
      num_factor: int
          number of latent factors
      alpha: float
          learning rate

    Returns: 

    pu: np.array
        user matrix updated
    qi: np.array
          item matrix updated
    """
    
    for k in range(num_factors):
        
        puf = alpha * (eui * qi[k, item] - lamb * pu[user, k])
        qif = alpha * (eui * pu[user, k] - lamb * qi[k, item]) 
        
        pu[user, k] +=  puf
        qi[k, item] +=  qif       
            
    return pu, qi

def update_bias(
    bu, bi, user, item, eui, alpha, lamb):
    """"
    Update bias weights.

    Args:
      pu: np.array
          user matrix
      qi: np.array
          item matrix
      user: str
          user code
      item: str
          item code
      eui: float
          predict error
      num_factor: int
          number of latent factors
      alpha: float
          learning rate

    Returns: 

    pu: np.array
        user matrix updated
    qi: np.array
          item matrix updated
    """
    
    buf = alpha * (eui * bi[item] - lamb * bu[user])
    bif = alpha * (eui * bu[user] - lamb * bi[item])
    
    bu[user] += buf
    bi[item] += bif
    
    return bu, bi


def one_svd_predict(
    pu, qi, bu, bi, user, item, num_factors, mean):
    """
    Predict rating to a pair user item.

    Args:
      pu: np.array
          user matrix
      qi: np.array
          item matrix
      user: str
          user code
      item: str
          item code
      num_factor: int
          number of latent factors
      mean: float
          mean ratings of all users

    Returns: 

    float: predicted value      
    """
    
    return sum([pu[user, k] * qi[k, item] for k in range(num_factors)]) + mean + bu[user] + bi[item]

def svd_predict(
    X, pu, qi, bu, bi, num_factors, mean_ratings, max_pred=5, min_pred=1):
    """
    Predict rating for all pairs users items.

    Args:
      X: np.array
          rating matrix
      pu: np.array
          user matrix
      qi: np.array
          item matrix
      num_factors: int
          number of latent factors
      mean: float
          mean ratings of all users
      max_pred: int
          highest possible prediction
      min_pred: int
          lowest possible prediction

    Returns: 

    y_pred: np.array
        predict values
    y_true: np.array
        targets values
    """
    
    y_pred = []
    y_true = []
    
    for idx in range(X.shape[0]):
        
        user, item, rui = X[idx, 0], X[idx, 1], X[idx, 2]

        mean = mean_ratings.get(item)

        pred = one_svd_predict(pu, qi, bu, bi, user, item, num_factors, mean)
        
        y_pred.append(pred)
        y_true.append(rui)
    
    return np.asarray(y_pred), np.asarray(y_true)

def inicialization(n, m, num_factors):
    """
    Initializes weights.

    Args:
      n: int
          number of unique users in ratings data
      m: int
          number of unique itens in ratings data
      num_factors: int
          number of latent factors

    Returns: 

    pu: np.array
        user matrix random initialize
    qi: np.array
        item matrix random initialize
    bu: np.array
        bias vector for user weights initialize with zeros
    bi: np.array
        bias vector for item weights initialize with zeros
    """
    
    pu = np.random.normal(0, 0.1, (n, num_factors))
    qi = np.random.normal(0, 0.1, (m, num_factors))
    
    bu = np.zeros(n)
    bi = np.zeros(m)
    
    return pu, qi, bu, bi  

def SGD(
    X, num_factors, n, m, mean_ratings, alpha=0.001, lamb=0.002):
    """
    Stochastic Gradiend Descent.

    Args:
      X: np.array
          ratings data 
      num_factor: int
          number of latent factors
      n: int
          number of unique users in ratings data
      m: int
          number of unique itens in ratings data
      alpha: float
          learning rate

    Returns: 

    pu: np.array
        user matrix factored
    qi: np.array
        item matrix factored
    bu: np.array
        bias vector for user weights
    bi: np.array
        bias vector for item weights
    """
    
    pu, qi, bu, bi = inicialization(n, m, num_factors)
    
    qi = qi.T
    
    for idx in range(X.shape[0]):
        
        user, item, rui = X[idx, 0], X[idx, 1], X[idx, 2]
        
        # get user mean ratings
        mean = mean_ratings.get(item)

        #predict rating
        pred = one_svd_predict(pu, qi, bu, bi, user, item, num_factors, mean)
        
        #calculate error
        eui = rui - pred
        
        #update bias
        bu, bi = update_bias(bu, bi, user, item, eui, alpha, lamb)
        
        #Adjust weights
        pu, qi = update_weights(pu, qi, user, item, eui, num_factors, alpha, lamb)
        
        
    return pu, qi, bu, bi


def fit(
    X_train, mean_ratings, num_factors, n, m, alpha=0.001, lamb=0.002, epochs=20, verbose=False):
    """
    Fit Stochastic Gradiend Descent.

    Args:
      X_train: np.array
          ratings data used to create the factored matrixes
      num_factor: int
          number of latent factors
      n: int
          number of unique users in ratings data
      m: int
          number of unique itens in ratings data
      alpha: float
          learning rate
      epochs: int
          number of steps 
      verbose: boolean
          if true, show error values at all steps of training

    Returns: 

    pu: np.array
        user matrix factored
    qi: np.array
        item matrix factored
    bu: np.array
        fitted user bias vector
    bi: np.array
        fitted item bias vector
    rmse: float
        root mean squared error between predict rating values and true ratings values
    mae: float
        mean absolute error between predict rating values and true ratings values
    """
    
    for epoch in range(epochs):
        
        #alpha = scheduler(epoch, alpha)
        
        pu, qi, bu, bi = SGD(X_train, num_factors, n, m, mean_ratings, alpha=alpha, lamb=lamb)
        
        y_pred, y_true = svd_predict(X_train, pu, qi, bu, bi, num_factors, mean_ratings)
        
        rmse = metrics.calc_rmse(y_pred, y_true)
        mae = metrics.calc_mae(y_pred, y_true)
        
        if rmse < 0.01:
            break
            
        if verbose:
            print("Epoch: {} - RMSE: {:.5f} - MAE: {:.5f}".format(epoch, rmse, mae))
            
    return pu, qi, bu, bi, rmse, mae

def main_matrix_factorization(
    train_df, num_factors=100, alpha=0.001, lamb=0.02, epochs=20,
    agg_metric='mean', agg_by='ItemId', user_column='UserId',
    item_column='ItemId', rating_column='Rating', verbose=False):
    """
        1. Format df
        2. Matrix Factorization

    Args:
      X_train: np.array
          ratings data used to create the factored matrixes
      num_factors: int
          number of latent factors
      alpha: float
          learning rate
      lamb: float
          regularization factor
      epochs: int
          number of steps 
      user_column: str, defaul UserId
          column name of users
      item_column: str, default ItemId
          column name of items
      verbose: boolean, default False
          if true, show all steps

    Returns: 

    pu: np.array
        user matrix factored
    qi: np.array
        item matrix factored
    bu: np.array
        user bias vector
    bi: np.array
        item bias vector
    dict_users: 
        dictionary mapped users and your new code
    dict_items: 
        dictionary mapped items and your new code 
    r_df: pd.DataFrame
        Format data
    rmse: float
        root mean squared error between predict rating values and true ratings values
    mae: float
        mean absolute error between predict rating values and true ratings values
    """
    
    users = pd.unique(train_df[user_column]).tolist()
    items = pd.unique(train_df[item_column]).tolist()

    num_users = len(users)
    num_items = len(items)

    if verbose:
        print("\tFormatting data")
        
    r, r_df, dict_users, dict_items = prepare_data.create_df (
        train_df.copy(), users, items, user_column=user_column, item_column=item_column)

    if verbose:
        print("\tAggregating ratings")
        
    _, dict_mean_ratings = utils.agg_ratings(
        train_df, dict_items, agg_metric=agg_metric, agg_by=agg_by, rating_column=rating_column)
      
    if verbose:
        print("\tFit SVD... ...\n")
        
    pu, qi, bu, bi, rmse, mae = fit(
        r, mean_ratings=dict_mean_ratings, num_factors=num_factors, n=num_users,
        m=num_items, alpha=alpha, lamb=lamb, epochs=epochs, verbose=verbose)
    
    return pu, qi, bu, bi, dict_users, dict_items, dict_mean_ratings, r_df, rmse, mae
import pandas as pd
import numpy as np

def predict_one_rating(row, pu, qi, bu, bi, dict_users, dict_items, mean_ratings, agg_by="ItemId"):
    """
    Predict rating for a pair user-item

    Args:
      row: pd.Series
          line of pandas df
      pu: np.array
          user matrix
      qi: np.array
          item matrix
      bu: np.array
          user bias vector
      bi: np.array
          item bias vector
      user_mean: float
          mean ratings of all users

    Returns: 

    pred: float
        predicted value      
    """

    user_target = dict_users.get(row[0])
    item_target = dict_items.get(row[1])
    
    if agg_by == 'ItemId':
        user_mean = mean_ratings.get(item_target)
    elif agg_by == 'UserId':
        user_mean = mean_ratings.get(user_target)
        
    pred = np.dot(pu[user_target], qi[item_target])
    pred = clip(pred + user_mean + bu[user_target] + bi[item_target])
    
    return pred

def predict_all_ratings(
    test,  mean_ratings, pu, qi, bu, bi, dict_users, dict_items,
    user_column='UserId', item_column='ItemId', rating_column='Rating', agg_by="ItemId"):
    """
    Predict ratings for all pair user-item in dataframe.

    Args:
      test: pd.DataFrame, columns default, [UserId, ItemId]
          train data
      user_mean: float
          mean ratings of all users
      train: pd.DataFrame, columns default, [UserId, ItemId, Ratings]
          train data
      pu: np.array
          user matrix
      qi: np.array
          item matrix
      bu: np.array
          user bias vector
      bi: np.array
          item bias vector
      dict_users: float
          mean ratings of all users
      dict_users: float
          mean ratings of all users
      user_column: str, defaul UserId
          column name of users
      item_column: str, default ItemId
          column name of items

    Returns: 

    DataFrame with all predict values     
    """
    
    test['ui'] = list(zip(test[user_column], test[item_column]))
    
    vfunc = np.vectorize(
        predict_one_rating,
        excluded=['pu', 'qi', 'bu', 'bi', 'dict_users', 'dict_items', 'mean_ratings', 'agg_by'])
    
    test[rating_column] = vfunc(
        row=test['ui'], pu=pu, qi=qi, bu=bu, bi=bi, dict_users=dict_users,
        dict_items=dict_items, mean_ratings=mean_ratings, agg_by=agg_by)
    
    return test[[user_column, item_column, rating_column]]
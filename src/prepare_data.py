import pandas as pd
import numpy as np

def create_df (df, users, items, user_column='UserId', item_column='ItemId'):
    """
    Creates a new rating dataframe where all users and itens are mapped for a continuous integer value
    and.

    Args:
      df: pd.DataFrame
          ratings_df
      user: list
          list with unique users code
      items: str
          list with unique items code
      user_column: str, defaul UserId
          column name of users
      item_column: str, default ItemId
          column name of items

    Returns: 

    df: 
        pandas DataFrame with all users items ratings
    dict_users: 
        dictionary mapped users and your new code
    dict_items: 
        dictionary mapped items and your new code     
    """
    
    dict_users = dict(zip(users, range(len(users))))
    dict_items = dict(zip(items, range(len(items))))

    df[user_column] = df[user_column].map(dict_users)
    df[item_column] = df[item_column].map(dict_items)

    df = df.fillna(-1)
    
    return np.asarray(df), df, dict_users, dict_items 


def agg_ratings(df, map_dict, agg_metric='mean', agg_by='ItemId', rating_column='Rating'):
    """
    Aggregated ratings values by user or by items using metric mean or median.

    Args:
      df: pd.DataFrame
          ratings_df
      map_dict: dictionary
          dictionary of users or items created by create_df() function
      agg_metric: str, default 'mean'
          used metric to aggregate ratings
      agg_by: str, default 'ItemId'
          column name used do aggregate ratings.
          If aggregate column is ItemId, map_dict must be dict items.
      rating_column: str, default 'Rating'
          column name of ratings

    Returns: 

    mean_ratings: 
        dataframe with aggregated ratings
    dict_mean_ratings: 
        dictionary with aggregated ratings 
    """
    
    grouped_ratings = df.groupby(agg_by)
    
    if agg_metric == "mean":
        mean_ratings = grouped_ratings.mean().reset_index()
    elif agg_metric == "median":
        mean_ratings = grouped_ratings.median().reset_index()
        
    mean_ratings[agg_by] = mean_ratings[agg_by].map(map_dict)
    
    mean_ratings[agg_by] = mean_ratings[agg_by].apply(round_)
    
    dict_mean_ratings = dict(zip(mean_ratings[agg_by], mean_ratings[rating_column]))
    
    return mean_ratings, dict_mean_ratings


def normalize_ratings(df, grouped_df, rating_column='Rating', agg_by='UserId'):
    """
    Normalize ratings.

    Args:
      df: pd.DataFrame
          ratings_df
      grouped_df: df
          dataframe with aggregated ratings
      agg_by: str, default 'ItemId'
          column name used do aggregate ratings.
          If aggregate column is ItemId, map_dict must be dict items.

    Returns: 

    grouped_df: 
       dataframe with normalize ratings
    """
    
    normalized_df = grouped_df.merge(df, on=agg_by)
    normalized_df[rating_column] =  normalized_df['Rating_y'] - normalized_df['Rating_x']  
    
    normalized_df = normalized_df.reset_index(drop=True)
    
    return normalized_df
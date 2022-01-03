import pandas as pd
import numpy as np

from numpy import dot

def clip(pred, max_pred=5, min_pred=1):
    """
    Clipping predict values in range [min_pred, max_pred].

    Args:
      pred : float
          predict value
      max_pred : int
          max value of rating
      min_pred : int
          min value of rating

    Returns: 

    pred: int
        predict value cliped
    """
    
    pred = max_pred if pred > max_pred else pred
    pred = min_pred if pred < min_pred else pred
    
    return pred

def train_test_split(X, test_split=0.2):
    """
    Split dataset in train and test.

    Args:
      X: np.array
          ratings 
      test_split: float
          percentage of data for the test dataset 
    
    Returns: 
    
    train: np.array
        train dataset
    test:
        test dataset
    """
    
    size = int(X.shape[0])
    all_indexes = list(range(size))
                       
    indexes_test = list(np.random.choice(np.arange(0,size), int(size*test_split), replace=False))
    indexes_train = set(all_indexes) - set(indexes_test)
    
    train = X.loc[indexes_train].reset_index(drop=True)
    test = X.loc[indexes_test].reset_index(drop=True)
    
    return train, test

def get_ratings_by_item(
    df, users_df, items, rating_column='Rating', user_column='UserId', item_column='ItemId'):
    """
    Get ratings of the items.

    Args:
      df: pd.DataFrame
          ratings df
      users_df: pd.DataFrame
          Dataframe with all unique users
      items : list of string
          items id
      rating_column: str, default Rating
          name of column with rating values
      user_column: str, default UserId
          name of column with users id
      item_column: str, default ItemId
      
    Returns: 
    
    list of strings with all items ids evaluated by user target
    """
    
    aux = df.loc[df[item_column].isin(items)]
    aux = aux.pivot_table(index=user_column, columns=item_column, values=rating_column)
    users_df = users_df.merge(aux, how='left', on=[user_column])
    users_df = users_df.fillna('0')

    return users_df

def get_items_by_users(
    df, users, user_column='UserId', item_column='ItemId'):
    """
    Get items id evaluates by user target.

    Args:
      df: ps.DataFrame
          predict rating value
      user: str
          user id
      user_column: str, default UserId
          name of column with users id
      item_column: str, default ItemId
      
    Returns: 
    
    list of strings with all items ids evaluated by user target
    """
    
    items = [list(df.loc[df[user_column] == user][item_column]) for user in users]
    items = sorted(set(sum(items, [])))
    
    return items


def get_common_users_items(
    df, item, user_column='UserId', item_column='ItemId'):
    """
    Get the ids of users that evaluate a item.

    Args:
      df: ps.DataFrame
          predict rating value
      item: str
          item id
      user_column: str, default UserId
          name of column with users id
      item_column: str, default ItemId
          name of column with items id
    Returns: 
    
    list of strings with all users ids that evaluate a target item
    """
    
    return list(df.loc[df[item_column] == item][user_column])

def cosine_similarity(x, y):
    """
    Calculates cosine similarity between two vectors.

    Args:
      x: np.array
         vector of values
      y: np.array
         vector of valuesitem id
         
    Returns: 
    
    cosine similarity between x and y
    """
    
    return np.dot(x, y)/(norm(x)*norm(y))

def get_item_neighbors(item_sim, item_target, ratings_df):
    """
    Retrieve an item's neighbors

    Args:
      item_target: str
         target item of neighborhood analysis 
      ratings_df: pd.DataFrame
          database with users ratings for itens
         
    Returns: 
    neighbors: pd.DataFrame
        target item neighbors
    """
    
    common_users = get_common_users_items(ratings_df, item=item_target)
    
    if common_users == []:
        return None
        
    candidate_items = get_items_by_users(ratings_df, users=common_users)
    
    if candidate_items == []:
        return None
    
    neighbors = item_sim[item_sim['items'].apply(lambda x: item_target in x)]
    neighbors = neighbors.loc[neighbors['items'].apply(lambda x: any(candidate in x for candidate in candidate_items))]
    neighbors = neighbors.reset_index(drop=True)
    
    return neighbors

def round_(x):
    """
    Rounds the value to the integer more close.

    Args:
      x: float
          Value to be rounded
    Returns: 
        integer value   
    """
    
    return int(x + 0.5)

def format_and_save(
    result, name_file='out.csv', user_column='UserId', item_column='ItemId',
    rating_column='Rating', round_preds=False):

    """
    Format predictions
    """
    
    if round_preds:
        result[rating_column] = result[rating_column].apply(round_)
    else:
        result[rating_column] = result[rating_column]
        
    new_result = pd.DataFrame(np.vstack([result.columns, result]))
    text = list(new_result[0] + ":" + new_result[1] + "," + new_result[2].astype(str))
    
    result = "\n".join(text)
    
    return result
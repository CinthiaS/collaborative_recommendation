import pandas as pd
import numpy as np
import time
import sys

from numpy import dot
from numpy.linalg import norm

import predict
import matrix_factorization


def main(
    train_file='ratings.csv', test_file='targets.csv', num_factors=40, alpha=0.001, lamb=0.002, epochs=2,
    agg_metric='mean', agg_by='ItemId', user_column='UserId', item_column='ItemId', rating_column='Rating',
    verbose=False):
    
    #start_time = time.time()
    
    columns = '{}:{}'.format(user_column, item_column)
    
    if verbose:
        print("\n\nRead datasets")
        
    train = pd.read_csv(train_file)
    train[columns] = train[columns].str.split(':')
    train_df = pd.DataFrame(train[columns].to_list(), columns=[user_column, item_column])
    train_df[rating_column] = train[rating_column]

    test = pd.read_csv(test_file)
    test[columns] = test[columns].str.split(':')
    test_df = pd.DataFrame(test[columns].to_list(), columns=[user_column, item_column])

    if verbose:
        print("Matrix Factorization")
        
    pu, qi, bu, bi, dict_users, dict_items, dict_mean_ratings, r_df, rmse, mae = matrix_factorization.main_matrix_factorization(
        train_df, num_factors=num_factors, alpha=alpha, epochs=epochs, lamb=lamb,
        agg_metric=agg_metric, verbose=verbose)
    qi = qi.T
    
    if verbose: 
        print("Predict Ratings")
        
    predictions = predict.predict_all_ratings(
        test_df, dict_mean_ratings, pu, qi, bu, bi, dict_users=dict_users,
        dict_items=dict_items, user_column=user_column, item_column=item_column,
        rating_column=rating_column)

    if verbose:
        print("Save Results")
        
    output = format_and_save(predictions.copy())
    print(output)
    
    #elapsed_time = (time.time() - start_time)/60
    #print("Time execution: {} minutes".format(elapsed_time))
    
if __name__ == "__main__":


    train_file = sys.argv[1]
    test_file = sys.argv[2]

    main(train_file=train_file, test_file=test_file)
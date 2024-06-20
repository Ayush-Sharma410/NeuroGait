import numpy as np

def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

import numpy as np


def compute_epv(df, original_features):
    """
    Compute the Events Per Variable (EPV) for a dataframe:
    - For classification tasks (when `min_samples_per_class` is not NaN):
        EPV = min_samples_per_class / number of original features.
    - For regression tasks (when `min_samples_per_class` is NaN):
        EPV = num_samples / number of original features.

    Parameters
    ----------
    df : pandas.DataFrame
    original_features : pandas.Series (lists)

    Returns
    -------
    pandas.Series
        EPV values for all rows
    """
    num_features = original_features.apply(len)

    return (
        df["min_samples_per_class"]
        .where(df["min_samples_per_class"].notna(), df["num_samples"])
        / num_features
    )


def compute_validity(selected_features, max_features):
    """
    Compute validity (selection precision) for a dataframe.

    Parameters
    ----------
    selected_features : pandas.Series (lists)
    max_features : pandas.Series (ints)

    Returns
    -------
    pandas.Series
        Validity values for all rows
    """
    TP = selected_features.apply(
        lambda features: sum(not f.startswith("__noise_feature_") for f in features)
    )

    return TP / max_features


def _getStability(Z):
    ''' 
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate as given in Definition 4 in  [1].
    https://github.com/nogueirs/JMLR2018/blob/master/python/stability/__init__.py.
    
    INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d).
           Each row of the binary matrix represents a feature set, where a 1 at the f^th position 
           means the f^th feature has been selected and a 0 means it has not been selected.
           
    OUTPUT: The stability of the feature selection procedure
    '''
    Z=_checkInputType(Z)
    M,d=Z.shape
    hatPF=np.mean(Z,axis=0)
    kbar=np.sum(hatPF)
    denom=(kbar/d)*(1-kbar/d)
    return 1-(M/(M-1))*np.mean(np.multiply(hatPF,1-hatPF))/denom


def _checkInputType(Z):
    ''' This function checks that Z is of the rigt type and dimension.
        It raises an exception if not.
        OUTPUT: The input Z as a numpy.ndarray
    '''
    ### We check that Z is a list or a numpy.array
    if isinstance(Z,list):
        Z=np.asarray(Z)
    elif not isinstance(Z,np.ndarray):
        raise ValueError('The input matrix Z should be of type list or numpy.ndarray')
    ### We check if Z is a matrix (2 dimensions)
    if Z.ndim!=2:
        raise ValueError('The input matrix Z should be of dimension 2')
    return Z

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:17:45 2023

@author: rdmouhouadi
"""

from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
        
def selectFeatures(labels, features, k, stat_func = "chi2"):
    """

    Parameters
    ----------
    *labels*: TYPE
        DESCRIPTION.
    *features*: TYPE
        DESCRIPTION.
    *k*: TYPE
        DESCRIPTION.
    *stat_func*: TYPE, optional
        DESCRIPTION. The default is "chi2".

    Returns
    -------
    selected_feat : TYPE
        DESCRIPTION.

    """
        
    if stat_func == "f_classif":
        selector = SelectKBest(f_classif, k)
    elif stat_func == "f_regression":
        selector = SelectKBest(f_regression, k)
    else:
        selector = SelectKBest(chi2, k)
        
    selector.fit_transform(labels, features)
        
    #Place the best labels as column names in the new dataframe (permanent change) 
    selected_feat = labels.columns[selector.get_support(indices= True)]
        
    return selected_feat

        
        
        
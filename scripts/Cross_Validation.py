# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:35:59 2023

@author: rdmouhouadi
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV

#RandomForestRegressor model
RFR = RandomForestRegressor()
RFR_params = {
                'n_estimator': [], 
                'bootstrap': [],
                'ccp_alpha': [],
                'criterion': [],
                'max_depth': [],
                'max_features': [],
                'max_leaf_nodes': [],
                'max_samples': [],
                'min_impurity_decrease': [],
                'min_samples_leaf': [],
                'min_samples_split': [],
                'min_weight_fraction_leaf': [],
                'n_jobs': [],
                'oob_score': [],
                'random_state': [],
                'verbose': [],
                'warm_start': []
            }

#DecisionTreeRegressor model
DTR = DecisionTreeRegressor()
DTR_params = {
                'ccp_alpha': [],
                'criterion': [],
                'max_depth': [],
                'max_features': [],
                'max_leaf_nodes': [],
                'min_impurity_decrease': [],
                'min_samples_leaf': [],
                'min_samples_split': [],
                'min_weight_fraction_leaf': [],
                'random_state': [],
                'splitter': [],
            }

#GradientBoostingRegressor model
GBR = GradientBoostingRegressor()
GBR_params = {
                'n_estimators': [],
                'alpha': [],
                'ccp_alpha': [],
                'criterion': [],
                'init': [],
                'learning_rate': [],
                'loss': [],
                'max_depth': [],
                'max_features': [],
                'max_leaf_nodes': [],
                'min_impurity_decrease': [],
                'min_samples_leaf': [],
                'min_samples_split': [],
                'min_weight_fraction_leaf': [],
                'n_iter_no_change': [],
                'random_state': [],
                'subsample': [],
                'tol': [],
                'validation_fraction': [],
                'verbose': [],
                'warm_start': []
            }

#KNeighborsRegressor model
KNR = KNeighborsRegressor()
KNR_params = {
                'n_neighbors': [],
                'algorithm': [],
                'leaf_size': [],
                'metric': [],
                'metric_params': [],
                'n_jobs': [],
                'p': [],
                'weights': []
            }

#SVR model
SVR = SVR()
SVR_params = {
                'C': [],
                'cache_size': [],
                'coef0': [],
                'degree': [],
                'epsilon': [],
                'gamma': [],
                'kernel': [],
                'max_iter': [],
                'shrinking': [],
                'tol': [],
                'verbose': [],
            }



def selectModel():
    """
    This function accounts for ML models available for the project

    Returns
    -------
    Classes (different ML models)

    """
    return {
            'RandomForestRegressor': RFR,
            'DecisionTreeRegressor': DTR,
            'GradientBoostingRegressor': GBR,
            'KNeighborsRegressor': KNR,
            'SVR': SVR
        }

def selectHyperparam(model_Name = 'RandomForestRegressor' ):
    """
    This function allows the selection of models hyperparameters

    Parameters
    ----------
    model_Name : TYPE: string, optional
        DESCRIPTION. The default is 'RandomForestRegressor'.

    Returns
    -------
    dict
        DESCRIPTION: a dictionnaire containing a list of optimization values 
        for the choosen ML model

    """
    if model_Name == 'RandomForestRegressor':
        return RFR_params
    
    elif model_Name == 'DecisionTreeRegressor':
        return DTR_params
    
    elif model_Name == 'GradientBoostingRegressor':
        return GBR_params
    
    elif model_Name == 'KNeighborsRegressor':
        return KNR_params
    
    else:
        return SVR_params

def optimizeModel(opt_method, model, X_train, Y_train, X_test, Y_test):
    
    """
    this function returns the score of the new optimized model

    Parameters
    ----------
    opt_method : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    X_train : TYPE
        DESCRIPTION.
    Y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    best_params : TYPE
        DESCRIPTION.

    """
   
    if opt_method == "GridSearchCV":
        
        if model =='RandomForestRegressor':
            
            grid_search = GridSearchCV(RFR, RFR_params, X_train, Y_train)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
        
        elif model == 'DecisionTreeRegressor':
  
            grid_search = GridSearchCV(DTR, DTR_params, X_train, Y_train)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
        
        elif model == 'GradientBoostingRegressor':
      
            grid_search = GridSearchCV(GBR, GBR_params, X_train, Y_train)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
        
        elif model == 'KNeighborsRegressor':
           
            grid_search = GridSearchCV(KNR, KNR_params, X_train, Y_train)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
        
        else:
           
            grid_search = GridSearchCV(SVR, SVR_params, X_train, Y_train)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
    
    elif opt_method == "CrossValidation":
        
        if model =='RandomForestRegressor':
            
            cross_val = cross_validate(RFR, RFR_params, X_train, Y_train)
            cross_val.fit(X_train, Y_train)
            best_params = cross_val.best_params_
        
        elif model == 'DecisionTreeRegressor':
            
            cross_val = cross_validate(DTR, DTR_params, X_train, Y_train)
            cross_val.fit(X_train, Y_train)
            best_params = cross_val.best_params_
        
        elif model == 'GradientBoostingRegressor':
       
            cross_val = cross_validate(GBR, GBR_params, X_train, Y_train)
            cross_val.fit(X_train, Y_train)
            best_params = cross_val.best_params_
        
        elif model == 'KNeighborsRegressor':
       
            cross_val = cross_validate(KNR, KNR_params, X_train, Y_train)
            cross_val.fit(X_train, Y_train)
            best_params = cross_val.best_params_
        
        else:
           
            cross_val = cross_validate(SVR, SVR_params, X_train, Y_train)
            cross_val.fit(X_train, Y_train)
            best_params = cross_val.best_params_
    
    else:
        if model =='RandomForestRegressor':
            
            rand_search = RandomizedSearchCV(RFR, RFR_params, X_train, Y_train)
            rand_search.fit(X_train, Y_train)
            best_params = rand_search.best_params_
        
        elif model == 'DecisionTreeRegressor':
            
            rand_search = RandomizedSearchCV(DTR, DTR_params, X_train, Y_train)
            rand_search.fit(X_train, Y_train)
            best_params = rand_search.best_params_
        
        elif model == 'GradientBoostingRegressor':
       
            rand_search = RandomizedSearchCV(GBR, GBR_params, X_train, Y_train)
            rand_search.fit(X_train, Y_train)
            best_params = rand_search.best_params_
            return best_params
        
        elif model == 'KNeighborsRegressor':
       
            rand_search = RandomizedSearchCV(KNR, KNR_params, X_train, Y_train)
            rand_search.fit(X_train, Y_train)
            best_params = rand_search.best_params_
        
        else:
           
            rand_search = RandomizedSearchCV(SVR, SVR_params, X_train, Y_train)
            rand_search.fit(X_train, Y_train)
            best_params = rand_search.best_params_
    
    model.set_params(**best_params)
    model.fit(X_train, Y_train)
    
    new_score = model.fit(X_test, Y_test)
    return new_score
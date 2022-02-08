import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import KFold
import itertools

"""
The function obtain the model fitting results
feature_set: is the collection of input predictors used in the model
data: is the dataframe containing all data
y: is the response vector
@return: a Series of quantities related to the model for model evaluation and selection
"""
def modelFitting(y, feature_set, data):
    # Fit model on feature_set and calculate RSS
    formula = y + '~' + '+'.join(feature_set)

    # fit the regression model
    model = smf.ols(formula=formula, data=data).fit()
    return model;


"""
The function obtain the results given a regression model feature set
feature_set: is the collection of input predictors used in the model
data: is the dataframe containing all data
y: is the response vector
@return: a Series of quantities related to the model for model evaluation and selection
"""
def processSubset(y, feature_set, data):
    # Fit model on feature_set and calculate RSS
    try:
        regr = modelFitting(y, feature_set, data);
        R2 = regr.rsquared;
        ar2 = regr.rsquared_adj;
        sse = regr.ssr;
        return {"model":feature_set, "SSE": sse, "R2":-R2, "AR2": -ar2, "AIC": regr.aic, "BIC": regr.bic, "Pnum": len(feature_set)}
    except:
        return {"model": ["1"], "SSE": float("inf"), "R2": 0, "AR2": 0, "AIC": float("inf"), "BIC": float("inf"), "Pnum": 0}

"""
The function find the regression results for all predictor combinations with fixed size
k: is the number of predictors (excluding constant)
data: is the dataframe containing all data
X: is the predictor name list
y: is the response vector
@return: a dataframe containing the regression results of the evaluated models
"""
def getAll(k, y, X, data):
    results = []
    # evaluate all the combinations with k predictors
    for combo in itertools.combinations(X, k):
        results.append(processSubset(y, combo, data))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results);
    models['Pnum'] = k;
    print("Processed ", models.shape[0], "models on", k, "predictors")
    # Return the best model, along with some other useful information about the model
    return models


"""
The function find the Mallow's Cp based on the full model and existing regression results
models: is the dataframe containing the regression results of different models
fullmodel: is the model containing all predictors to calculate the Cp statistic
@return: a dataframe of models with Cp statistics calculated
"""
def getMallowCp(models, fullmodel):
    nobs = fullmodel.nobs;
    sigma2 = fullmodel.mse_resid;
    models['Cp'] = models['SSE']/sigma2 + 2*(models['Pnum']+1) - nobs
    return models

"""
The function find the best models among all lists using the criterion specified
models: is the dataframe containing the regression results of different models
criterion: is the selection critierion, can take values "AIC", "BIC", "Cp", "AR2", "R2" (only for educational purpose)
k: is the number of predictors as the constraints, if None, all models are compared
@return: the best model satisfied the requirement
"""
def findBest(models, criterion='AIC', k=None):
    # the list of models with given predictor number
    if k is None:
        submodels = models;
    else:
        submodels = models.loc[models['Pnum']==k,];

    # Use the criterion to find the best one
    bestm = submodels.loc[submodels[criterion].idxmin(0), ];
    # return the selected model
    return bestm;


"""
The function use forward selection to find the best model given criterion
models: is the dataframe containing the regression results of different models
X: is the name list of all predictors to be considered
y: is the response vector
data: is the dataframe containing all data
criterion: is the selection critierion, can take values "AIC", "BIC", "Cp", "AR2", "R2" (only for educational purpose)
fullmodel: is the full model to evaluate the Cp criterion
@return: the best model selected by the function
"""
def forward(y, X, data, criterion="AIC", fullmodel = None):
    remaining = X;   
    selected = []

    basemodel = processSubset(y, '1', data)
    current_score = basemodel[criterion]
    best_new_score = current_score;

    while remaining: # and current_score == best_new_score:
        scores_with_candidates = []
        
        for candidate in remaining:
            # print(candidate)
            scores_with_candidates.append(processSubset(y, selected+[candidate], data))
                        
        models = pd.DataFrame(scores_with_candidates)

        # if full model is provided, calculate the Cp
        if fullmodel is not None:
            models = getMallowCp(models, fullmodel);
            
        best_model = findBest(models, criterion, k=None)
        best_new_score = best_model[criterion];

        if current_score > best_new_score:
            selected = best_model['model'];
            remaining = [p for p in X if p not in selected]
            print(selected)
            current_score = best_new_score
        else :
            break;
            
    model = modelFitting(y, selected, data)
    return model

"""
The function use backward elimination to find the best model given criterion
models: is the dataframe containing the regression results of different models
X: is the name list of all predictors to be considered
y: is the response vector
data: is the dataframe containing all data
criterion: is the selection critierion, can take values "AIC", "BIC", "Cp", "AR2", "R2" (only for educational purpose)
fullmodel: is the full model to evaluate the Cp criterion
@return: the best model selected by the function
"""
def backward(y, X, data, criterion="AIC", fullmodel = None):
    remaining = X;
    removed = []
    basemodel = processSubset(y, remaining, data)
    current_score = basemodel[criterion]
    best_new_score = current_score;

    while remaining: # and current_score == best_new_score:
        scores_with_candidates = []
        
        for combo in itertools.combinations(remaining, len(remaining)-1):
            scores_with_candidates.append(processSubset(y, combo, data))
                        
        models = pd.DataFrame(scores_with_candidates)
        # if full model is provided, calculate the Cp
        if fullmodel is not None:
            models = getMallowCp(models, fullmodel);
            
        best_model = findBest(models, criterion, k=None)
        best_new_score = best_model[criterion];

                
        if current_score > best_new_score:
            remaining = best_model['model'];
            removed = [p for p in X if p not in remaining]
            print(removed)
            current_score = best_new_score
        else :
            break;
            
    model = modelFitting(y, remaining, data)
    return model



"""
The function compute the cross validation results 
X: is the dataframe containing all predictors to be included
y: is the response vector
data: is the dataframe of all data
kf: is the kfold generated by the function
@return: the cross validated MSE 
"""
def CrossValidation(y, X, data, kf):
    results = []
    formula = y + '~' + '+'.join(X)

    # evaluate all accuracy based on the folds
    for train_index, test_index in kf:
        d_train, d_test = data.ix[train_index,], data.ix[test_index,]

        # fit the model and evaluate the prediction
        lmfit = smf.ols(formula=formula, data=d_train).fit()
        pred = lmfit.predict(d_test)
        prederror = ((pred - d_test[y]) ** 2).mean();
        results.append(prederror);
        
    # Wrap everything up in a nice dataframe
    return results;

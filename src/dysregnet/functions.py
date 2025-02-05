import pandas as pd
from scipy.stats import zscore, norm, combine_pvalues
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
from scipy import stats
import statsmodels.stats.multitest as mt
import statsmodels.api as sm
#_____________
import pickle
import joblib
import time
import os
from itertools import product
from scipy.stats import combine_pvalues
from .linearmodel import LinearModel

#_____________

def process_data(data):
       
        
        # process covariates and design matrix
        
        if data.CatCov is not None:
            all_covariates = data.CatCov + data.ConCov
        else:
            all_covariates = None

        if not all_covariates or len(data.meta)==1:
                    print('You did not input any covariates in CatCov or ConCov parameters, proceeding without them.')
                    cov_df=None

        elif data.meta is not None:

                # check if covariates exist in meta
                if not set(all_covariates).issubset(data.meta.columns):
                    raise ValueError("Invalid elements in CatCov or ConCov. Please check that all covariates names (continuous or categorials) are in the meta DataFrame. ")

                cov_df = data.meta[all_covariates]

                # process categorial covariate
                # drop_first is important to avoid multicollinear
                cov_df = pd.get_dummies(cov_df, columns=data.CatCov, drop_first=True, dtype=int)
                
                
                
        # z scoring
        if data.zscoring:

            # expression data
            # fit a scaler base on the control samples
            scaler = StandardScaler()
            scaler.fit(data.expression_data[data.meta[data.conCol]==0])

            # scale the expression data
            expr = pd.DataFrame(
                scaler.transform(data.expression_data),
                columns=data.expression_data.columns, 
                index=data.expression_data.index
            )
            
            # continuous confounders
            if cov_df is not None and len(data.ConCov)>0:

                # fit a scaler base on the control samples
                scaler = StandardScaler()
                scaler.fit(data.meta.loc[data.meta[data.conCol]==0,data.ConCov])

                # scale the continuous confounders data
                cov_df[data.ConCov] = scaler.transform(data.meta[data.ConCov])
            
        else:  
            expr = data.expression_data
        
        
        if data.meta is not None:
            #get control and case sample 
            control = data.meta[ data.meta[data.conCol]==0 ].index.values.tolist()
            case = data.meta[ data.meta[data.conCol]==1 ].index.values.tolist()
        else:
            control = None
            case = None

        return cov_df, expr, control, case



def dyregnet_model(data):
    """
    For every edge (TF -> target) the function either loads a pre-fitted model or fits a new OLS model
    using the control samples. If control samples are provided, it fits a 
    model on the control samples and compares it to the loaded model. Only if the coefficients 
    are sufficiently similar (i.e. their absolute differences are all below a specified threshold) does it
    proceed with testing on the case samples.
    
    Parameters:
      data: an object (or namespace) with the following attributes:
         - expr: DataFrame of gene expression values (samples x genes)
         - cov_df: (optional) DataFrame with covariate information (samples x covariates)
         - control, case: indexers for control and case samples
         - GRN: DataFrame (or similar) of edges (each row with TF and target)
         - load_model: boolean, whether to load pre-computed models from file
         - model_dir: directory where models are stored (if load_model is True)
         - skip_poor_fits: boolean, if True perform extra checks on the control model fit
         - direction_condition: boolean, if True use one-sided p-values based on the sign of the TF coefficient
         - bonferroni_alpha: significance level for multiple testing correction
         - similarity_threshold: maximum allowed absolute difference between coefficients of the two models.
    
    Returns:
      results_df: DataFrame with z-scores (one row per case sample) for each edge that passed the checks
     
    """
    case_flag = True

    # Prepare data: merge covariates if provided
    if data.cov_df is not None:
        control = pd.merge(data.cov_df.loc[data.control], data.expr, left_index=True, right_index=True).drop_duplicates()
        case = pd.merge(data.cov_df.loc[data.case], data.expr, left_index=True, right_index=True).drop_duplicates()
        covariate_name = list(data.cov_df.columns)

    elif data.load_model:
        if data.case is None:
            case_flag = False
            control = data.control
            case = data.expr
        else: 
            control = data.expr.loc[data.control]
            case = data.expr.loc[data.case]
        covariate_name = []
    else:  # When GRN is provided but no covariates
        control = data.expr.loc[data.control]
        if len(control) > 3:
            print("Warning: You have more than 3 control samples")
        case = data.expr.loc[data.case]

    # The dictionary "edges" will store z-scores for each edge.
    edges = {}
    edges['patient id'] = case.index.tolist()
    model_stats = {}  # To collect model stats per edge

    found = 0
    notfound = 0
    skipped = 0
    notskipped = 0

    # Loop over every edge (TF -> target)
    for tup in tqdm(data.GRN.itertuples(), desc="Processing edges"):
        edge = (tup[1], tup[2])
        # Skip self-loops
        if edge[0] == edge[1]:
            continue

        # Initialize "results" to hold the model we will use.
        results = None

        # ===============================================================
        # CASE 1: A pre-existing model is available (data.load_model == True)
        # ===============================================================
        if data.load_model:
            filename = os.path.join(data.model_dir, f"{edge[0]}_{edge[1]}.pkl")
            try:
                loaded_results = LinearModel.load(filename)
                found += 1
            except FileNotFoundError:
                #print(f"Model file not found for edge {edge}. Skipping.")
                notfound += 1
                continue

            # If control samples are available and you wish to check the model’s quality,
            # then also fit a model using the control data and compare it to the loaded model.
            if control is not None:
                if len(control) > 3 and case_flag:
                    # -----------------------------------------------
                    # (A) Fit the control model using the control data.
                    # -----------------------------------------------
                    x_train_control = control[[edge[0]] + covariate_name]
                    x_train_control = sm.add_constant(x_train_control, has_constant='add')
                    y_train_control = control[edge[1]].values
                    #model = LinearModel(predictors=[edge[0]] + covariate_name, target=edge[1])
                    #results_control = model.train(x_train_control, y_train_control)
                    model = sm.OLS(y_train_control, x_train_control)
                    results_control = model.fit()
                    #results_control = model_control.fit()

                    # Compute residuals for the control model:
                    residuals_control = y_train_control - results_control.predict(x_train_control)
                    mean_res_control = np.mean(residuals_control)
                    std_res_control = np.std(residuals_control)
                    if std_res_control == 0:
                        std_res_control = 1e-6  # safeguard against division by zero
                    # Calculate z-scores and then p-values for the control model:
                    z_scores_control = (residuals_control - mean_res_control) / std_res_control
                    pvalues_control = stats.norm.sf(np.abs(z_scores_control))
                    # Combine the p-values (using Fisher's method, for example)
                    _, combined_pvalue_control = combine_pvalues(pvalues_control, method='fisher')

                    # -----------------------------------------------
                    # (B) Evaluate the loaded model on the same control data.
                    # -----------------------------------------------
                    # Now predict using the loaded model.
                    x_train_control_reduced = x_train_control[['const', edge[0]]]
                    y_pred_loaded = loaded_results.predict(x_train_control_reduced)
                    #y_pred_loaded = loaded_results.predict(x_train_control)
                    residuals_loaded = y_train_control - y_pred_loaded
                    mean_res_loaded = np.mean(residuals_loaded)
                    std_res_loaded = np.std(residuals_loaded)
                    if std_res_loaded == 0:
                        std_res_loaded = 1e-6
                    z_scores_loaded = (residuals_loaded - mean_res_loaded) / std_res_loaded
                    pvalues_loaded = stats.norm.sf(np.abs(z_scores_loaded))
                    _, combined_pvalue_loaded = combine_pvalues(pvalues_loaded, method='fisher')


                    # -----------------------------------------------
                    # (C) Compare the two models on control data.
                    # -----------------------------------------------
                    # Here we compare the combined p-values. The logic is: if the difference
                    # between the loaded model's combined p-value and the control model's combined
                    # p-value is larger than a threshold, then the loaded model may not be describing
                    # the control data well.
                    diff = np.abs(combined_pvalue_loaded - combined_pvalue_control)
                    if diff < 0.05:
                        # The difference is too high: the loaded model doesn't match the control data well.
                        skipped += 1
                        results = loaded_results
                        continue
                    else:
                        notskipped += 1
                        # Accept the control model.
                        results = results_control
            else: 
                results = loaded_results


        # ===============================================================
        # CASE 2: No pre-existing model; fit the model using control samples.
        # ===============================================================
        else:
            x_train = control[[edge[0]] + covariate_name]
            x_train = sm.add_constant(x_train, has_constant='add')
            y_train = control[edge[1]].values
            model = sm.OLS(y_train, x_train)
            results = model.fit()

        # -----------------------------------------------------------
        # Process the case samples with the (loaded or fitted) model.
        # -----------------------------------------------------------
        # Process the case samples with the (loaded or fitted) model.
        x_test = case[[edge[0]] + covariate_name]
        x_test = sm.add_constant(x_test, has_constant='add')
        y_test = case[edge[1]].values


        # Predict on case samples and calculate residuals
        y_pred = results.predict(x_test)
        resid_case = y_test - y_pred

        # Determine the directional condition 
        if hasattr(results.params, 'iloc'):
            direction = np.sign(results.params.iloc[1])
        else:
            direction = np.sign(results.params[1])

        sides = 2  # default two-sided p-value
        cond = True
        if data.direction_condition:
            cond = (direction * resid_case) < 0
            sides = 1  # use one-sided p-values

        # Compute z-scores (avoid division by zero)
        if np.std(resid_case) == 0:
            zscore = np.zeros_like(resid_case)
        else:
            zscore = resid_case / np.std(resid_case)

        # Convert z-scores to p-values and correct for multiple testing (Bonferroni)
        pvals = stats.norm.sf(np.abs(zscore)) * sides
        _, pvals_corrected, _, _ = sm.stats.multipletests(pvals, method='bonferroni', alpha=data.bonferroni_alpha)
        valid = (pvals_corrected < data.bonferroni_alpha) & cond

        # Set insignificant z-scores to zero.
        zscore[~valid] = 0.0

        # Store the rounded z-scores for this edge.
        edges[edge] = np.round(zscore, 1)


    # Print some summary statistics.
    if data.load_model:
        total = found + notfound
        print("Ratio of found models: ", found / total if total > 0 else "N/A")
    if (skipped + notskipped) > 0:
        print("Skipped models ratio: ", skipped / (skipped + notskipped))

    if not edges:
        print("Edges dictionary is empty.")
        raise ValueError("Edges dictionary is empty. No data to process.")

    # Convert the edges dictionary into a DataFrame.
    try:
        results_df = pd.DataFrame.from_dict(edges)
        results_df = results_df.set_index('patient id')
    except ValueError as e:
        print("Error creating DataFrame from edges:", e)
        for key, value in edges.items():
            print(f"{key}: {len(value)}")
        raise

    """# Prepare model statistics DataFrame.
    model_stats_cols = (["R2"] + 
                        ["coef_" + coef for coef in ["intercept", "TF"] + covariate_name] +
                        ["pval_" + coef for coef in ["intercept", "TF"] + covariate_name])
    model_stats_df = pd.DataFrame(
        [model_stats[edge] for edge in results_df.columns],
        index=results_df.columns,
        columns=model_stats_cols
    )"""

    return results_df #, model_stats_df

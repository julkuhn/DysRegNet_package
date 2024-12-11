import pandas as pd
from scipy.stats import zscore
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
#_____________

def process_data(data):
       

        # process covariates and design matrix
        
        all_covariates = data.CatCov + data.ConCov

        if not all_covariates or len(data.meta)==1:

                # No covariate provided
                print('You did not input any covariates in CatCov or ConCov parameters, proceeding without them.')
                cov_df=None

        else:


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
        
        
        #get control and case sample 
        control = data.meta[ data.meta[data.conCol]==0 ].index.values.tolist()
        case = data.meta[ data.meta[data.conCol]==1 ].index.values.tolist()

        return cov_df, expr, control, case
    
    
    
def dyregnet_model(data):
        
        # Fit a model for every edge
        # Detect outliers and calculate zscore and pvalues
        # correct for multiple testing 
        
        # prepare data
        
        #_______
        output_dir = "/Users/juliakuhn/Desktop/uni/WS2425/Systems_BioMedicine/Project_Phase/pickle_models/" # TODO adapt to given files
        #_______
        
        # Prepare case data
        if data.cov_df is not None:
            case = pd.merge(data.cov_df.loc[data.case], data.expr, left_index=True, right_index=True).drop_duplicates()
            covariate_name = list(data.cov_df.columns)
        else:
            case = data.expr.loc[data.case]
            covariate_name = []

        edges = {}
        edges['patient id'] = list(case.index.values)
        model_stats = {}

        for tup in tqdm(data.GRN.itertuples(), desc="Processing edges"):
            edge = (tup[1], tup[2])  # Extract TF → target pair

            # Skip self-loops
            if edge[0] == edge[1]:
                continue

            # Prepare design matrix for case samples
            x_test = case[[edge[0]] + covariate_name]
            x_test = sm.add_constant(x_test, has_constant='add')  # Add intercept
            y_test = case[edge[1]].values

            # Load pre-trained model
            filename = os.path.join(output_dir, f"{edge[0]}_{edge[1]}.pkl")
            try:
                with open(filename, "rb") as file:
                    results = pickle.load(file)
            except FileNotFoundError:
                print(f"Model file not found for edge {edge}. Skipping.")
                continue

            # Save model stats
            model_stats[edge] = [results.rsquared] + list(results.params.values) + list(results.pvalues.values)

            # Predict target gene expression for case samples
            y_pred = results.predict(x_test)

            # Residuals for case samples
            resid_case = y_test - y_pred

            # Directional condition (if applicable)
            cond = True
            direction = np.sign(results.params.iloc[1])  # Direction of TF influence
            sides = 2  # Default: two-sided p-value
            if data.direction_condition:
                cond = (direction * resid_case) < 0
                sides = 1  # One-sided p-value

            # Z-score calculation (assuming a standard normal distribution for residuals)
            zscore = resid_case / resid_case.std()

            # Convert z-scores to p-values and apply multiple testing correction
            pvalues = stats.norm.sf(abs(zscore)) * sides
            pvalues = sm.stats.multipletests(pvalues, method='bonferroni', alpha=data.bonferroni_alpha)[1]
            valid = cond * (pvalues < data.bonferroni_alpha)

            # Filter insignificant z-scores
            zscore[~valid] = 0.0
            edges[edge] = np.round(zscore, 1)

        # Convert results to DataFrame
        results = pd.DataFrame.from_dict(edges)
        results = results.set_index('patient id')

        # Model stats DataFrame
        model_stats_cols = ["R2"] + ["coef_" + coef for coef in ["intercept", "TF"] + covariate_name] + \
                        ["pval_" + coef for coef in ["intercept", "TF"] + covariate_name]
        model_stats = pd.DataFrame([model_stats[edge] for edge in results.columns], index=results.columns, columns=model_stats_cols)

        return results, model_stats
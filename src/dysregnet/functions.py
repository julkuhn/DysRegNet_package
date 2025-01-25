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
import tracemalloc
import matplotlib.pyplot as plt
import os
from .linearmodel import LinearModel
#_____________

def process_data(data):
       
        pass
        # skipping this function since we dont need it for model training
        """# process covariates and design matrix
        
        all_covariates = data.CatCov + data.ConCov
       

        if not all_covariates: # or len(data.meta)==1:

                # No covariate provided
                print('You did not input any covariates in CatCov or ConCov parameters, proceeding without them.')
                cov_df=None

        else:


                # check if covariates exist in meta
                if not set(all_covariates).issubset(data.meta.columns):
                    raise ValueError("Invalid elements in CatCov or ConCov. Please check that all covariates names (continuous or categorials) are in the meta DataFrame. ")

                #cov_df = data.meta[all_covariates]

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

        return cov_df, expr #, control, case"""

def dysregnet_model(data):
    """
    Train models based on the expected (healthy) dataset and save the trained models.
    """
    # Directory to save models
    output_dir = "models/breast"
    os.makedirs(output_dir, exist_ok=True)

    # Data preparation: Use the whole dataset
    covariate_name = list(data.cov_df.columns) if hasattr(data, 'cov_df') and data.cov_df is not None else []

    # Dictionary to store model statistics
    model_stats = {}
    edges = {}

    for tup in tqdm(data.GRN.itertuples(), desc="Training models for edges"):
        edge = (tup[1], tup[2])  # Extract TF → target pair

        # Skip self-loops
        if edge[0] == edge[1]:
            continue

        # Ensure genes exist in the expression data
        if edge[0] not in data.expression_data.columns or edge[1] not in data.expression_data.columns:
            print(f"Skipping edge {edge}: Genes not found in expression data.")
            continue

        # Prepare x_train and y_train
        x_train = data.expression_data[[edge[0]] + covariate_name]  # Predictor variables
        y_train = data.expression_data[edge[1]]  # Target variable

        # Drop missing values and align indices
        aligned_data = pd.concat([x_train, y_train], axis=1).dropna()
        x_train = aligned_data[[edge[0]] + covariate_name]  # Predictor variables
        y_train = aligned_data[edge[1]]  # Target variable


        # Skip if no data is available
        if x_train.empty or y_train.empty:
            print(f"Skipping edge {edge}: No data available for this edge.")
            continue

        try:
            model = LinearModel(predictors=[edge[0]] + covariate_name, target=edge[1])
            results = model.train(x_train, y_train)

            # Save the trained model
            pickle_filename = os.path.join(output_dir, f"{edge[0]}_{edge[1]}.pkl")
            model.save(pickle_filename)

        except Exception as e:
            print(f"Error training model for edge {edge}: {e}")
            continue


    return results

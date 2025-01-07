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
        # process covariates and design matrix
        """
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
        case = data.meta[ data.meta[data.conCol]==1 ].index.values.tolist()"""

        #return cov_df, expr #, control, case
    
    
    
def dyregnet_model_old(data):
        
        # Fit a model for every edge
        # Detect outliers and calculate zscore and pvalues
        # correct for multiple testing 
        
        # prepare data
        
        #_______
        average_pickle = []
        average_joblib = []

        pickle_memory_usage = []
        joblib_memory_usage = []
        edge_memory_usage = {} # get the biggest edge 
        edgej_memory_usage = {} # get the biggest edge
        output_dir = "models/exercise"
        os.makedirs(output_dir, exist_ok=True)
        #_______
        
        if data.cov_df is not None:
            control=pd.merge(data.cov_df.loc[data.control],data.expr, left_index=True, right_index=True).drop_duplicates()
            case=pd.merge(data.cov_df.loc[data.case],data.expr, left_index=True, right_index=True).drop_duplicates()
            covariate_name= list(data.cov_df.columns)
        
        else:
            control=data.expr.loc[data.control]
            case=data.expr.loc[data.case]
            covariate_name=[]
            
        edges = {}
        edges['patient id']=list(case.index.values)
        model_stats = {}
        for tup in tqdm(data.GRN.itertuples()):
                    # pvalues for the same edge for all patients

                    edge = (tup[1],tup[2])
                    
                    # skip self loops
                    if edge[0] != edge[1]:

                        # prepare control for fitting model
                        x_train = control[  [edge[0]] + covariate_name ]
                        x_train = sm.add_constant(x_train, has_constant='add') # add bias
                        y_train = control[edge[1]].values

                        # fit the model
                        model = sm.OLS(y_train, x_train)
                        results = model.fit() # TODO interessant

                        # __________________________________________Save the model
                        "start our code"

                        # Measure time for pickle
                        tracemalloc.start()
                        start_pickle = time.time()
                        pickle_filename = os.path.join(output_dir, f"{edge[0]}_{edge[1]}.pkl")  # Name based on TF and target + TODO add tissue
                        with open(pickle_filename, "wb") as file:
                            pickle.dump(results, file)
                        with open(pickle_filename, "rb") as file:
                            results_pickle = pickle.load(file)
                        end_pickle = time.time()
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        average_pickle.append(end_pickle - start_pickle)
                        pickle_memory_usage.append(peak / 1024)  # Convert to KB
                        #print("Pickle takes ", end_pickle - start_pickle, "seconds")

                        # Measure time for joblib
                        tracemalloc.start()
                        start_joblib = time.time()
                        joblib.dump(model, "ols_model.joblib")
                        results_joblib = joblib.load("ols_model.joblib")
                        end_joblib = time.time()
                        average_joblib.append(end_joblib - start_joblib)
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        joblib_memory_usage.append(peak / 1024)  # Convert to KB
                        #print("Joblib takes ", end_joblib - start_joblib, "seconds")

                    
                        edge_memory_usage[edge] = pickle_memory_usage[-1]  # Memory used in the last pickle operation
                        edgej_memory_usage[edge] = joblib_memory_usage[-1]  # Memory used in the last pickle operation
                        # _____________________________________________
                        model_stats[edge] = [results.rsquared] + list(results.params.values) + list(results.pvalues.values)
                        
                        # get residuals of control
                        resid_control = y_train - results.predict(x_train) 

                        # test data (case or condition)
                        x_test = case[  [edge[0]]+ covariate_name    ]
                        x_test = sm.add_constant(x_test, has_constant='add') # add bias
                        y_test = case[edge[1]].values


                        # define residue for cases
                        resid_case =  y_test - results.predict(x_test)

                        
                        # condition of direction
                        cond = True
                        direction = np.sign(results.params.iloc[1])
                        
                        
                        # two sided p_value as default
                        # if direction_condition is false calculate, two sided p value
                        sides = 2

                        if data.direction_condition: 
                            cond = ( direction * resid_case ) < 0
                            
                            # if direction_condition is true only calculate one sided p value
                            sides = 1

                        
                        # calculate zscore
                        zscore= (resid_case - resid_control.mean()) / resid_control.std()
      


                        # Quality check of the fitness (optionally and must be provided by user)


                        if (data.R2_threshold is not None) and  ( data.R2_threshold > results.rsquared ):
                            # model fit is not that good on training
                            # shrink the zscores
                            edges[edge]= [0.0] * len(zscore)
                            continue

                        #normality test for residuals
                        if  data.normaltest:
                            pv = stats.normaltest(resid_control)[1]
                            if pv> data.normaltest_alpha:
                                # shrink the zscores to 0s
                                edges[edge]= [0.0] * len(zscore)
                                continue


                        # zscores to p values
                        pvalues=stats.norm.sf(abs(zscore)) * sides

                        # correct for multi. testing
                        pvalues=sm.stats.multipletests(pvalues,method='bonferroni',alpha=data.bonferroni_alpha)[1]

                        pvalues= pvalues < data.bonferroni_alpha



                        # direction condition and a p_value 
                        valid= cond * pvalues



                        # shrink the z scores that are not signifcant or not in the condition
                        zscore[~valid]=0.0


                        edges[edge] = np.round(zscore, 1)

        #______
        # save results for evaluation
        with open("memory_time_usage.txt", "w") as f:
            f.write(f"Average pickle: {np.average(average_pickle)}\n")
            f.write(f"Average joblib: {np.average(average_joblib)}\n")
            f.write(f"Average Memory Usage (Pickle): {np.mean(pickle_memory_usage):.2f} KB\n")
            f.write(f"Average Memory Usage (Joblib): {np.mean(joblib_memory_usage):.2f} KB\n")
            
            # Save top 5 edges by memory usage
            top_5_pickle_edges = sorted(edge_memory_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            f.write("Top 5 edges by pickle memory usage:\n")
            for edge, memory in top_5_pickle_edges:
                f.write(f"Edge: {edge}, Memory Usage: {memory:.2f} KB\n")

            top_5_joblib_edges = sorted(edgej_memory_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            f.write("Top 5 edges by joblib memory usage:\n")
            for edge, memory in top_5_joblib_edges:
                f.write(f"Edge: {edge}, Memory Usage: {memory:.2f} KB\n")
            
            # Save individual memory usage data
            f.write("Pickle Memory Usage (KB):\n")
            f.write(",".join(map(str, pickle_memory_usage)) + "\n")
            
            f.write("Joblib Memory Usage (KB):\n")
            f.write(",".join(map(str, joblib_memory_usage)) + "\n")
        #______
                    
        results = pd.DataFrame.from_dict(edges)
        results = results.set_index('patient id')
        
        model_stats_cols = ["R2"] + ["coef_" + coef for coef in ["intercept", "TF"] + covariate_name] + ["pval_" + coef for coef in ["intercept", "TF"] + covariate_name]
        model_stats = pd.DataFrame([model_stats[edge] for edge in results.columns], index=results.columns, columns=model_stats_cols)

        
        return results, model_stats



def dysregnet_model(data):
    """
    Train models based on the expected (healthy) dataset and save the trained models.
    """
    # Directory to save models
    output_dir = "models/lung"
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

        # load model 
        loaded_model = LinearModel.load(os.path.join(output_dir, f"{edge[0]}_{edge[1]}.pkl"))

        # Vorhersagen machen
        x_test = data.expression_data[[edge[0]] + covariate_name]
        y_pred = loaded_model.predict(x_test)


        # Add model statistics
        model_stats[edge] = [loaded_model.rsquared] + list(loaded_model.params) + list(loaded_model.pvalues)

    # Convert model statistics to a DataFrame
    model_stats_df = pd.DataFrame.from_dict(model_stats, orient='index')

    return model_stats_df

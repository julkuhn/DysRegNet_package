import os
import pickle
import numpy as np
import statsmodels.api as sm


class LinearModel:
    def __init__(self, predictors, target, params=None, rsquared=None, pvalues=None):
        """
        Initialize the linear model
        """
        self.predictors = predictors
        self.target = target
        self.params = params
        self.rsquared = rsquared
        self.pvalues = pvalues

    def train(self, x_train, y_train):
        """
        Train model with the input data
        """
        x_train = sm.add_constant(x_train, has_constant='add')  # add intercept
        model = sm.OLS(y_train, x_train)
        results = model.fit()

        # Speichere Ergebnisse
        self.params = results.params.values
        self.rsquared = results.rsquared
        self.pvalues = results.pvalues.values
        return results

    def predict(self, x_test):
        """
        predicts based on the given parameters
        """
        if self.params is None:
            raise ValueError("The model is not trained")
        
        #x_test = sm.add_constant(x_test, has_constant='add')  # add intercept
        return np.dot(x_test, self.params)

    def save(self, filename):
        """
        saves model
        """
        with open(filename, "wb") as file:
            pickle.dump({
                "predictors": self.predictors,
                "target": self.target,
                "params": self.params,
                "rsquared": self.rsquared,
                "pvalues": self.pvalues
            }, file)

    @classmethod
    def load(cls, filename):
        """
        Loads a LinearModel instance from a file.
        """
        try:
            with open(filename, "rb") as file:
                data = pickle.load(file)
                # Check if data is already an instance of LinearModel
                if isinstance(data, cls):
                    return data
                # Otherwise, assume it's a dictionary of attributes
                return cls(
                    predictors=data["predictors"],
                    target=data["target"],
                    params=np.array(data["params"]),
                    rsquared=data["rsquared"],
                    pvalues=np.array(data["pvalues"])
                )
        except Exception as e:
            raise Exception(f"An error occurred while loading the model: {e}") from e

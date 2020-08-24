
# model.py placeholder
# untested


import pandas as pd
import numpy as np
import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from pathlib import Path

class Model:
    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self.load()

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        self._model = None
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def save(self):
        if self._model is not None:
            final_result.save("/api/data/finalized_model.pickle")
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load(self):
        try:
            self._model = joblib.load(self._model_path)
            self._model = sm.load(self._model_path)
        except:
            self._model = None
        return self

model_path = Path(__file__).parent / "/api/data/finalized_model.pickle"
train_json_path = Path(__file__).parent / "/api/data/train.json"
test_json_path = Path(__file__).parent / "/api/data/test.json"
model = Model(model_path)


def get_model():
    return model

if __name__ == "__main__":

    jsondf = pd.read_json()
    print(jsondf.head())
    train_set = pd.to_json(orient='records')

    ## To do: pull in feature engineering and elements from main.py

    ## To do: Set up Depends dependency in main.py to call get_model()

    ## To do: Set up model load, save, train functionality via API

    #model.train(X, y)
    #model.save()


# main.py 

import pickle
import numpy as np
from pydantic import BaseModel, ValidationError, validator, parse_obj_as
from typing import List
from typing import Type
import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY


# Server
import uvicorn
from fastapi import FastAPI, Query, File, UploadFile, HTTPException, Depends

# Initialize model from pickle files

model_file = "api/data/finalized_model.pickle"
glm_model = sm.load(model_file)

variables = pickle.load(open('api/data/variables.pickle', 'rb'))

raw_train=pd.read_json('api/data/train.json')

train_val = raw_train.copy(deep=True)

#1. Fixing the money and percents#
train_val['x12'] = train_val['x12'].str.replace('$','')
train_val['x12'] = train_val['x12'].str.replace(',','')
train_val['x12'] = train_val['x12'].str.replace(')','')
train_val['x12'] = train_val['x12'].str.replace('(','-')
train_val['x12'] = train_val['x12'].astype(float)
train_val['x63'] = train_val['x63'].str.replace('%','')
train_val['x63'] = train_val['x63'].astype(float)


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
train_imputed = pd.DataFrame(imputer.fit_transform(train_val.drop(columns=['y', 'x5', 'x31',  'x81' ,'x82'])), columns=train_val.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
std_scaler = StandardScaler()
train_imputed_std = pd.DataFrame(std_scaler.fit_transform(train_imputed), columns=train_imputed.columns)

cat_columns = ['x5', 'x31','x81','x82']

train_val = train_val.drop(columns=['y'])

df_processed = pd.get_dummies(train_val, prefix_sep = '_', columns=cat_columns)

cat_dummies = [col for col in df_processed 
               if "_" in col 
               and col.split('_')[0] in cat_columns]

processed_columns = list(df_processed.columns[:])


class Item(BaseModel):
    x0: float = None
    x1: float = None
    x2: float = None
    x3: float = None
    x4: float = None
    x5: str = None
    x6: float = None
    x7: float = None
    x8: float = None
    x9: float = None
    x10: float = None
    x11: float = None
    x12: str = None
    x13: float = None
    x14: float = None
    x15: float = None
    x16: float = None
    x17: float = None
    x18: float = None
    x19: float = None
    x20: float = None
    x21: float = None
    x22: float = None
    x23: float = None
    x24: float = None
    x25: float = None
    x26: float = None
    x27: float = None
    x28: float = None
    x29: float = None
    x30: float = None
    x31: str = None
    x32: float = None
    x33: float = None
    x34: float = None
    x35: float = None
    x36: float = None
    x37: float = None
    x38: float = None
    x39: float = None
    x40: float = None
    x41: float = None
    x42: float = None
    x43: float = None
    x44: float = None
    x45: float = None
    x46: float = None
    x47: float = None
    x48: float = None
    x49: float = None
    x50: float = None
    x51: float = None
    x52: float = None
    x53: float = None
    x54: float = None
    x55: float = None
    x56: float = None
    x57: float = None
    x58: float = None
    x59: float = None
    x60: float = None
    x61: float = None
    x62: float = None
    x63: str = None
    x64: float = None
    x65: float = None
    x66: float = None
    x67: float = None
    x68: float = None
    x69: float = None
    x70: float = None
    x71: float = None
    x72: float = None
    x73: float = None
    x74: float = None
    x75: float = None
    x76: float = None
    x77: float = None
    x78: float = None
    x79: float = None
    x80: float = None
    x81: str = None
    x82: str = None
    x83: float = None
    x84: float = None
    x85: float = None
    x86: float = None
    x87: float = None
    x88: float = None
    x89: float = None
    x90: float = None
    x91: float = None
    x92: float = None
    x93: float = None
    x94: float = None
    x95: float = None
    x96: float = None
    x97: float = None
    x98: float = None
    x99: float = None

app = FastAPI()

@app.post("/predict")
def predict(items: List[Item]):

    # extract data from JSON into pandas data frame for preprocessing

    df = pd.DataFrame()
    for i in range(len(items)):
        data_dict = items[i].dict()
        df = df.append(data_dict, ignore_index=True)

    # feature engineering steps
    # to do: pass to to model.py for processing

    # fix nulls, replacing with 0 for now

    df['x12'] = df['x12'].str.replace('$','')
    df['x12'] = df['x12'].str.replace(',','')
    df['x12'] = df['x12'].str.replace(')','')
    df['x12'] = df['x12'].str.replace('(','-')
    df['x12'] = df['x12'].astype(float)
    df['x63'] = df['x63'].str.replace('%','')
    df['x63'] = df['x63'].astype(float)

    df_imputed = pd.DataFrame(imputer.transform(df.drop(columns=['x5', 'x31', 'x81' ,'x82'])), columns=train_val.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
    df_imputed_std = pd.DataFrame(std_scaler.transform(df_imputed), columns=train_imputed.columns)

    # get imputed columns for merge
    imputed_columns = list(df_imputed_std.columns[:])

    df_test_processed = pd.get_dummies(df, prefix_sep= '_', columns=cat_columns)

    # Remove additional columns
    for col in df_test_processed.columns:
        if ("_" in col) and (col.split('_')[0] in cat_columns) and col not in cat_dummies:
            print("Removing additional feature {}".format(col))
            df_test_processed.drop(col, axis=1, inplace=True)

    for col in cat_dummies:
        if col not in df_test_processed.columns:
            print("Adding missing feature {}".format(col))
            df_test_processed[col] = 0

    df_test_processed = df_test_processed[processed_columns]

    df_test_processed.drop(labels=imputed_columns, axis="columns", inplace=True)
    df_test_processed[imputed_columns] = df_imputed_std[imputed_columns]

    #print('df_test_processed.head')
    #print(df_test_processed.head())

    # to do: pass to model.py for proper processing

    # set up the X matrix in the correct order
    X = df_test_processed[variables]

    print(X.head())

    # should be redundant by this step, but clearning any missed NaN
    X = X.fillna(0)

    responses = []

    y_pred = glm_model.predict(X)

    for i in range(len(X)):

        if(y_pred[i] > 0.75):
            pred = 1
        else:
            pred = 0

        response_object = {
            "class probability": y_pred[i],
            "variables": X.loc[i],
            "predicted class": pred
        }

        responses.append(response_object)

    return responses





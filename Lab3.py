import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing datasets
data= pd.read_csv('country.csv')
print("Data before preprocessing:\n", data)
from sklearn.impute import SimpleImputer
#taking care of missing data
#'np.nan' signifies that we are targetting missing values by replacing it with 'mean' (Strategy)
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(data.iloc[:,1:3])
data.iloc[:, 1:3] = imputer.transform(data.iloc[:,1:3])
#OneHotEncoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],
remainder='passthrough')
# [0] singnifies the index of the column where the encoding is done
data= pd.DataFrame(ct.fit_transform(data))
#LabelEncoding
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
data.iloc[:,-1] = le.fit_transform(data.iloc[:,-1])
#Normalizing the data
from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data))
print("\nData after preprocessing:\n",data)
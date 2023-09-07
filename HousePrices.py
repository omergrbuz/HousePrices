import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score


# importing Data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_real_results = pd.read_csv("sample_submission.csv")

X_train = pd.concat([df_train.iloc[:,1:6],df_train.iloc[:,7:71], df_train.iloc[:,73:75], df_train.iloc[:,76:80]], axis = 1)
X_test = pd.concat([df_test.iloc[:,1:6],df_test.iloc[:,7:71], df_test.iloc[:,73:75], df_test.iloc[:,76:]], axis = 1)


# Filling NaN values and transforming categoricial data
X_test.loc[X_test['Neighborhood'] == 'IDOTRR', 'MSZoning'] = X_test.loc[X_test['Neighborhood'] == 'IDOTRR', 'MSZoning'].fillna('RM')
X_test.MSZoning.fillna("RL", inplace = True)

X_train.replace("Ex", 5, inplace = True)
X_train.replace("Gd", 4, inplace = True)
X_train.replace("TA", 3, inplace = True)
X_train.replace("Fa", 2, inplace = True)
X_train.replace("Po", 1, inplace = True)

X_test.replace("Ex", 5, inplace = True)
X_test.replace("Gd", 4, inplace = True)
X_test.replace("TA", 3, inplace = True)
X_test.replace("Fa", 2, inplace = True)
X_test.replace("Po", 1, inplace = True)

X_test.fillna(dict.fromkeys(["BsmtQual","BsmtCond","FireplaceQu",
                             "GarageQual", "GarageCond","PoolQC",
                             "MasVnrArea", "GarageCars", "GarageArea",
                             "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                             "TotalBsmtSF","BsmtFullBath", "BsmtHalfBath",
                             "GarageYrBlt", "BsmtExposure"], 0), inplace = True)


X_train.fillna(dict.fromkeys(["BsmtQual","BsmtCond","FireplaceQu",
                              "GarageQual", "GarageCond","PoolQC",
                              "MasVnrArea", "GarageYrBlt", "BsmtExposure"], 0), inplace = True)

X_train.BsmtExposure.replace('Av', 3, inplace = True)
X_train.BsmtExposure.replace('Mn', 2, inplace = True)
X_train.BsmtExposure.replace('No', 1, inplace = True)

X_test.BsmtExposure.replace('Av', 3, inplace = True)
X_test.BsmtExposure.replace('Mn', 2, inplace = True)
X_test.BsmtExposure.replace('No', 1, inplace = True)

X_test.Exterior1st.fillna("HdBoard", inplace = True)

X_train.MasVnrType.fillna("None", inplace = True)
X_test.MasVnrType.fillna("None", inplace = True)

X_train.Electrical.fillna("SBrkr", inplace = True)

X_train.LotFrontage.fillna(X_train.LotFrontage.median(), inplace = True)
X_test.LotFrontage.fillna(X_train.LotFrontage.median(), inplace = True)

X_test.SaleType.fillna("WD", inplace = True)

X_test.Exterior2nd.fillna("VinylSd", inplace = True)

X_test.Functional.fillna("Typ", inplace = True)
X_test.KitchenQual.fillna(3, inplace = True)


# DataFrame to Array
X_train = X_train.values
Y_train = df_train.iloc[:,80:81].values
X_test = X_test.values
Y_test = df_real_results.iloc[:,1:2].values


# OneHotEncoding
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(handle_unknown = "ignore"),
                                      [0,1,4,5,6,7,8,9,10,11,12,13,14,19,20,
                                       21,22,23,27,30,31,33,37,39,40,53,56,
                                       58,63,69,70,73,74])],remainder="passthrough")

X_train = ct.fit_transform(X_train).toarray()
X_test = ct.transform(X_test).toarray()


# Scaling data
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# GradiantBossting
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()
gb.fit(X_train, Y_train.ravel())


# Prediction
predict = gb.predict(X_test)


# Creating sub for kaggle
sub = df_real_results[['Id']]
sub['SalePrice'] = predict
sub.to_csv("sub.csv", index = None)











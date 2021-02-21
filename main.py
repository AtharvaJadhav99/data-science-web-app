import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction

This app predicts the **Boston House Price**
""")


boston=datasets.load_boston()

X=pd.DataFrame(boston.data,columns=boston.feature_names)
Y=pd.DataFrame(boston.target,columns=["MEDV"])

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM=st.sidebar.slider('CRIM- per capita crime rate by town',float(X.CRIM.min()),float(X.CRIM.max()),float(X.CRIM.mean()))
    ZN=st.sidebar.slider('ZN-proportion of residential land zoned for lots over 25,000 sq.ft',float(X.ZN.min()),float(X.ZN.max()),float(X.ZN.mean()))
    INDUS=st.sidebar.slider('INDUS- proportion of non-retail business acres per town',float(X.INDUS.min()),float(X.INDUS.max()),float(X.INDUS.mean()))
    CHAS=st.sidebar.slider('CHAS-Charles River dummy variable (1 if tract bounds river; 0 otherwise)',float(X.CHAS.min()),float(X.CHAS.max()),float(X.CHAS.mean()))
    NOX=st.sidebar.slider('NOX-nitric oxides concentration (parts per 10 million)',float(X.NOX.min()),float(X.NOX.max()),float(X.NOX.mean()))
    RM=st.sidebar.slider('RM-average number of rooms per dwelling',float(X.RM.min()),float(X.RM.max()),float(X.RM.mean()))
    AGE=st.sidebar.slider('AGE-proportion of owner-occupied units built prior to 1940',float(X.AGE.min()),float(X.AGE.max()),float(X.AGE.mean()))
    DIS=st.sidebar.slider('DIS-weighted distances to five Boston employment centres',float(X.DIS.min()),float(X.DIS.max()),float(X.DIS.mean()))
    RAD=st.sidebar.slider('RAD-index of accessibility to radial highways',float(X.RAD.min()),float(X.RAD.max()),float(X.RAD.mean()))
    TAX=st.sidebar.slider('TAX-full-value property-tax rate per $10,000',float(X.TAX.min()),float(X.TAX.max()),float(X.TAX.mean()))
    PTRATIO=st.sidebar.slider('PTRATIO- pupil-teacher ratio by town',float(X.PTRATIO.min()),float(X.PTRATIO.max()),float(X.PTRATIO.mean()))
    B=st.sidebar.slider('B-1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',float(X.B.min()),float(X.B.max()),float(X.B.mean()))
    LSTAT=st.sidebar.slider('LSTAT- % lower status of the population',float(X.LSTAT.min()),float(X.LSTAT.max()),float(X.LSTAT.mean()))
    
    data={'CRIM':CRIM,
           'ZN':ZN,
           'INDUS':INDUS,
           'CHAS':CHAS,
           'NOX':NOX,
           'RM':RM,
           'AGE':AGE,
           'DIS':DIS,
           'RAD':RAD,
           'TAX':TAX,
           'PTRATIO':PTRATIO,
           'B':B,
           'LSTAT':LSTAT}
    
    features=pd.DataFrame(data,index=[0])
    return features
    
df=user_input_features()


st.header('Specified Input Parameters')
st.write(df)
st.write('---')

model= RandomForestRegressor()
model.fit(X,Y)
prediction=model.predict(df)

st.header('Prediction of MEDV - Median value of owner-occupied homes in $*1000')
st.write(prediction)
st.write('---')

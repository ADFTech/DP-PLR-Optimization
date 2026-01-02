import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor 
import pickle as p

import random


random.seed(10)


# DF = pd.read_csv(r'DF_CHiller_OPT_Inputs.csv', header=0)
DF = pd.read_csv(r'chiller_train.csv', header=0)

DF['KW']=(DF['HLX_B1_Chiller_DPM_MSB_IN_1_kW']+DF['HLX_B1_Chiller_DPM_MSB_IN_2_kW'])
DF['KWH']=(DF['HLX_B1_Chiller_DPM_MSB_IN_1_kW']+DF['HLX_B1_Chiller_DPM_MSB_IN_2_kW'])/60.0
# DF['kW_RT']=DF['KW']/ DF['HLX_B1_Chiller_BTU_METER_Cooling_Capacity_Header']
DF['kW_RT'] = np.where(
    DF['HLX_B1_Chiller_BTU_METER_Cooling_Capacity_Header'] == 0,
    0,   # if RT = 0 → kW/RT = 0
    DF['KW'] / DF['HLX_B1_Chiller_BTU_METER_Cooling_Capacity_Header']
)

df=DF.drop(columns=['KW'])
df['Time'] = pd.to_datetime(df['Timestamp'])

# df['Timestamp'] =  pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')


### @@4. Define Training 
df['Year'] = df.Time.apply(lambda x: x.year)
df['Month'] = df.Time.apply(lambda x: x.month)
df['Day'] = df.Time.apply(lambda x: x.day)
df['Hour'] = df.Time.apply(lambda x: x.hour)
df['Minute'] = df.Time.apply(lambda x: x.minute)

df=df.drop(columns=['Time'])

features =   ['HLX_L18_AHU_Header_Low_Zone_Differential_Pressure','HLX_B1_Chiller_BTU_METER_Cooling_Capacity_Header','kW_RT','temperature','humidity','Year','Month','Day','Hour','Minute']



target = 'KWH'
# Extract X input features
X = df[features]

# Extract y output
y = df[target]

# # Features(X) values, drop the Y1 and Y2 columns

# X = df.drop(df[kWh], axis = 1)
# print(X.describe())

# # Create Target (Y2) values by dropping X columns and Y1 column

# y = df.drop(df[features], axis = 1)

# Set Seed to use for all models
SEED = 45
#Split into 72% train and 28% test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 0)

#Instatiate xgboost
hlx_gbt_kwh_model = XGBRegressor(n_estimators = 100, seed = 0)
# fit model
Predicted_Train=hlx_gbt_kwh_model.fit(X_train, y_train)
# predict on the test set
Predicted_Test = hlx_gbt_kwh_model.predict(X_test)
PRedicted_Test = pd.DataFrame(Predicted_Test, columns = ['PRedicted_kwh'])

# Y_test = pd.DataFrame(y_test, columns = ['kWh'])


PRedicted_Test.to_csv("PRedicted_HLX_KWH.csv")


# Y_test.to_csv("Y_testt_T.csv")

# Evaluate the test set RMSE and r2_score
# rmse_test = mean_squared_error(y_test, PRedicted_Test, squared=False)
r2_scores = r2_score(y_test, PRedicted_Test)

# Print the test set RMSE
print(f"Test set r2_score: {round(r2_scores, 2)}")
# print(f"Test set RMSE: {round(rmse_test, 2)}")
# ### @@5. Save Model
ModelPath = "Model/hlx_gbt_kwh_model.p"
p_model_file = open( ModelPath, 'wb')
p.dump(hlx_gbt_kwh_model, p_model_file)
p_model_file.close()


print("_____________DONEEEEE________________")

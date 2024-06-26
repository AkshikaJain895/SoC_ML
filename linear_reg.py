#import train.csv
#import test.csv
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('train.csv')
x_train = df[['crim' , 'zn' , 'indus' , 'chas' , 'nox' , 'rm' , 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]
y_train = df['medv']
df_2 = pd.read_csv('test.csv')
x_test = df_2[['crim' , 'zn' , 'indus' , 'chas' , 'nox' , 'rm' , 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]
#y_test = df_2['medv']


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = LinearRegression()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
#mae = mean_absolute_error(y_test, y_pred)
#mse = mean_squared_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)
print (model.coef_)

#print(f'MAE: {mae}')
#print(f'MSE: {mse}')
#print(f'R²: {r2}')
print (f'y is {y_pred}')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


root = 'E:/17-20/全.xlsx'
df = pd.read_excel(root, sheet_name='Sheet1', header=0, usecols='F:AY')
'''
nan_rows = df[df.isna().any(axis=1)] 检查NaN
df.dropna(inplace=True) 删除Nan
df.dtypes  检查数据类型
'''
df_y = df['NNI']
df_x = df[df.columns[4:]]

correlation_matrix = df_x.corrwith(pd.Series(df_y))
r2_values = correlation_matrix ** 2
top_features = r2_values.sort_values(ascending=False).head(5)
selected_features = top_features.index.tolist()
X_selected = df_x[selected_features] # 从 X 中选择对应的特征构建新的 X 数据集
print("选定的特征：", selected_features)


x_train,x_test,y_train,y_test=train_test_split(X_selected,df_y ,test_size=0.2,random_state=77)
lr = LinearRegression()
lr.fit(x_train,y_train)
y_p=lr.predict(x_test)
dff = pd.DataFrame({'Prediction': y_p, 'Y_test': y_test})

r2 = r2_score(y_test, y_p)
mse = mean_squared_error(y_test, y_p)
rmse = np.sqrt(mse)
print('系数:'+str(lr.coef_))
print('截距:'+str(lr.intercept_))
print('r2:'+str(r2))
print('rmse:'+str(rmse))
dff.to_csv('E:/17-20/2.1/抽穗后/AGB.csv', index=False)
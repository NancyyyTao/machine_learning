import optunity.metrics
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import r2_score
import optunity.metrics
import optunity
import shap
from sklearn.model_selection import train_test_split
import numpy as np

def ANN(optimizer = 'adam',neurons=32,activation='relu',loss='mse'):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(df_x.shape[1],), activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer = optimizer, loss=loss)
    return model


root = 'E:/17-20/抽穗后.xlsx'
df = pd.read_excel(root, sheet_name='Sheet1', header=0, usecols='F:AY')
df_y = df['AND']
df_x = df[df.columns[4:]]

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y ,test_size=0.2,random_state=77)
df_x=df_x.values
df_y=df_y.values

"""默认参数拟合  optimizer = 'adam',neurons=32,batch_size=32,epochs=50,activation='relu',patience=5,loss='mse'"""
estimator_o = KerasRegressor(model=ANN(),batch_size=32, epochs=50,verbose=0).fit(x_train, y_train,)
predictions_o = estimator_o.predict(x_test)
r2_o= r2_score(y_test,predictions_o)
RMSE_o= np.sqrt(optunity.metrics.mse(y_test, predictions_o))
print('R2_原始:'+str(r2_o),'RMSE_原始:'+str(RMSE_o))



"""----------模型拟合优化-----------"""
search = {
    'optimizer':[0,2],
    'activation':[0,2],
    'batch_size': [0, 2],
    'neurons': [10, 100],
    'epochs': [20, 50]
         }
@optunity.cross_validated(x=df_x, y=df_y, num_folds=5)
def performance(x_train, y_train, x_test, y_test,optimizer=None,activation=None,batch_size=None,neurons=None,epochs=None):
    # fit the model
    if optimizer<1:
        op='adam'
    else:
        op='rmsprop'
    if activation<1:
        ac='relu'
    else:
        ac='tanh'
    if batch_size<1:
        ba=16
    else:
        ba=32
    model = ANN(optimizer=op,activation=ac,neurons=int(neurons))
    estimator = KerasRegressor(model=model,batch_size=ba, verbose=0,epochs=int(epochs)).fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    scores_mse = optunity.metrics.mse(y_test, y_predict)

    return scores_mse

optimal_configuration, _, _ = optunity.minimize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

if optimal_configuration['optimizer']<1:
    optimal_configuration['optimizer']='adam'
else:
    optimal_configuration['optimizer']='rmsprop'
if optimal_configuration['activation']<1:
    optimal_configuration['activation']='relu'
else:
    optimal_configuration['activation']='tanh'
if optimal_configuration['batch_size']<1:
    optimal_configuration['batch_size']=16
else:
    optimal_configuration['batch_size']=32

optimal_configuration['neurons']=int(optimal_configuration['neurons'])
optimal_configuration['epochs']=int(optimal_configuration['epochs'])

print('optimal hyperparameter:'+ str(optimal_configuration))

tuned_model = ANN(optimizer=optimal_configuration['optimizer'],
                  activation=optimal_configuration['activation'],
                  neurons=optimal_configuration['neurons'])
estimator = KerasRegressor(model=tuned_model,
                           batch_size=optimal_configuration['batch_size'],
                           verbose=0,
                           epochs=optimal_configuration['epochs']).fit(x_train, y_train)
predictions= estimator.predict(x_test)
r2= r2_score(y_test,predictions)
RMSE= np.sqrt(optunity.metrics.mse(y_test, predictions))
print('R2:'+str(r2),'RMSE:'+str(RMSE))

# explainer = shap.DeepExplainer(tuned_model,x_train[:1012])
# shap_values = explainer.shap_values(x_train)

""" ------------散点图数据导出-----"""
dff = pd.DataFrame({'Y_test': y_test,'Prediction': predictions})
dff.to_csv('E:/17-20/2.1/抽穗后/ANN/AND.csv', index=False)
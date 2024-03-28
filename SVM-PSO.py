import optunity.metrics
import pandas as pd
from sklearn.metrics import r2_score
import optunity.metrics
import optunity
import shap
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

root = 'E:/17-20/抽穗前.xlsx'
df = pd.read_excel(root, sheet_name='Sheet1', header=0, usecols='F:AY')
df_y = df['AND']
df_x = df[df.columns[4:]]

"""----------模型拟合优化-----------"""
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y ,test_size=0.2,random_state=77)
df_x=df_x.values
df_y=df_y.values

search = {
    'C': (0,50),
    'kernel':[0,3],
    'epsilon': (0, 1)
         }
@optunity.cross_validated(x=df_x, y=df_y, num_folds=5)
def performance(x_train, y_train, x_test, y_test,C=None,kernel=None,epsilon=None):

    if kernel<1:
        ke='poly'
    elif kernel<2:
        ke='rbf'
    else:
        ke='sigmoid'
    model = SVR(C=float(C),
                kernel=ke,
                gamma='scale',
                epsilon=float(epsilon)
                                  ).fit(x_train,y_train)

    y_predict = model.predict(x_test)
    scores_mse = optunity.metrics.mse(y_test, y_predict)

    return scores_mse

optimal_configuration, _, _ = optunity.minimize(performance,solver_name= 'particle swarm',num_evals=20, **search) # num_evals 参数搜索迭代次数

if optimal_configuration['kernel'] < 1:
    optimal_configuration['kernel'] = 'poly'
elif optimal_configuration['kernel'] < 2:
    optimal_configuration['kernel'] = 'rbf'
else:
    optimal_configuration['kernel'] = 'sigmoid'

print('optimal hyperparameter:'+ str(optimal_configuration))

tuned_model = SVR(**optimal_configuration).fit(x_train,y_train)
predictions= tuned_model.predict(x_test)
r2= r2_score(y_test,predictions)
RMSE= np.sqrt(optunity.metrics.mse(y_test, predictions))
print('R2:'+str(r2),'RMSE:'+str(RMSE))

f = lambda x: tuned_model.predict(x)
med = x_train.median().values.reshape((1,x_train.shape[1]))
explainer = shap.KernelExplainer(f, med)
shap_values = explainer.shap_values(x_train.iloc[0:1012,:], nsamples=1000)

"""-----------开始画图--------------"""
plt.rcParams.update(
    {
        'text.usetex': False,
        'mathtext.fontset': 'stix',
        "font.family": 'serif',
        "font.serif": ['Times New Roman'],
        "axes.unicode_minus": False,
        "font.size": 18,
    })
""" ----重要性排序-----"""
shap.summary_plot(shap_values,x_train,show=False,max_display=10,plot_type='bar')
fig, ax = plt.gcf(), plt.gca() # 获取当前图像对象和坐标轴对象
fig.set_size_inches(10, 8) #修改画布大小
ax.set_position([0.25, 0.2, 0.6, 0.7])  # 左下角位于图形的横坐标 和纵坐标的位置。
                                        # 子图的宽度为图形宽度的比例，高度为图形高度的比例
ax.spines['top'].set_visible(True) # 上框线显示
ax.spines['right'].set_visible(True) # 右框线显示
ax.spines['left'].set_visible(True) # 左框线显示
ax.set_xlabel('') # 重置横坐标轴名称
ax.set_xlim(right=5) # 设置坐标轴长度
ax.tick_params(labelsize=18,direction='in') # 修改所有坐标轴字体大小和刻度方向
ax.bar_label(ax.containers[0],fontsize=18,padding=5,fmt='%.2f')

"""！！！！！修改文件名！！！！！"""
plt.savefig('E:/17-20/2.1/AGB.png',dpi=600)
plt.show()

""" ----瀑布图-----"""
shap.summary_plot(shap_values,x_train,max_display=10,show=False)
fig, ax = plt.gcf(), plt.gca() # 获取当前图像对象和坐标轴对象
fig.set_size_inches(10, 8) #修改画布大小
ax.set_position([0.2, 0.2, 0.6, 0.7])  # 左下角位于图形的横坐标 和纵坐标的位置。
                                        # 子图的宽度为图形宽度的比例，高度为图形高度的比例
ax.spines['top'].set_visible(True) # 上框线显示
ax.spines['right'].set_visible(True) # 右框线显示
ax.spines['left'].set_visible(True) # 左框线显示
ax.set_xlabel('') # 重置横坐标轴名称
ax.tick_params(labelsize=18,direction='in') # 修改所有坐标轴字体大小和刻度方向
cb =fig.axes[1] # 获取颜色条
cb.tick_params(labelsize=18)
cb.set_ylabel('') # 重置颜色条轴名称

"""！！！！！修改文件名！！！！！"""
plt.savefig('E:/17-20/2.1/AGB_瀑布.png',dpi=600)
plt.show()

""" ------------散点图数据导出-----"""
dff = pd.DataFrame({'Prediction': predictions, 'Y_test': y_test})

"""！！！！！修改文件名！！！！！"""
dff.to_csv('E:/17-20/2.1/NNI.csv', index=False)

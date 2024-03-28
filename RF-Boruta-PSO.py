import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
import optunity.metrics
import optunity
import shap
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import matplotlib.pyplot as plt
from  BorutaShap import  BorutaShap

root = 'E:/17-20/抽穗前.xlsx'
df = pd.read_excel(root, sheet_name='Sheet1', header=0, usecols='F:AY')
'''
nan_rows = df[df.isna().any(axis=1)] 检查NaN
df.dropna(inplace=True) 删除Nan
'''
df_y = df['AND']
df_x = df[df.columns[4:]]

feat_selector = BorutaShap(
    model=RandomForestRegressor(random_state=0,max_depth=5),
    importance_measure='shap',
    classification=False)
feat_selector.fit(X=df_x,y=df_y,n_trials=100, random_state=0)
subset = feat_selector.Subset()
df_x = subset

"""
RFE 选择
min_features_to_select = 1  # Minimum number of features to consider
clf = RandomForestRegressor(random_state=0)
cv = KFold(5,shuffle=True)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="neg_mean_squared_error",
    min_features_to_select=min_features_to_select,
    )
rfecv.fit(df_x, df_y )
print(f"Optimal number of features: {rfecv.n_features_}")

selected_feature_names = df_x.columns[rfecv.support_]
df_x = pd.DataFrame(df_x[selected_feature_names], columns=selected_feature_names)
"""

"""----------模型拟合优化-----------"""
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y ,test_size=0.2,random_state=77)
df_x=df_x.values
df_y=df_y.values

search = {
    'n_estimators': [10, 1000],
    'max_features': [1, 13],
    'max_depth': [3, 50],
    "min_samples_split": [2, 11],
    "min_samples_leaf": [1, 11],
}

@optunity.cross_validated(x=df_x, y=df_y, num_folds=5) # num_iter 迭代次数 num_folds 交叉验证次数
def performance(x_train, y_train, x_test, y_test,n_estimators=None, max_features=None,max_depth=None,min_samples_split=None,min_samples_leaf=None):

    model = RandomForestRegressor(n_estimators=int(n_estimators),
                                       max_features=int(max_features),
                                       max_depth=int(max_depth),
                                       min_samples_split=int(min_samples_split),
                                       min_samples_leaf=int(min_samples_leaf),
                                      ).fit(x_train,y_train)


    """
    多输出头
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=int(n_estimators),
                                  max_features=int(max_features),
                                  max_depth=int(max_depth),
                                  min_samples_split=int(min_samples_split),
                                  min_samples_leaf=int(min_samples_leaf),
                                  )).fit(x_train,y_train)
                                  
    """


    y_predict = model.predict(x_test)
    scores_mse = optunity.metrics.mse(y_test, y_predict)

    return scores_mse

optimal_configuration, _, _ = optunity.minimize(performance,solver_name= 'particle swarm',num_evals=20, **search) # num_evals 参数搜索迭代次数

def integerize_kwargs(**kwargs):  # 字典值取整
    for key, value in kwargs.items():
        kwargs[key] = int(value)
    return kwargs

optimal_configuration = integerize_kwargs(**optimal_configuration)
print('optimal hyperparameter:'+ str(optimal_configuration))

tuned_model = RandomForestRegressor(**optimal_configuration).fit(x_train,y_train)
predictions= tuned_model.predict(x_test)
r2= r2_score(y_test,predictions)
RMSE= np.sqrt(optunity.metrics.mse(y_test, predictions))
print('R2:'+str(r2),'RMSE:'+str(RMSE))

explainer = shap.Explainer(tuned_model)
shap_values = explainer(x_train)

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
ax.set_xlim(right=10) # 设置坐标轴长度
ax.tick_params(labelsize=18,direction='in') # 修改所有坐标轴字体大小和刻度方向
ax.bar_label(ax.containers[0],fontsize=18,padding=5,fmt='%.2f')

"""！！！！！修改文件名！！！！！"""
plt.savefig('E:/17-20/2.1/AND.png',dpi=600)
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
plt.savefig('E:/17-20/2.1/PNA_瀑布.png',dpi=600)
plt.show()

""" ------------散点图数据导出-----"""
dff = pd.DataFrame({'Prediction': predictions, 'Y_test': y_test})

"""！！！！！修改文件名！！！！！"""
dff.to_csv('E:/17-20/2.1/PNA.csv', index=False)
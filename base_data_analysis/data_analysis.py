import os
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
import urllib.parse
import seaborn as sns
import statsmodels.api as sm
from scipy import linalg
import math
from mpl_toolkits.mplot3d import Axes3D  #3Dplot
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FactorAnalysis as FA
from seaborn_analyzer import regplot
from sklearn.decomposition import PCA

sns.set()


dirname = os.getcwd()
filepath=dirname + '/city_data.csv'
#間接的な影響
setu='Nonwhite'
#全体
col_name=['Rainfall','Education','Popden','Nonwhite','NOX','SO2']

#データ読み込み
df=pd.read_csv(filepath,index_col=0)
df1=df.copy()

#標準化
df1=((df-df.mean())/df.std())
"""
#課題1
#p値を考慮
regplot.linear_plot(x='SO2', y='Mortality',data=df1)
plt.show()
#SO2とMortalityの散布図
sns.scatterplot(x='SO2',y='Mortality',data=df1)
plt.show()

#全体のヒートマップ
sns.heatmap(df.corr(),annot=True)
plt.show()
"""

#課題2
"""

#変数代入
X = np.array(df1[['SO2', setu]])
#X=np.array(df1[col_name])
#Y = df1[['Mortality']]
Y=np.array(df1['Mortality'])
# グラフ可視化用
X1=df1[['SO2']]
X2=df1[[setu]]





#グラフで可視化

fig=plt.figure()
ax=Axes3D(fig)

ax.scatter3D(X1, X2, Y)
ax.set_xlabel("SO2")
ax.set_ylabel(setu)
ax.set_zlabel("Mortality")

plt.show()



"""

"""
#因子分析

# 因子数を指定
n_components=3

# 因子分析の実行
fa = FA(n_components, max_iter=5000) # モデルを定義
fitted = fa.fit_transform(df1) # fitとtransformを一括処理

# 変数Factor_loading_matrixに格納
Factor_loading_matrix = fa.components_.T

print(fitted)
pd.DataFrame(fitted).to_csv("insitokuten.csv", sep=",")

# データフレームに変換
df_in=pd.DataFrame(Factor_loading_matrix, 
             columns=["1", "2", "3"], 
             #columns=["第1因子", "第2因子"], 
             #columns=["1", "2","3","4","5","6","7"], 
             index=[df.columns])

#print(df_in)
#df_in.to_csv("result3.csv", sep=",")
#sns.scatterplot(x='1', y='2', data=df_in)
#plt.show()

"""
"""
#固有値を推測
pca=PCA()
pca.fit(df1)
# 寄与率の取得
evr = pca.explained_variance_ratio_

# 行名･列名を付与してデータフレームに変換
df_syu=pd.DataFrame(evr, 
             index=["PC{}".format(x + 1) for x in range(len(df.columns))], 
             columns=["寄与率"])
print(df_syu)
# 起点0と寄与率の累積値をプロット
plt.plot([0] + list(np.cumsum(evr)), "-o")

plt.xlabel("主成分の数")
plt.ylabel("累積寄与率")

plt.grid()
plt.show()
"""









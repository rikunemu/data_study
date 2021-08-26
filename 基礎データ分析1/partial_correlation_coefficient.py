import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import linalg
import math

# 偏相関係数
def p_corr(y,data):
    corr = data.corr()
    #1.相関行列の逆行列を求める
    inv_corr = pd.DataFrame(np.linalg.inv(corr),columns=corr.columns,index=corr.columns)
    p_corr_list = []
    #2.逆行列の各要素を2つの対角要素の積の平方根で割り，符号を逆転する
    for x in corr.columns:
        p_corr = -(inv_corr[x][y] / np.sqrt(inv_corr[x][x]*inv_corr[y][y]))
        p_corr_list.append([x ,p_corr])
    return p_corr_list

#アイスクリーム偏相関:aの影響を取り除く
def p_corr_san(df,a,b,y):
    rab=df[a].corr(df[b])
    ray=df[a].corr(df[y])
    rby=df[b].corr(df[y])
    molecule=rby-(ray*rab)
    denominator=(math.sqrt(1-ray**2))*(math.sqrt(1-rab**2))
    return molecule/denominator



filepath="/Users/shinohararikunin/Desktop/共同勉強/データ分析"
#dfにcsvデータを格納
df=pd.read_csv(filepath+"/aripoll.csv",  encoding="utf-8", sep=",",engine="python")
#df=df.iloc[:,1:]
#print(df)
pcor=p_corr_san(df,'SO2','Mortality','NOX')
#pcor=cor2pcor(df)
print(pcor)


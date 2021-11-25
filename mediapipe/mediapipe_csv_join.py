import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import csv
import os

fp="/Users/shinohararikunin/Desktop/data_analysis/mediapipe/csv_save"
files=os.listdir(fp)
files.sort()
data_list=[]
count=0
for file in files:
  print(file)
  filepath=fp+'/'+file
  if 'valid' in file:
    df=pd.read_csv(filepath)
    df=df[df["0_x"]!="NAN"]
    df["correct"]=count
    data_list.append(df)
    count+=1
df1 = pd.concat(data_list, axis=0, sort=True)
df1.to_csv("/Users/shinohararikunin/Desktop/data_analysis/mediapipe/csv_save/csvvalidtotal.csv",index=False)
print(count)
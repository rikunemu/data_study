import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import csv
import os
from tqdm import tqdm
len_sign=20
resultcsv=[]
dir_input=r"D:\output_file\csvfile\test"
dir_output=r"D:\output_file\join_csv\csvtestresult.csv"
def csvjoin_static(fp):
  files=os.listdir(fp)
  files.sort()
  data_list=[]
  count=0
  for file in files:
    print(file)
    filepath=fp+'/'+file
    df=pd.read_csv(filepath)
    df=df.iloc[:,478*3+21*3+1:]
    #df=df.iloc[:,"right_hand_19_y":]
    #df=df[df["0_x"]!="NAN"]
    #df["correct"]=count
    data_list.append(df)
    count+=1
  df1 = pd.concat(data_list, axis=0, sort=True)
  return df1
  #df1.to_csv("/Users/shinohararikunin/Desktop/data_analysis/mediapipe/csv_save/csvvalidtotal.csv",index=False)


for i in range(len_sign):
  file_path="sign_"+str(i+1)
  basefile_inputpath=os.path.join(dir_input,file_path)
  #basefile_outputpath=os.path.join(dir_output_image,class_i,file_path)
  #dir_output_csvpath=os.path.join(dir_output,file_path)
  #ディレクトリの中身分ループ
  #for index in tqdm(range(len(os.listdir(basefile_inputpath)))):
  df1=csvjoin_static(basefile_inputpath)
  if i==0:
    df2=pd.concat([df1],axis=0,sort=True)
  else:
    df2=pd.concat([df2,df1],axis=0,sort=True)
print(df2)
#df2.to_csv(dir_output,index=False)
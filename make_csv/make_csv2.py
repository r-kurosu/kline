from gurobipy.gurobipy import Column
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

df_input = [0]*13
df_output =[0]*13
for booking in range(1,2):
    for i in range(1,13):
        df_input[i] = pd.read_csv("data/old_car_data/carinfo_"+str(i)+"_"+str(booking)+".csv")
        df_output[i] = pd.read_csv("data/old_car_data/template.csv")

        count = 0
        for j in range(len(df_input[i])):
            df = df_input[i]
            # 計算
            hold = df.at[j,'HOLD']
            if hold == 1 or hold == 2:
                seg = 1
            else:
                seg = 2

            width = int(df.at[j,'WIDTH']/10)
            height = int(df.at[j,'HEIGHT']/10)
            
            # 出力
            df_output[i] = df_output[i].append({'ID':j,'HOLD':hold,'SEG':seg,'HEIGHT':height,'WIDTH':width,'AMOUNT':df.at[j,'AMOUNT'],'LP':df.at[j,'LP'],'DP':df.at[j,'DP']}, ignore_index = True)

        df_output[i].to_csv('data/detailed_data/car'+str(i)+'_'+str(booking)+'.csv', index = False)

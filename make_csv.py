from gurobipy.gurobipy import Column
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

df_input = [0]*13
df_output =[0]*13
df_output_seg = [0]*13
for booking in range(1,2):
    for i in range(1,13):
        df_input[i] = pd.read_csv("data/new_car/carinfo_"+str(i)+"_"+str(booking)+".csv")
        df_output_seg[i] = pd.read_csv("data/new_car/template.csv")
        
        count = 0
        for SEG in range(1,3):
            for LP in range(1,10):
                for DP in range(1,15):
                    df = df_input[i]
                    df_temp = df[((df['HOLD']==2*SEG) | (df['HOLD']==2*SEG-1)) & (df['LP']==LP) & (df['DP']==DP)]
                    amount = df_temp.loc[:,'AMOUNT'].sum()
                    if amount == 0:
                        continue

                    width = df_temp['WIDTH']*df_temp['AMOUNT']
                    width_ave = sum(width)/amount/10
                    height = df_temp['HEIGHT']*df_temp['AMOUNT']
                    height_ave = sum(height)/amount/10
                    area = df_temp['WIDTH']*df_temp['HEIGHT']*df_temp['AMOUNT']
                    area_sum = sum(area)/100
                    
                    # 出力
                    df_output_seg[i] = df_output_seg[i].append({'ID':count,'HOLD':0,'SEG':SEG,'HEIGHT':int(height_ave),'WIDTH':int(width_ave),'AMOUNT':amount,'LP':LP,'DP':DP}, ignore_index = True)
                    count += 1

        df_output_seg[i].to_csv('data/car_group/seggroup'+str(i)+'_'+str(booking)+'.csv', index = False)
        print(df_output_seg[i])

from statistics import mean
import pandas as pd

ship_area = 350*2000


print('---------------------------------------')
for BOOKING in range(1,4):
    print('===========================================================================')
    print('here is booking {}'.format(BOOKING))
    for DK in range(8,13):
        df_obs = pd.read_csv('data/obs_data/obs_'+str(DK)+'dk.csv')
        sum_area = 0 
        for i in range(len(df_obs)):
            area = df_obs.at[i,'WIDTH']*df_obs.at[i,'HEIGHT']
            sum_area += area
        print(str(DK)+'dkの障害物の合計は{}'.format(sum_area))
        print('よって'+str(DK)+'dkの配置可能面積は{}'.format(ship_area - sum_area))
        
        df_car = pd.read_csv('data/new_data/car'+str(DK)+'_'+str(BOOKING)+'.csv')
        sum_car_area = 0
        for i in range(len(df_car)):
            car_area = df_car.at[i,'WIDTH']*df_car.at[i,'HEIGHT'] + 2*df_car.at[i,'HEIGHT'] + 6*df_car.at[i,'WIDTH']
            sum_car_area += car_area
        print(str(DK)+'dkの車の面積の合計は{}'.format(sum_car_area))
        print('***')
        df_car_ave = pd.read_csv('data/car_group/seggroup'+str(DK)+'_'+str(BOOKING)+'.csv')
        s_i = [0]*len(df_car_ave)
        for j in range(len(df_car_ave)):
            s_i[j] = df_car_ave.at[j,'WIDTH']*df_car_ave.at[j,'HEIGHT'] + 2*df_car_ave.at[j,'HEIGHT'] + 6*df_car_ave.at[j,'WIDTH']

        max_car_area = max(s_i)
        min_car_area = min(s_i)
        ave_car_area = mean(s_i)
        
        ub = (ship_area - sum_area)/max_car_area
        ub_2 = (ship_area - sum_area)/ave_car_area
        lb = (sum_area - sum_car_area)/min_car_area
        print(str(DK)+'dkに詰め込める車の上界は{}台'.format(int(ub)))
        print(str(DK)+'dkに詰め込める車の上界(2)は{}台'.format(int(ub_2)))
        print(str(DK)+'dkにおける余りの数の下界は{}台'.format(int(lb)))
        print('配置可能面積 -- 車の面積: {}'.format(sum_area - sum_car_area))
        print('-------------------------------------')


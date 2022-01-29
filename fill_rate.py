import pandas as pd

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
        print('よって'+str(DK)+'dkの配置可能面積は{}'.format(2000*360 - sum_area))
        
        df_car = pd.read_csv('data/new_data/car'+str(DK)+'_'+str(BOOKING)+'.csv')
        sum_car_area = 0
        for i in range(len(df_car)):
            car_area = df_car.at[i,'WIDTH']*df_car.at[i,'HEIGHT']
            sum_car_area += car_area
        print(str(DK)+'dkの車の面積の合計は{}'.format(sum_car_area))
        print('***')
        df_car_ave = pd.read_csv('data/car_group/seggroup'+str(DK)+'_'+str(BOOKING)+'.csv')
        max_width = df_car_ave.loc[:,'WIDTH'].max()
        max_height = df_car_ave.loc[:,'HEIGHT'].max()
        max_car_area = max_width*max_height

        ave_width = df_car_ave.loc[:,'WIDTH'].mean()        
        ave_height = df_car_ave.loc[:,'HEIGHT'].mean()
        ave_car_area = ave_width*ave_height
        
        ub = (2000*350 - sum_area)/ave_car_area
        lb = (sum_area - sum_car_area)/ave_car_area
        print(str(DK)+'dkに詰め込める車の上界は{}台'.format(int(ub)))
        print(str(DK)+'dkにおける余りの数の下界は{}台'.format(int(lb)))
        print('配置可能面積 -- 車の面積: {}'.format(sum_area - sum_car_area))
        print('-------------------------------------')


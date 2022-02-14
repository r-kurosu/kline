from unittest import case
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np


df_car = [0]*13
df_ship_ = [0]*13
df_ramp_ = [0]*13
df_obs_ = [0]*13
df_aisle_ = [0]*13
def datainput(DK):
    for i in range(1,13):
        df_ship_[i] = pd.read_csv("data/ship_data/ship_"+str(i)+"dk.csv")
        df_ramp_[i] = pd.read_csv("data/ramp_data/ramp_"+str(i)+"dk.csv")
        df_obs_[i] = pd.read_csv("data/obs_data/obs_"+str(i)+"dk.csv")
        df_aisle_[i] = pd.read_csv("data/aisle_data/aisle_"+str(i)+"dk.csv")
    return df_ship_[DK], df_ramp_[DK], df_obs_[DK], df_aisle_[DK]


# ランプの矢印を書く
def make_arrow(DK):
    match DK:
        case 12:
            axes[DK].text(300, 750, '↓')
        case 11:
            axes[DK].text(300, 1100, '↓')
            axes[DK].text(300, 900, '↓', color = 'w')
        case 10:
            axes[DK].text(70, 1250, '↑')
            axes[DK].text(300, 1250, '↓', color = 'w')
        case 9:
            axes[DK].text(60, 900, '↑')
            axes[DK].text(60, 1100, '↑', color = 'w')
        case 8:
            axes[DK].text(60, 800, '↑', color = 'w')
            axes[DK].text(10, 800, '↓')
        case _:
            pass
        

# 船体，障害物，ランプをプロット
def plot_funk(df_ship, df_obs, df_ramp):
    ship = patches.Rectangle(xy=(0, 0), width=df_ship.at[0,'WIDTH'], height=df_ship.at[0,'HEIGHT'], ec='k', fill =False, linewidth = 0.2)
    axes[DK].add_patch(ship)
    for i in range(len(df_obs)):
        obstacle = patches.Rectangle(xy=(df_obs.at[i,'X'], df_obs.at[i,'Y']), width = df_obs.at[i,'WIDTH'], height = df_obs.at[i,'HEIGHT'], fc = 'k')
        axes[DK].add_patch(obstacle)
    for i in range(len(df_ramp)):
        ramp = patches.Rectangle(xy=(df_ramp.at[i,'X'], df_ramp.at[i,'Y']), width = df_ramp.at[i,'WIDTH'], height = df_ramp.at[i,'HEIGHT'], fc = 'silver', ec = 'k', linewidth = 0.2)
        axes[DK].add_patch(ramp)
        

# output --
fig = plt.figure()
axes = [0]*13
for DK in range(1,13):
    df_ship, df_ramp, df_obs, df_aisle = datainput(DK)
    axes[DK] = fig.add_subplot(1,12,DK)
    subplot_title = (str(DK)+'dk')
    axes[DK].set_title(subplot_title)
    plot_funk(df_ship, df_obs, df_ramp)
    plt.axis("scaled")
    axes[DK].axis('off')

plt.axis("scaled")
plt.tight_layout()
plt.savefig('output_data/test.png', dpi = 100)
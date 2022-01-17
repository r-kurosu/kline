import random
from tokenize import group
from typing import BinaryIO
from anytree import node
from anytree.node import nodemixin
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from anytree import Node, RenderTree, PostOrderIter
import anytree
from matplotlib.backends.backend_pdf import PdfPages
import gurobipy as gp
from pandas.core.indexing import _iLocIndexer
from multiprocessing import Process, cpu_count, process
import math


t1 = time.time()

df_list = [0]*5
for i in range(5):
    df_list[i] = pd.read_csv('data/car_group/seggroup'+str(12-i)+'_1.csv')


seg_dp12 = [0,0,0]
seg_dp11 = [0,0,0]
seg_dp10 = [0,0,0]
seg_dp9 = [0,0,0]
seg_dp8 = [0,0,0]
seg_lp12 = [0,0,0]
seg_lp11 = [0,0,0]
seg_lp10 = [0,0,0]
seg_lp9 = [0,0,0]
seg_lp8 = [0,0,0]
def make_segtree():
    global seg_dp12, seg_dp11, seg_dp10, seg_dp9, seg_dp8
    global seg_lp12, seg_lp11, seg_lp10, seg_lp9, seg_lp8
    
    root1 = Node(0, parent=None)
    seg_dp8[2] = Node(df_list[4][df_list[4]['SEG'] == 2].loc[:,'DP'].min(), parent=root1)
    seg_dp8[1] = Node(df_list[4][df_list[4]['SEG'] == 1].loc[:,'DP'].min(), parent=seg_dp8[2])

    seg_dp9[1] = Node(df_list[3][df_list[3]['SEG'] == 1].loc[:,'DP'].min(), parent=seg_dp8[2])
    seg_dp9[2] = Node(df_list[3][df_list[3]['SEG'] == 2].loc[:,'DP'].min(), parent=seg_dp9[1])

    seg_dp10[2] = Node(df_list[2][df_list[2]['SEG'] == 2].loc[:,'DP'].min(), parent=seg_dp9[2])
    seg_dp10[1] = Node(df_list[2][df_list[2]['SEG'] == 1].loc[:,'DP'].min(), parent=seg_dp10[2])

    seg_dp11[2] = Node(df_list[1][df_list[1]['SEG'] == 2].loc[:,'DP'].min(), parent=seg_dp10[2])
    seg_dp11[1] = Node(df_list[1][df_list[1]['SEG'] == 1].loc[:,'DP'].min(), parent=seg_dp11[2])

    seg_dp12[1] = Node(df_list[0][df_list[0]['SEG'] == 1].loc[:,'DP'].min(), parent=seg_dp11[1])
    seg_dp12[2] = Node(df_list[0][df_list[0]['SEG'] == 2].loc[:,'DP'].min(), parent=seg_dp12[1])

    # 各ノードには部分木のDP最小値を保存する
    for node in PostOrderIter(root1):
        if node.is_leaf:
            continue
        for child in node.children:
            node.name = min(node.name, child.name)

    root2 = Node(0, parent=None)
    seg_lp8[2] = Node(df_list[4][df_list[4]['SEG'] == 2].loc[:,'LP'].max(), parent=root2)
    seg_lp8[1] = Node(df_list[4][df_list[4]['SEG'] == 1].loc[:,'LP'].max(), parent=seg_lp8[2])

    seg_lp9[1] = Node(df_list[3][df_list[3]['SEG'] == 1].loc[:,'LP'].max(), parent=seg_lp8[2])
    seg_lp9[2] = Node(df_list[3][df_list[3]['SEG'] == 2].loc[:,'LP'].max(), parent=seg_lp9[1])

    seg_lp10[2] = Node(df_list[2][df_list[2]['SEG'] == 2].loc[:,'LP'].max(), parent=seg_lp9[2])
    seg_lp10[1] = Node(df_list[2][df_list[2]['SEG'] == 1].loc[:,'LP'].max(), parent=seg_lp10[2])

    seg_lp11[2] = Node(df_list[1][df_list[1]['SEG'] == 2].loc[:,'LP'].max(), parent=seg_lp10[2])
    seg_lp11[1] = Node(df_list[1][df_list[1]['SEG'] == 1].loc[:,'LP'].max(), parent=seg_lp11[2])

    seg_lp12[1] = Node(df_list[0][df_list[0]['SEG'] == 1].loc[:,'LP'].max(), parent=seg_lp11[1])
    seg_lp12[2] = Node(df_list[0][df_list[0]['SEG'] == 2].loc[:,'LP'].max(), parent=seg_lp12[1])    

    # 各ノードには部分木のDP最小値を保存する
    for node in PostOrderIter(root2):
        if node.is_leaf:
            continue
        for child in node.children:
            node.name = max(node.name, child.name)

make_segtree()


# input data
df_car = [0]*13
df_ship_ = [0]*13
df_ramp_ = [0]*13
df_obs_ = [0]*13
df_aisle_ = [0]*13
def datainput(DK):
    for i in range(8,13):
        # df_car[i] = pd.read_csv("data/car_"+str(i)+"dk_1.csv")
        # df_car[i] = pd.read_csv("data/car_group/cargroup"+str(i)+"_1.csv")
        df_car[i] = pd.read_csv("data/car_group/seggroup"+str(i)+"_1.csv")
        df_ship_[i] = pd.read_csv("data/ship_data/ship_"+str(i)+"dk.csv")
        df_ramp_[i] = pd.read_csv("data/ramp_data/ramp_"+str(i)+"dk.csv")
        df_obs_[i] = pd.read_csv("data/obs_data/obs_"+str(i)+"dk.csv")
        df_aisle_[i] = pd.read_csv("data/aisle_data/aisle_"+str(i)+"dk.csv")
    return df_car[DK], df_ship_[DK], df_ramp_[DK], df_obs_[DK], df_aisle_[DK]

# ここで使用するデッキを選択
df, df_ship, df_ramp, df_obs, df_aisle = datainput(12)

car_w = 0 
car_h = 0
car_amount = 0
car_color = 0
car_hold = 0
n = len(df)
car_x = []
car_y = []

# 船体情報
ship_w = df_ship.iloc[0,0]
ship_h = df_ship.iloc[0,1]
deck_enter = df_ship.iloc[0,2]
center_line = df_ship.iloc[0,3]
stock_sheet = [0]*ship_w
reverse_sheet = [ship_h] * ship_w

# ランプ情報
enter_line = df_ramp.iloc[0,1]

# # 障害物を定義
class Obstacle():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

obs = []
for i in range(len(df_obs)):
    class_obs = Obstacle(df_obs.iloc[i, 0], df_obs.iloc[i, 1], df_obs.iloc[i,2], df_obs.iloc[i,3])
    obs.append(class_obs)

# 通路を定義
aisle = []
class Aisle():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
for i in range(len(df_aisle)):
    class_aisle = Obstacle(df_aisle.at[i,'X'], df_aisle.at[i,'Y'], df_aisle.at[i,'WIDTH'], df_aisle.at[i,'HEIGHT'])
    aisle.append(class_aisle)


class NFP():
    def __init__(self, x, y, w, h, car_w, car_h):
        self.x1 = x - car_w
        self.y1 = y - car_h
        self.w1 = w + car_w
        self.h1 = h + car_h
        self.x2 = x - car_w
        self.y2 = y - car_w - car_h 
        self.w2 = w + car_h
        self.h2 = h + car_w
        self.x3 = x - car_h
        self.y3 = y - car_w - car_h 
        self.y4 = y - 2*car_h
        self.x5 = x
        self.y5 = y
        self.x6 = x - car_w - car_h
        self.y6 = y + car_h
        self.x7 = x - car_w
        self.y7 = y - car_w
        self.y8 = y - 2*car_w - car_h
        self.h8 = h + 2*car_w
        self.x9 = x - car_w - 4 
        self.w9 = w + car_w + 8
        self.h9 = 1
        self.y9 = y + w
        self.y10 = y - 1


def find_lowest_gap(stock_sheet, w):
    x = 0
    y = 0
    gap = 0
    x = stock_sheet.index(min(stock_sheet))
    y = min(stock_sheet)
    for ii in range(x, w):
        if stock_sheet[ii] == min(stock_sheet):
            gap += 1
        else:
            break
    return x, y, gap

def find_highest_gap(reverse_sheet, w):  #左上から詰めていく
    x = 0
    y = ship_h
    gap = 0
    x = reverse_sheet.index(max(reverse_sheet))
    y = max(reverse_sheet)
    for ii in range(x, w):
        if reverse_sheet[ii] == max(reverse_sheet):
            gap += 1
        else:
            break
    return x, y, gap


def calc_nfp(x, y, car_w, car_h):   #左下が起点
    if x < 0 or x > ship_w - car_w or y < 0 or y > ship_h - car_h:
        return False
    p = True
    for i in range(len(nfp)):
        if (nfp[i].x9 < x < nfp[i].x9 + nfp[i].w9 and nfp[i].y1 < y < nfp[i].y1 + nfp[i].h1) or (nfp[i].x2 < x < nfp[i].x2 + nfp[i].w2 and nfp[i].y2 < y < nfp[i].y2 + nfp[i].h2) or (nfp[i].x1 < x < nfp[i].w1 and nfp[i].y9 < y < nfp[i].y9 + nfp[i].h9):
            p = False
            break
    if p == True:
        # print('左向きに駐車します!')
        return True
    p = True
    for i in range(len(nfp)):
        if (nfp[i].x9 < x < nfp[i].x9 + nfp[i].w9 and nfp[i].y1 < y < nfp[i].y1 + nfp[i].h1) or (nfp[i].x3 < x < nfp[i].x3 + nfp[i].w2 and nfp[i].y3 < y < nfp[i].y3 + nfp[i].h2 or (nfp[i].x1 < x < nfp[i].w1 and nfp[i].y9 < y < nfp[i].y9 + nfp[i].h9)):
            p = False
            break
    if p ==True:
        # print('右駐車!')
        return True
    return False

def calc_nfp_reverse(x, y, car_w, car_h):   #左上が起点
    # if x < 0 or x > ship_w - car_w or y < car_h or y > ship_h:
    #     return False
    p = True
    for j in range(len(nfp)):
        if (nfp[j].x9 < x < nfp[j].x9 + nfp[j].w9 and nfp[j].y5 < y < nfp[j].y5 + nfp[j].h1) or (nfp[j].x2 < x < nfp[j].x2 + nfp[j].w2 and nfp[j].y6 < y < nfp[j].y6 + nfp[j].h2) or (nfp[j].x1 < x < nfp[j].x1 + nfp[j].w1 and nfp[j].y10 < y < nfp[j].y10 + nfp[j].h9):
            p = False
            break
    if p == True:
        return True
    p = True
    for j in range(len(nfp)):
        if (nfp[j].x9 < x < nfp[j].x9 + nfp[j].w9 and nfp[j].y5 < y < nfp[j].y5 + nfp[j].h1) or (nfp[j].x3 < x < nfp[j].x3 + nfp[j].w2 and nfp[j].y6 < y < nfp[j].y6 + nfp[j].h2) or (nfp[j].x1 < x < nfp[j].x1 + nfp[j].w1 and nfp[j].y10 < y < nfp[j].y10 + nfp[j].h9):
            p = False
            break
    if p ==True:
        return True
    return False

def calc_pnfp(x, y, car_w, car_h):  #左下が起点の縦列駐車
    if x < 0 or x > ship_w - car_h or y < 0 or y > ship_h - car_w:
        return False
    p = True
    for i in range(len(nfp_p)):
        if (nfp_p[i].x3 < x < nfp_p[i].x3 + nfp_p[i].w2 and nfp_p[i].y7 < y < nfp_p[i].y7 + nfp_p[i].h2) or (nfp_p[i].x7 < x < nfp_p[i].x7 + nfp_p[i].w1 and nfp_p[i].y1 < y < nfp_p[i].y1 + nfp_p[i].h1):
            p = False
            break
    if p == True:
        return True
    p = True
    for i in range(len(nfp_p)):
        if (nfp_p[i].x3 < x < nfp_p[i].x3 + nfp_p[i].w2 and nfp_p[i].y7 < y < nfp_p[i].y7 + nfp_p[i].h2) or (nfp_p[i].x6 < x < nfp_p[i].x6 + nfp_p[i].w1 and nfp_p[i].y1 < y < nfp_p[i].y1 + nfp_p[i].h1):
            p = False
            break
    if p ==True:
        return True
    return False

def calc_pnfp_reverse(x, y, car_w, car_h):   #左上が起点に縦列駐車
    if x < 0 or x > ship_w - car_h or y < car_w or y > ship_h:
        return False
    p = True
    for j in range(len(nfp_p)):
        if (nfp_p[j].x3 < x < nfp_p[j].x3 + nfp_p[j].w2 and nfp_p[j].y5 < y < nfp_p[j].y5 + nfp_p[j].h2) or (nfp_p[j].x6 < x < nfp_p[j].x6 + nfp_p[j].w1 and nfp_p[j].y5 < y < nfp_p[j].y5 + nfp_p[j].h1):
            p = False
            break
    if p == True:
        return True
    p = True
    for j in range(len(nfp)):
        if (nfp_p[j].x3 < x < nfp_p[j].x3 + nfp_p[j].w2 and nfp_p[j].y5 < y < nfp_p[j].y5 + nfp_p[j].h2) or (nfp_p[j].x5 < x < nfp_p[j].x5 + nfp_p[j].w1 and nfp_p[j].y5 < y < nfp_p[j].y5 + nfp_p[j].h1):
            p = False
            break
    if p ==True:
        return True
    return False

def check_overlap(new_x, new_y, car_w, car_h):
    if new_x + car_w > ship_w:
        for l in range(ship_w - new_x):
            if reverse_sheet[new_x + l] <= new_y + car_h: 
                print('これ以上詰めません!')
                print('{}台詰めませんでした．．．'.format(car_amount - count - tuduki))
                return False
    else:
        for l in range(car_w):
            if reverse_sheet[new_x + l] <= new_y + car_h: 
                print('これ以上詰めません!')
                print('{}台詰めませんでした．．．'.format(car_amount - count - tuduki))
                return False
    return True

def check_overlap_r(new_x, car_w, car_h):
    if new_x + car_w > ship_w:
        for l in range(ship_w - new_x):
            if reverse_sheet[new_x + l] - car_h <= stock_sheet[new_x + l]:
                print('もう詰めません')
                return False
    else:    
        for l in range(car_w):
            if reverse_sheet[new_x + l] - car_h <= stock_sheet[new_x + l]:
                print('もう詰めません')
                return False
    return True

color_dict = {
    1:'lightgreen',
    2:'red',
    3:'aqua',
    4:'magenta',
    5:'blue',
    6:'olive',
    7:'brown',
    8:'orange',
    9:'mediumseagreen',
    10:'deepskyblue',
    11:'navy',
    12:'purple',
    13:'hotpink',
    14:'gray',
}

def color_check(key):  #出力時のカラーを決める
    color = color_dict[key]
    return color

# ランプの矢印を書く
def make_arrow(DK):
    match 12-DK:
        case 12:
            axes[4-DK].text(300, 750, '↓')
        case 11:
            axes[4-DK].text(300, 1100, '↓')
            axes[4-DK].text(300, 900, '↓', color = 'w')
        case 10:
            axes[4-DK].text(70, 1250, '↑')
            axes[4-DK].text(300, 1250, '↓', color = 'w')
        case 9:
            axes[4-DK].text(60, 900, '↑')
            axes[4-DK].text(60, 1100, '↑', color = 'w')
        case 8:
            axes[4-DK].text(60, 800, '↑', color = 'w')
            axes[4-DK].text(10, 800, '↓')
        case _:
            pass


# 通路制約を追加(hold)
# 通路制約を追加(seg)
def aisle_check_seg(DK, seg):
    global mindp, maxlp
    mindp = 100
    maxlp = 0
    if DK == 12:
        for child in seg_dp12[seg].children:
            if child.name > mindp:
                mindp = child.name
        for child in seg_lp12[seg].children:
            if child.name < maxlp:
                maxlp = child.name
    elif DK == 11:
        for child in seg_dp11[seg].children:
            if child.name > mindp:
                mindp = child.name
        for child in seg_lp11[seg].children:
            if child.name < maxlp:
                maxlp = child.name
    elif DK == 10:
        for child in seg_dp10[seg].children:
            if child.name > mindp:
                mindp = child.name
        for child in seg_lp10[seg].children:
            if child.name < maxlp:
                maxlp = child.name
    elif DK == 9:
        for child in seg_dp9[seg].children:
            if child.name > mindp:
                mindp = child.name
        for child in seg_lp9[seg].children:
            if child.name < maxlp:
                maxlp = child.name
    elif DK == 8:
        for child in seg_dp8[seg].children:
            if child.name > mindp:
                mindp = child.name
        for child in seg_lp8[seg].children:
            if child.name < maxlp:
                maxlp = child.name
    return mindp, maxlp


def add_hold_size(DK):
    global sepalation_line
    df_hold = pd.read_csv('data/hold_data/hold.csv')
    if DK == 12:
        seg1_size = 206+246
        seg2_size = 206+180
    elif DK == 11:
        seg1_size = 216+236
        seg2_size = 192+180
    elif DK == 10:
        seg1_size = 209+246
        seg2_size = 180+188
    elif DK == 9:
        seg1_size = 212+252
        seg2_size = 170+187
    elif DK == 8:
        seg1_size = 200+215
        seg2_size = 206+184
        
    sepalation_line = int(2000 * seg1_size / (seg1_size+seg2_size))
    return sepalation_line

# output --
fig = plt.figure()
axes = []
for i in range(5):
    axes.append(fig.add_subplot(1,5,i+1))
    subplot_title = (str(i+8)+'dk')
    axes[i].set_title(subplot_title)
    ship = patches.Rectangle(xy=(0, 0), width=ship_w, height=ship_h, ec='k', fill =False, linewidth = 0.2)
    axes[i].add_patch(ship)
    plt.axis("scaled")
    axes[i].axis('off')

# 変数定義（なくてもいいかも）
new_x = 0
new_y = 0
nfp = []
nfp_p = []
count = 0
tuduki = 0
area = []
remain_car = [0]*5

# first packing --
def group_packing(DK,b,output):
    global x_sol, y_sol, w_sol, h_sol
    global model, model2
    df, df_ship, df_ramp, df_obs, df_aisle= datainput(12-DK)
    obs = []
    sepalation_line = add_hold_size(12-DK)
    print(sepalation_line)
    
    # ランプ情報
    enter_line = df_ramp.iloc[0,1]
    if output == 1:
        for i in range(len(df_ramp)):
            ramp = patches.Rectangle(xy=(df_ramp.at[i,'X'], df_ramp.at[i,'Y']), width = df_ramp.at[i,'WIDTH'], height = df_ramp.at[i,'HEIGHT'], fc = 'silver', ec = 'k', linewidth = 0.2)
            axes[4-DK].add_patch(ramp)

    n = len(df)
    df_lp = df.sort_values(by=['SEG','LP','DP'], ascending = [True, True, False])
    df_dp = df.sort_values(by=['SEG','DP','LP'], ascending = [True, False, True])
    x = [0]*n
    y = [0]*n
    w = [0]*n
    h = [0]*n
    siguma = [0]*n
    siguma_ = [0]*n
    for i in range(n):
        siguma[i] = df_lp.iloc[i,0]
        siguma_[i] = df_dp.iloc[i,0]
    if output == 1:
        print('siguma+ : {}'.format(siguma))
        print('siguma- : {}'.format(siguma_))
    width = ship_w
    height = sepalation_line
    depth = sepalation_line
    area = [df.iloc[i,1]*df.iloc[i,2]*df.iloc[i,3] for i in range(n)]
    
    df_car = pd.read_csv('data/new_data/car'+str(12-DK)+'_1.csv')
    max_height = [0]*n
    max_width = [0]*n
    for i in range(n):
        seg_i = df.at[i,'SEG']
        lp_i = df.at[i,'LP']
        dp_i = df.at[i,'DP']
        max_height[i] = df_car[(df_car['SEG'] == seg_i) & (df_car['LP'] == lp_i) & (df_car['DP'] == dp_i)].loc[:,'HEIGHT'].max()
        max_width[i] = df_car[(df_car['SEG'] == seg_i) & (df_car['LP'] == lp_i) & (df_car['DP'] == dp_i)].loc[:,'WIDTH'].max()
    
    x_sol = [0]*n
    y_sol = [0]*n
    w_sol = [0]*n
    h_sol = [0]*n
    # modeling
    for segment in range(1,3):
        if segment == 1:
            model = gp.Model(name = "Gurobisample1")
            # 変数定義
            x = [0]*n
            y = [0]*n
            w = [0]*n
            h = [0]*n
            a = model.addVar(lb=1, vtype=gp.GRB.CONTINUOUS)
            for i in range(n):
                x[i] = model.addVar(lb = 0, vtype=gp.GRB.CONTINUOUS)
                y[i] = model.addVar(lb = 0, vtype=gp.GRB.CONTINUOUS)
                w[i] = model.addVar(lb = max_width[i] + 4, vtype=gp.GRB.CONTINUOUS)
                h[i] = model.addVar(lb = max_height[i] + 1, vtype=gp.GRB.CONTINUOUS)
            model.update
            # 目的関数
            model.setObjective(gp.quicksum(w[j]*h[j] for j in range(n) if df.iloc[j,4] == 1) - a, sense=gp.GRB.MAXIMIZE)
            # model.setObjective(gp.quicksum(w[j]*h[j] for j in range(n) if df.iloc[j,4] == 1)*10000 - a*10000 - gp.quicksum(w[j]*h[j]-area[j] for j in range(n) if df.iloc[j,4] == 1), sense=gp.GRB.MAXIMIZE)
            # 制約
            c1 = [0]*n
            c2 = [0]*n
            c3 = [0]*n
            c4 = [0]*n
            c5 = [0]*n
            c = []
            for i in range(n):
                if df.at[i,'SEG'] != segment:
                    continue
                c1[i] = model.addConstr(x[i] <= width - w[i])
                c2[i] = model.addConstr(y[i] <= height - h[i])
                c3[i] = model.addConstr(w[i]*h[i] >= area[i])
                c5[i] = model.addConstr(w[i]*h[i] <= a*area[i])
                for j in range(n):
                    if df.at[j,'SEG'] != segment:
                        continue
                    if i == j:
                        continue
                    elif siguma.index(i) < siguma.index(j) and siguma_.index(i) < siguma_.index(j):
                        c.append(model.addConstr(y[i] + h[i] <= y[j]))
                    elif siguma.index(i) < siguma.index(j) and siguma_.index(i) > siguma_.index(j):
                        c.append(model.addConstr(x[i] + w[i] <= x[j]))
            model.update

            # 実行
            model.params.NonConvex = 2
            model.Params.OutputFlag = 0
            model.Params.MIPFocus = 3
            model.optimize()
            print(a.X)
            # output
            if model.Status == gp.GRB.OPTIMAL:
                # print(a.X)
                for i in range(n):
                    if df.at[i,'SEG'] != segment:
                        continue
                    x_sol[i] = round(x[i].X)
                    y_sol[i] = round(y[i].X)
                    w_sol[i] = round(w[i].X)
                    h_sol[i] = round(h[i].X)
                    if output == 1:
                        cars = patches.Rectangle(xy=(x_sol[i], y_sol[i]), width = w_sol[i], height = h_sol[i], ec = 'k', fill = False)
                        axes[4-DK].add_patch(cars)
                        axes[4-DK].text(x_sol[i]+0.5*w_sol[i], y_sol[i]+0.5*h_sol[i], i, horizontalalignment = 'center', verticalalignment = 'center' , fontsize = 10)
            else:
                remain_car[DK] = 10000
                print('最適解が見つかりませんでした')
                print('model1.status = {}'.format(model.Status))
                
        elif segment == 2:
            model2 = gp.Model(name = "Gurobisample2")
            # 変数定義
            x = [0]*n
            y = [0]*n
            w = [0]*n
            h = [0]*n
            a = model2.addVar(lb=1, vtype=gp.GRB.CONTINUOUS)
            for i in range(n):
                x[i] = model2.addVar(lb = 0, vtype=gp.GRB.CONTINUOUS)
                y[i] = model2.addVar(lb = 0, vtype=gp.GRB.CONTINUOUS)
                w[i] = model2.addVar(lb = max_width[i] + 4, vtype=gp.GRB.CONTINUOUS)
                h[i] = model2.addVar(lb = max_height[i] + 1, vtype=gp.GRB.CONTINUOUS)
            model2.update

            # 目的関数
            model2.setObjective(gp.quicksum(w[j]*h[j] for j in range(n) if df.iloc[j,4] == 2) - a, sense=gp.GRB.MAXIMIZE)

            # 制約›
            c1 = [0]*n
            c2 = [0]*n
            c3 = [0]*n
            c4 = [0]*n
            c5 = [0]*n
            c = []
            for i in range(n):
                if df.at[i,'SEG'] != 2:
                    continue
                c1[i] = model2.addConstr(x[i] <= width - w[i])
                c2[i] = model2.addConstr(y[i] >= depth)
                c3[i] = model2.addConstr(y[i] <= ship_h - h[i])
                c4[i] = model2.addConstr(w[i]*h[i] >= area[i])
                # -original constraint- #
                c5[i] = model2.addConstr(w[i]*h[i] <= a*area[i])
                
                temp_list = [j for j in range(len(df)) if df.at[j,'SEG'] == 2]
                if siguma.index(i) <= min(temp_list) or siguma_.index(i) <= min(temp_list):
                    c4[i] = model2.addConstr(w[i]*h[i] >= b*area[i])
                    c5[i] = model2.addConstr(w[i]*h[i] <= b*a*area[i])
                # -original constraint- #
                for j in range(n):
                    if df.at[j,'SEG'] != segment:
                        continue
                    if i == j:
                        continue
                    elif siguma.index(i) < siguma.index(j) and siguma_.index(i) < siguma_.index(j):
                        c.append(model2.addConstr(y[j] + h[j] <= y[i]))
                    elif siguma.index(i) < siguma.index(j) and siguma_.index(i) > siguma_.index(j):
                        c.append(model2.addConstr(x[i] + w[i] <= x[j]))
                    # ランプと重ならない制約を追加したい．．．

            # # 実行
            model2.params.NonConvex = 2
            model2.Params.OutputFlag = 0
            model2.Params.MIPFocus = 3
            model2.optimize()
            # output
            if model2.Status == gp.GRB.OPTIMAL:
                for i in range(n):
                    if df.at[i,'SEG'] != segment:
                        continue
                    x_sol[i] = round(x[i].X)
                    y_sol[i] = round(y[i].X)
                    w_sol[i] = round(w[i].X)
                    h_sol[i] = round(h[i].X)
                    if output == 1:
                        cars = patches.Rectangle(xy=(x_sol[i], y_sol[i]), width = w_sol[i], height = h_sol[i], ec = 'k', fill = False)
                        axes[4-DK].add_patch(cars)
                        axes[4-DK].text(x_sol[i]+0.5*w_sol[i], y_sol[i]+0.5*h_sol[i], i, horizontalalignment = 'center', verticalalignment = 'center' , fontsize = 10)
            else:
                print('最適解が見つかりませんでした．')
                print('model2.status = {}'.format(model2.Status))
                remain_car[DK] = 10000


# second packing -- 
def detailed_packing(DK,output):
    if output == 1:
        print("{}dkに車を詰め込みます".format(12-DK))
    global new_x, new_y
    global df_obs, obs
    global nfp, nfp_p
    global enter_line
    global stock_sheet, reverse_sheet
    global x_sol, y_sol, w_sol, h_sol
    global remain_car
    
    df, df_ship, df_ramp, df_obs, df_aisle = datainput(12 - DK)
    n = len(df)
    remain_car[DK] = 0
    center_line_list = [0,1,2,3,4,5,6,7,1000,1050,1300,1000,600]
    
    if output == 1:
        for i in range(len(df_ramp)):
            ramp = patches.Rectangle(xy=(df_ramp.at[i,'X'], df_ramp.at[i,'Y']), width = df_ramp.at[i,'WIDTH'], height = df_ramp.at[i,'HEIGHT'], fc = 'silver', ec = 'k', linewidth = 0.2)
            axes[4-DK].add_patch(ramp)

    # 障害物情報
    obs = []
    for i in range(len(df_obs)):
        class_obs = Obstacle(df_obs.iloc[i, 0], df_obs.iloc[i, 1], df_obs.iloc[i,2], df_obs.iloc[i,3])
        obs.append(class_obs)
    if output == 1:
        for i in range(len(obs)):
            obstacle = patches.Rectangle(xy=(obs[i].x, obs[i].y), width = obs[i].w, height = obs[i].h, fc = 'k')
            axes[4-DK].add_patch(obstacle)

    df_lp = df.sort_values(by=['SEG','LP','DP'], ascending = [True, True, False])
    lp_order = [df_lp.iloc[i,0] for i in range(len(df_lp))]
    # print(lp_order)
    # main packing
    df_car = pd.read_csv('data/new_data/car'+str(12-DK)+'_1.csv')
    seg1 = df_car['SEG'] == 1
    seg2 = df_car['SEG'] == 2
    df_car = df_car.sort_values(by=['SEG','LP','DP','HEIGHT'], ascending=[True,True,False,False])
    
    # for i in range(n):
    for i in lp_order:
        stock_sheet = [0]*w_sol[i]
        reverse_sheet = [h_sol[i]]*w_sol[i]
        lp = df.iloc[i,6]
        dp = df.iloc[i,7]
        mindp, maxlp = aisle_check_seg(12-DK, df.iloc[i,4])

        df_car = df_car.sort_values(by=['SEG','LP','DP','HEIGHT'], ascending=[True,True,False,False])
        for car in range(seg1.sum()):
            car_w = df_car.iloc[car,1]
            car_h = df_car.iloc[car,2]
            car_amount = df_car.iloc[car,3]
            car_seg = df_car.iloc[car,4]
            car_lp = df_car.iloc[car,6]
            car_dp = df_car.iloc[car,7]
            count = 0
            if car_seg != df.at[i,'SEG'] or car_lp != lp or car_dp != dp:
                continue
            nfp = []
            nfp_p = []
            for j in range(len(df_obs)):
                nfp_obs = NFP(df_obs.at[j,'X'], df_obs.at[j,'Y'], df_obs.at[j,'WIDTH'], df_obs.at[j,'HEIGHT'], car_w, car_h)
                nfp.append(nfp_obs)
                nfpp_obs = NFP(df_obs.at[j,'X'], df_obs.at[j,'Y'], df_obs.at[j,'WIDTH'], df_obs.at[j,'HEIGHT'], car_w, car_h)
                nfp_p.append(nfpp_obs)
            
            if  car_seg == 1:
                while car_amount != count:
                    new_x, new_y, gap = find_lowest_gap(stock_sheet, w_sol[i])
                    if new_y + car_h > h_sol[i]:
                        remain_car[DK] += car_amount - count
                        break
                    # DP,LPによる通路制約を確認
                    if car_dp > mindp or car_lp < maxlp:
                        flag = 1
                        for l in range(len(df_aisle)):
                            if (df_aisle.iloc[l,0] - car_w < new_x < df_aisle.iloc[l,0] + df_aisle.iloc[l,2]) and (df_aisle.iloc[l,1] - car_h < new_y < df_aisle.iloc[l,1] + df_aisle.iloc[l,3]):
                                stock_sheet[new_x] += 1
                                flag = 0
                                break
                        if flag == 0:
                            continue
                    if gap >= car_w and (calc_nfp(new_x+x_sol[i], new_y+y_sol[i], car_w, car_h) == True):
                        for j in range(car_w):
                            stock_sheet[new_x + j] += car_h
                        if output == 1:
                            cars = patches.Rectangle(xy=(new_x+x_sol[i], new_y+y_sol[i]), width = car_w, height = car_h, fc = color_check(df.iloc[i,7]), ec = 'k', linewidth = 0.2)
                            axes[4-DK].add_patch(cars)
                            axes[4-DK].text(new_x+x_sol[i]+0.5, new_y+y_sol[i]+2, count, fontsize = 1)
                            axes[4-DK].text(new_x+x_sol[i]+car_w/2, new_y+y_sol[i] + car_h/2, '↑', fontsize = 1)
                            df_obs = df_obs.append({'X':new_x+x_sol[i], 'Y':new_y+y_sol[i], 'WIDTH':car_w, 'HEIGHT':car_h}, ignore_index = True)
                        count += 1

                    elif gap >= car_w and calc_nfp(new_x+x_sol[i], new_y+y_sol[i], car_w, car_h) == False:
                        stock_sheet[new_x] += 1
                    else:
                        if new_x == 0:
                            tonari = stock_sheet[gap]
                        elif new_x + gap == w_sol[i]:
                            tonari = stock_sheet[new_x - 1]
                        else:
                            tonari = min(stock_sheet[new_x - 1], stock_sheet[new_x + gap ])
                        for j in range(gap):
                            stock_sheet[new_x + j] = tonari

                df.iloc[i,3] -= count

        df_car = df_car.sort_values(by=['SEG','LP','DP','HEIGHT'], ascending=[False,True,False,False])
        for car in range(seg2.sum()):
            car_w = df_car.iloc[car,1]
            car_h = df_car.iloc[car,2]
            car_amount = df_car.iloc[car,3]
            car_seg = df_car.iloc[car,4]
            car_lp = df_car.iloc[car,6]
            car_dp = df_car.iloc[car,7]
            count = 0
            if car_seg != df.at[i,'SEG'] or car_lp != lp or car_dp != dp:
                continue
            nfp = []
            nfp_p = []
            for j in range(len(df_obs)):
                nfp_obs = NFP(df_obs.at[j,'X'], df_obs.at[j,'Y'], df_obs.at[j,'WIDTH'], df_obs.at[j,'HEIGHT'], car_w, car_h)
                nfp.append(nfp_obs)
                nfpp_obs = NFP(df_obs.at[j,'X'], df_obs.at[j,'Y'], df_obs.at[j,'WIDTH'], df_obs.at[j,'HEIGHT'], car_w, car_h)
                nfp_p.append(nfpp_obs)
                
            if car_seg == 2:
                while car_amount != count:
                    new_x, new_y, gap = find_highest_gap(reverse_sheet, w_sol[i])
                    if new_y - car_h < 0:
                        remain_car[DK] += car_amount - count
                        break
                    # DP,LPによる通路制約を確認
                    if car_dp > mindp or car_lp < maxlp:
                        flag = 1
                        for l in range(len(df_aisle)):
                            if (df_aisle.iloc[l,0] - car_w < new_x < df_aisle.iloc[l,0] + df_aisle.iloc[l,2]) and (df_aisle.iloc[l,1] < new_y < df_aisle.iloc[l,1] + df_aisle.iloc[l,3] + car_h):
                                reverse_sheet[new_x] -= 1
                                flag = 0
                                break
                        if flag == 0:
                            continue
                    if gap >= car_w and (calc_nfp_reverse(new_x+x_sol[i], new_y+y_sol[i], car_w, car_h) == True):
                        for j in range(car_w):
                            reverse_sheet[new_x + j] -= car_h
                        if output == 1:    
                            cars = patches.Rectangle(xy=(new_x+x_sol[i], new_y+y_sol[i] - car_h), width = car_w, height = car_h, fc = color_check(df.iloc[i,7]), ec = 'k', linewidth = 0.2)
                            axes[4-DK].add_patch(cars)
                            axes[4-DK].text(new_x+x_sol[i] + 0.5, new_y+y_sol[i] - car_h + 2, count, fontsize = 1)
                            axes[4-DK].text(new_x+x_sol[i] + car_w/2, new_y+y_sol[i] - car_h/2, '↓', fontsize = 1)
                            df_obs = df_obs.append({'X':new_x+x_sol[i], 'Y':new_y+y_sol[i] - car_h, 'WIDTH':car_w, 'HEIGHT':car_h}, ignore_index = True)
                        count += 1
                    elif gap >= car_w and (calc_nfp_reverse(new_x+x_sol[i], new_y+y_sol[i], car_w, car_h) == False):
                        reverse_sheet[new_x] -= 1
                    else:
                        if new_x == 0:
                            tonari = reverse_sheet[gap]
                        elif new_x + gap == w_sol[i]:
                            tonari = reverse_sheet[new_x - 1]
                        else:
                            tonari = max(reverse_sheet[new_x - 1], reverse_sheet[new_x + gap])
                        for j in range(gap):
                            reverse_sheet[new_x + j] = tonari
                    
                df.iloc[i,3] -= count
                
        # remain_car[DK] += car_amount - count
    if output == 1:
        for i in range(len(df_ramp)):
                ramp = patches.Rectangle(xy=(df_ramp.at[i,'X'], df_ramp.at[i,'Y']), width = df_ramp.at[i,'WIDTH'], height = df_ramp.at[i,'HEIGHT'], fc = 'silver', ec = 'k', linewidth = 0.2)
                axes[4-DK].add_patch(ramp)
    # make_arrow(DK) 
    if output == 1:
        print(df)

def bl_packing(stock_sheet,count):
    new_x, new_y, gap = find_lowest_gap(stock_sheet, w_sol[i])
    print(new_x, new_y)
    # print(stock_sheet)
    if new_y + car_h > h_sol[i]:
        return
    # DP,LPによる通路制約は今は省略
    if gap >= car_w and (calc_nfp(new_x+x_sol[i], new_y+y_sol[i], car_w, car_h) == True):
        for j in range(car_w):
            stock_sheet[new_x + j] += car_h
        count += 1
        
    elif gap >= car_w and calc_nfp(new_x+x_sol[i], new_y+y_sol[i], car_w, car_h) == False:
        stock_sheet[new_x] += 1
    else:
        if new_x == 0:
            tonari = stock_sheet[gap]
        elif new_x + gap == w_sol[i]:
            tonari = stock_sheet[new_x - 1]
        else:
            tonari = min(stock_sheet[new_x - 1], stock_sheet[new_x + gap ])
        for j in range(gap):
            stock_sheet[new_x + j] = tonari

def tl_packing(reverse_sheet,count):
    new_x, new_y, gap = find_highest_gap(reverse_sheet, w_sol[i])
    if new_y - car_h < 0:
        return
    # DP,LPによる通路制約は省略
    if gap >= car_w and (calc_nfp_reverse(new_x+x_sol[i], new_y+y_sol[i], car_w, car_h) == True):
        for j in range(car_w):
            reverse_sheet[new_x + j] -= car_h
        count += 1
    elif gap >= car_w and (calc_nfp_reverse(new_x+x_sol[i], new_y+y_sol[i], car_w, car_h) == False):
        reverse_sheet[new_x] -= 1
    else:
        if new_x == 0:
            tonari = reverse_sheet[gap]
        elif new_x + gap == w_sol[i]:
            tonari = reverse_sheet[new_x - 1]
        else:
            tonari = max(reverse_sheet[new_x - 1], reverse_sheet[new_x + gap])
        for j in range(gap):
            reverse_sheet[new_x + j] = tonari


center_line_list = [0,1,2,3,4,5,6,7,1000,1050,1300,1000,600]

def new_detailed_packing(DK):
    print('今から'+str(DK)+'dkに詰め込みます')
    global new_x, new_y
    global df_obs, obs
    global nfp, nfp_p
    global center_line_list
    global stock_sheet, reverse_sheet
    global x_sol, y_sol, w_sol, h_sol
    global count
    global car_w, car_h, car_amount
    
    df, df_ship, df_ramp, df_obs, df_aisle = datainput(DK)
    center_line_list = [0,1,2,3,4,5,6,7,1000,1050,1300,1000,600]
    
    df_lp = df.sort_values(by=['SEG','LP','DP'], ascending = [True, True, False])
    lp_order = [df_lp.iloc[i,0] for i in range(len(df_lp))]

    df_car = pd.read_csv('data/new_data/car'+str(DK)+'_1.csv')
    df_car = df_car.sort_values(by=['SEG','LP','DP','HEIGHT'], ascending=[True,True,False,False])
    
    for i in lp_order:
        global width
        width = w_sol[i]
        stock_sheet = [0]*w_sol[i]
        reverse_sheet = [h_sol[i]]*w_sol[i]
        group_i = (df_car['SEG'] == df_lp.at[i,'SEG']) & (df_car['LP'] == df_lp.at[i,'LP']) & (df_car['DP'] == df_lp.at[i,'DP'])
        print(group_i)
        print(group_i.sum())
        for car in range(group_i.sum()): # for car in group(i)にしたい
            car_w = df_car.iloc[car,1]
            car_h = df_car.iloc[car,2]
            car_amount = df_car.iloc[car,3]
            car_seg = df_car.iloc[car,4]
            car_lp = df_car.iloc[car,6]
            car_dp = df_car.iloc[car,7]
            count = 0
            
            while car_amount != count:
                new_x,new_y,gap = find_lowest_gap(stock_sheet, w_sol[i])
                print(new_x, new_y, gap)
                if new_y + y_sol[i] > h_sol[i]:
                    break
                if new_y + y_sol[i] > center_line_list[DK]:
                    print('tl_p')
                    tl_packing(stock_sheet,count)
                else:
                    bl_packing(reverse_sheet,count)
                    # print('bl_p')
            
            df.iloc[i,3] -= count


group_packing(0,1,0)
new_detailed_packing(12)


# main
def packing():
    global b
    for DK in range(5):
        start = time.time()
        bestvalue = 1000
        bestsol = 1
        for b in range(4,5):
            best = b/10 + 1
            group_packing(DK,best,0)
            if model.Status != gp.GRB.OPTIMAL or model2.Status != gp.GRB.OPTIMAL:
                continue
            detailed_packing(DK,0)
            if bestvalue >= remain_car[DK]:
                bestvalue = remain_car[DK]
                bestsol = best
        group_packing(DK,bestsol,1)
        detailed_packing(DK,1)
        make_arrow(DK)
        print('このデッキの余りは{}台です．'.format(remain_car[DK]))
        print(bestsol)
        end = time.time()
        print(str(12-DK)+'の計算時間:{:.1f}s'.format(end-start))

# packing()

def packing_func(DK):
    global b
    bestvalue = 1000
    bestsol = 1
    for b in range(10):
        best = b/10 + 1
        group_packing(DK,best,0)
        if model.Status != gp.GRB.OPTIMAL or model2.Status != gp.GRB.OPTIMAL:
            continue
        detailed_packing(DK,0)
        if bestvalue >= remain_car[DK]:
            bestvalue = remain_car[DK]
            bestsol = best
    group_packing(DK,bestsol,1)
    detailed_packing(DK,1)
    make_arrow(DK)
    print('このデッキの余りは{}台です．'.format(remain_car[DK]))
    print(bestsol)


# process_list = []
# if __name__ == '__main__':
#     for i in range(5):
#         process = Process(
#             target=packing_func(i),
#         )
#         process.start()
#         process_list.append(process)

#     for process in process_list:
#         process.join()
    

sum_remain = sum(remain_car)
print(remain_car)
print('合計で{}台余っています'.format(sum(remain_car)))

t2 = time.time()
print('計算時間:{:.1f}s'.format(t2-t1))


plt.axis("scaled")
plt.tight_layout()
# ax.set_aspect('equal')
plt.savefig('output_data/output.png', dpi = 2000)


t3 = time.time()
print('書き出し時間:{:.1f}s'.format(t3-t2))

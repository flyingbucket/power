import numpy as np
import networkx as nx
from scipy.optimize import linprog
import pandas as pd

data_path='D:\mypython\math_modeling\power\data.xlsx'
# 读取数据
df_nodes = pd.read_excel(data_path, sheet_name='node', header=0)
df_edges = pd.read_excel(data_path, sheet_name='edge', header=0)
# 创建无向图
G = nx.Graph()

'''
 创建节点,consume表示该节点的总消耗量,consume是上层目标最小化传输成本的决策变量,
 cc(consume clean)和cd(consumedirty)为决策变量，表示该节点的清洁能源和非清洁能源消耗量,是下层目标最小化碳排放的决策变量
 下层约束条件是cc+cd=consume
'''
for index,row in df_nodes.iterrows():
    G.add_node(row['name'], s_clean=row['s_clean'], s_dirty=row['s_dirty'], demand=row['demand'],
               cc=0.0,cd=0.0,consume=0.0)
# 添加边和传输成本，tc和td为决策变量，表示该边的清洁能源和非清洁能源传输量，是下层目标的决策变量，t为总传输量，是上层目标最小化传输成本的决策变量
for index,row in df_edges.iterrows():
    G.add_edge(row['end1'], row['end2'], price=row['price'],tc=0.0,td=0.0,t=0.0)

node_ls=list(G.nodes(data=True))
edge_ls=list(G.edges(data=True))

print(node_ls)
print(edge_ls)
# def cost():
# 求解上层目标

# 上层目标函数系数，即各边的传输成本
c = []  
for u, v, data in edge_ls:
    c.append(data['price'])
print(c)
# 上层约束条件，即各节点的总消耗量
A_ubi=np.zeros(shape=(5,5))
for i in range(5):
    A_ubi[i][i-1]=1
    A_ubi[i][i]=-1
# print(A_ubi)

b_ubi=df_nodes['demand'].values
# print(b_ubi)


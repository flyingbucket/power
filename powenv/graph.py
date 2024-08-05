import numpy as np
import networkx as nx
from scipy.optimize import linprog
import pandas as pd

data_path='D:\mypython\math_modeling\power\data.xlsx'
# 读取数据
df_nodes = pd.read_excel(data_path, sheet_name='node', header=0)
df_edges = pd.read_excel(data_path, sheet_name='edge', header=0)
print(df_nodes)
print(df_edges)
breakpoint()
# 创建无向图
G = nx.Graph()

# 创建节点，cc（consume clean)和cd(consumedirty)为决策变量，表示该节点的清洁能源和非清洁能源消耗量
for index,row in df_nodes.iterrows():
    G.add_node(row['name'], s_clean=row['s_clean'], s_dirty=row['s_dirty'], rest=row['rest'],cc=None,cd=None)
# 添加边和传输成本，tc和td为决策变量，表示该边的清洁能源和非清洁能源传输量
for index,row in df_edges.iterrows():
    G.add_edge(row['end1'], row['end2'], cost=row['cost'],tc=None,td=None)

# 提取节点和边数据
nodes = list(G.nodes(data=True))
edges = list(G.edges(data=True))

# 构建线性规划问题
c = []  # 目标函数系数
A_eq = []  # 等式约束矩阵
b_eq = []  # 等式约束向量

# 添加传输成本到目标函数
for u, v, data in edges:
	c.append(data['cost'])
breakpoint()
# 添加碳排放到目标函数
for node, data in nodes:
	s_clean = data['s_clean']
	s_dirty = data['s_dirty']
	rest = data['rest']
	c.append(s_clean * 100 + s_dirty* 500)  # 假设环保电力碳排放系数为100，非环保电力为500

# 添加供需平衡约束
for node, data in nodes:
	row = [0] * len(edges)
	for i, (u, v, _) in enumerate(edges):
		if u == node:
			row[i] = -1
		elif v == node:
			row[i] = 1
	A_eq.append(row)
	b_eq.append(data['rest'])

# 求解线性规划问题
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

# 输出结果
if result.success:
	print("Optimal solution found:")
	for i, (u, v, data) in enumerate(edges):
		print(f"Power from {u} to {v}: {result.x[i]}")
else:
	print("No optimal solution found.")

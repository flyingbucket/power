import numpy as np
import networkx as nx
from scipy.optimize import linprog,minimize
import pandas as pd
import matplotlib.pyplot as plt

data_path='D:\mypython\math_modeling\power\data.xlsx'
# 读取数据
df_nodes = pd.read_excel(data_path, sheet_name='node', header=0)
df_edges = pd.read_excel(data_path, sheet_name='edge', header=0)
# 创建无向图
G = nx.Graph()

'''
 创建节点,consume表示该节点的总消耗量,consume是上层目标最小化传输成本的决策变量,
 cc(consume clean)和cd(consume dirty)为决策变量，表示该节点的清洁能源和非清洁能源消耗量,是下层目标最小化碳排放的决策变量
 下层约束条件是cc+cd=consume
'''
for index,row in df_nodes.iterrows():
    G.add_node(row['name'], s_clean=row['s_clean'], s_dirty=row['s_dirty'], demand=row['demand'],
               cc=0.0,cd=0.0,consume=0.0)
'''
创建边,price表示该传输线路的传输价格,t为总传输量,是上层目标最小化传输成本的决策变量,
tc(trans clean)和td(trans dirty)为决策变量,表示该边的清洁能源和非清洁能源传输量,是下层目标的决策变量
'''
for index,row in df_edges.iterrows():
    G.add_edge(row['end1'], row['end2'], price=row['price'],tc=0.0,td=0.0,t=0.0)

node_ls=list(G.nodes(data=True))
edge_ls=list(G.edges(data=True))

# 求解上层目标

# 上层目标函数系数，即各边的传输成本
cup = []  
for u, v, data in edge_ls:
    cup.append(data['price'])
# print(c)
cup_liner=np.hstack((cup,cup))
# print(c_liner)

# 上层约束条件，即各节点的总消耗量
A_ub1=np.zeros(shape=(5,5))
for i in range(5):
    A_ub1[i][i-1]=1
    A_ub1[i][i]=-1

A_ub2=np.hstack((-A_ub1,A_ub1))

b_ub1=df_nodes['rest'].values
b_ub1=np.array(b_ub1)


up_res0=linprog(cup_liner,A_ub=A_ub2,b_ub=b_ub1,method='highs')
up_res=up_res0.x[:5]-up_res0.x[5:]
cost=up_res0.fun
print(f"各个边的传输量为{up_res}")  
print(f"最小传输成本为{cost*0.1}万元")

# 更新edge的传输量t
for i,(u,v,data) in enumerate(edge_ls):
    data['t']=up_res[i]

# 求解下层目标,碳排放量
def obj(x):
    cdw=[]
    # cdw前十位是各个节点的cc，cd的碳排放系数
    for i in range(1,11):
        if i%2==1:
            cdw.append(100)
        else:
            cdw.append(500)

    # 决策变量11至20位是各边的tc,td，不参与碳排放量计算，故系数用0占位
    cdw_liner=np.hstack((cdw,np.zeros(10)))
    return cdw_liner@x

# 下层约束条件，各节点的总消耗量构成不等式约束(消耗不多于拥有）和等式约束(总消耗等于需求)，各边的传输量构成等式约束

# 各边的传输量构成等式约束
def eq_cons(x):
    A_eq_post=np.zeros(shape=(5,10))
    for i in range(5):
        A_eq_post[i][2*i]=1
        A_eq_post[i][2*i+1]=1
 
    A_eq1=np.hstack((np.zeros((5,10)),A_eq_post))

    B_eq1=up_res

    #各节点等式约束(总消耗等于需求)
    A_eq2_pre=np.zeros(shape=(5,10))
    for i in range(5):
        A_eq2_pre[i][2*i]=1
        A_eq2_pre[i][2*i+1]=1

    A_eq2=np.hstack((A_eq2_pre,np.zeros(shape=(5,10))))
    A_eq=np.vstack((A_eq1,A_eq2))
    B_eq2=df_nodes['demand'].values
    B_eq=np.hstack((B_eq1,B_eq2))

    return B_eq-A_eq@x

def ub_cons(x):
    A_ub_post=np.zeros(shape=(10,10))
    for i in range(10):
        A_ub_post[i][i]=1
        A_ub_post[i][i-2]=-1

    A_ub_pre=np.zeros((10,10))
    for i in range(10):
        A_ub_pre[i][i]=1

    A_ub=np.hstack((A_ub_pre,A_ub_post))
    B_ub_ls=[]
    for node,data in node_ls:
        B_ub_ls.append(data['s_clean'])
        B_ub_ls.append(data['s_dirty'])
    B_ub=np.array(B_ub_ls)

    return B_ub-A_ub@x

def nl_cons(x):
    temp=[]
    for i in range(5):
        temp.append(x[10+2*i]*x[10+2*i+1])
    temp=np.array(temp)
    return temp

cons1={'type':'eq','fun':eq_cons}
cons2={'type':'ineq','fun':ub_cons}
cons3={'type':'ineq','fun':nl_cons}
cons=[cons1,cons2,cons3]


x0=[150.,0.,0.,200.,180.,0.,130.,0.,120.,10.,-5.,-5.,0,-10,-10,0.,0.,0.,0.,0.]
x1=np.zeros(20)
dw_res0=minimize(obj,x0=x1,constraints=cons,method='SLSQP',
                 options={'disp':True})
format_x=[round(i,2) for i in dw_res0.x]
print(format_x)
print(f"产生二氧化碳{10**(-3)*dw_res0.fun:.2f}吨")

for i in range(5):  # 假设 node_ls 和 edge_ls 的长度为 10
    node, node_data = node_ls[i]  # 解包元组
    node_data['cc'] = format_x[2*i]
    node_data['cd'] = format_x[2*i + 1]
    end1,end2, edge_data = edge_ls[i]  # 解包元组
    edge_data['tc'] = format_x[10 + 2 * i]
    edge_data['td'] = format_x[10 + 2 * i + 1]


# 创建空的 DataFrame
node_data = []
edge_data = []

# 遍历节点并添加到 DataFrame
for node in node_ls:
    name, data = node  # 解包元组
    data['consume'] = data['cc'] + data['cd']
    node_data.append({'name': name, **data})

# 遍历边并添加到 DataFrame
for edge in edge_ls:
    end1, end2, data = edge  # 解包元组
    data['t'] = data['tc'] + data['td']
    edge_data.append({'end1': end1, 'end2': end2, **data})

# 创建 DataFrame
node_df = pd.DataFrame(node_data)
edge_df = pd.DataFrame(edge_data)

# 写入到 Excel 文件
with pd.ExcelWriter('D:\mypython\math_modeling\power\graph_data.xlsx') as writer:
    node_df.to_excel(writer, sheet_name='Nodes', index=False)
    edge_df.to_excel(writer, sheet_name='Edges', index=False)

# 画图
node_labels = {n: f"cc: {d['cc']}\ncd: {d['cd']}" for n, d in G.nodes(data=True)}
edge_labels = {(u, v): f"tc: {d['tc']}\ntd: {d['td']}" for u, v, d in G.edges(data=True)}
pos = nx.spring_layout(G)  # 使用spring布局
offset_pos = {node: (coords[0] + 0.1, coords[1] + 0.1) for node, coords in pos.items()}

nx.draw(G, pos, with_labels=True, node_color='skyblue', 
        node_size=2000, edge_color='gray', font_size=15, font_color='black')

nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, 
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6),
                                  horizontalalignment='left', verticalalignment='bottom',)

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# 显示图形
plt.show()


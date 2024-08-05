import numpy as np

class Node:
    def __init__(self, name, s_clean, s_dirty, rest,edge_obj):
        self.name = name
        self.s_clean = s_clean
        self.s_dirty = s_dirty
        self.rest = rest
        self.edge_obj=edge_obj

        
class Edge:
    def __init__(self, start, end, cost):
        self.start = start
        self.end = end
        self.cost = cost
from scipy.spatial import KDTree 
import numpy as np

def path_optimization(args, stipples):
    #Greedy approach to path optimization
    #Just choose the closest node to current one
    p = [0, 0] 
    buf_stipples = stipples
    optimized_stipples = []
    
    while buf_stipples.shape[0] > 0: 
        kd = KDTree(buf_stipples, compact_nodes = False, balanced_tree = False)
        idx = kd.query(p)[1]
        p = buf_stipples[idx].tolist()
        buf_stipples = np.delete(buf_stipples, idx, axis = 0)
        optimized_stipples.append(p)

    return optimized_stipples 
 

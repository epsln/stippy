from scipy.spatial import KDTree 
import numpy as np

def path_optimization(args, stipples):
    p = stipples[0] 
    exclude_list = []
    
    kd = KDTree(stipples)

    for i in range(args.num_pts - 1):
        n = 2
        idx_list = kd.query(p, k = 15)[1]
        in_list = False 
        for idx in idx_list:
            if idx not in exclude_list:
                exclude_list.append(idx)
                in_list = True
                break

        if not in_list:
            while True:
                idx = np.random.randint(0, args.num_pts)
                if idx not in exclude_list:
                    exclude_list.append(idx)
                    break

        p = stipples[idx].tolist()

    optimized_stipples = []
    for idx in exclude_list:
        optimized_stipples.append(stipples[idx])

    return optimized_stipples 
 

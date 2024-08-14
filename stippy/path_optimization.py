from scipy.spatial import KDTree
import numpy as np


def path_optimization(stipples):
    """
    Optimize the path by reordering all points. The goal is to minimize pen-up
    movement in drawing robots, by finding the shortest path between all points.

    Unfortunately, this is a TSP problem, so we find a solution using a greedy
    nearest neighbor algorithm. Also, we can't use any TSP solver since the number
    of points where this is useful far outpace the capability of them.

    Parameters:
    ----------
    stipples: np.array
    List of 2D points

    Returns
    -------
    optimized_stipples: np.array
    Optimized list of 2D points, where each points is placed next to its closest neighbord

    """
    p = [0, 0]
    buf_stipples = stipples
    optimized_stipples = []

    while buf_stipples.shape[0] > 0:
        kd = KDTree(buf_stipples, compact_nodes=False, balanced_tree=False)
        idx = kd.query(p)[1]
        p = buf_stipples[idx].tolist()
        buf_stipples = np.delete(buf_stipples, idx, axis=0)
        optimized_stipples.append(p)

    return optimized_stipples

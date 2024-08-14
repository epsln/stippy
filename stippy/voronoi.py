from multiprocessing import Pool

import numpy as np
from scipy.spatial import KDTree
import cv2

from .path_optimization import path_optimization


def split(a, n):
    """
    Split a list into N roughly equal sub list

    Parameters
    ----------
    a: List
    n: Number of sub list to create

    Returns
    -------
    List[List] 
    List of list containing the a array split into N chunks
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def weighted_centroid_compute(num_points, img, xy_grid, idx_list):
    """
    Compute the weighted centroid of all regions contained in the idx_list. Used in 
    parallel. The points array contains an xy_grid, and the idx_list contains
    the idx of the corresponding centroid for each point in the xy_grid 

    Parameters
    ----------
    num_points: int
    Number of centroids

    img: np.array
    Input image to use for the weighted centroid

    xy_grid: List[List]
    List of 2D points 

    idx_list: List[int]
    List of index of voronoi region, corresponding to each point in xy_grid

    Returns
    -------
    centroids: List[List]
    List containing the 2D position of the centroid.
    """
    centroids = np.zeros((num_points, 3))
    for pt, idx in zip(xy_grid, idx_list):
        x, y = pt[0], pt[1]
        i, j = int(x * img.shape[0] - 1), int(y * img.shape[1] - 1)

        centroids[idx][0] += x * (1 - img[i, j] / 255.0)
        centroids[idx][1] += y * (1 - img[i, j] / 255.0)
        centroids[idx][2] += 1 - img[i, j] / 255.0

    return centroids


def rejection_sampling(num_points, img):
    """
    Creates an initial distribution of points that is roughly follows the distribution of the image.

    Parameters
    ----------
    num_points: int
    Number of points to generate.
    
    img: np.array
    The input image.

    Returns
    -------
    seed_points: np.array
    List of 2D points
    """
    seed_pts = np.zeros((num_points, 2))
    for i in range(num_points):
        for _ in range(500):
            x = np.random.randint(img.shape[0] - 1)
            y = np.random.randint(img.shape[1] - 1)
            seed_pts[i] = np.array([x * 1.0 / img.shape[0], y * 1.0 / img.shape[1]])
            if np.random.uniform() < 1 - img[x, y] / 255.0:
                break

    return seed_pts


def compute_points(args, img):
    """
    Compute the weighted voronoi of an image, and converge the voronoi points to their centroids. 
    Implementation of https://www.cs.ubc.ca/labs/imager/tr/2002/secord2002b/secord.2002b.pdf

    Parameters
    ----------
    args: dict
    Input arguments

    img: np.array
    Input image

    Returns:
    stipples: np.array
    List of 2D points corresponding to stipples on the image
    """

    x_pts = np.linspace(0, 1, img.shape[0] - 1)
    y_pts = np.linspace(0, 1, img.shape[1] - 1)
    xy_grid = [[x, y] for x in x_pts for y in y_pts]

    stipples = rejection_sampling(args.num_pts, img)

    for n in range(args.num_iter):
        if args.debug:
            output_image = np.zeros(img.shape)
        w = np.power(n + 1, -0.8) * args.learning_rate
        kd = KDTree(stipples)
        centroids = np.zeros((args.num_pts, 3))
        _, idx_list = kd.query(xy_grid, workers=args.num_workers)

        func_args = []
        for pts, idxs in zip(
            split(xy_grid, args.num_workers), split(idx_list, args.num_workers)
        ):
            sp = np.array([stipples[idx] for idx in idxs])
            func_args.append((args.num_pts, img, pts, idxs))

        with Pool(args.num_workers) as p:
            output = p.starmap(weighted_centroid_compute, func_args)

        for o in output:
            for i, c in enumerate(o):
                centroids[i][0] += c[0]
                centroids[i][1] += c[1]
                centroids[i][2] += c[2]

        for idx in set(idx_list):
            cen = centroids[idx]
            sp = stipples[idx]
            if cen[2] != 0:
                x = cen[0] / cen[2]
                y = cen[1] / cen[2]
            else:
                x = sp[0]
                y = sp[1]

            stipples[idx][0] -= (sp[0] - x) * w
            stipples[idx][1] -= (sp[1] - y) * w

        if args.debug:
            for sp in stipples:
                if 0 < sp[0] < 1 and 0 < sp[1] < 1:
                    output_image[
                        int(sp[0] * img.shape[0]), int(sp[1] * img.shape[1])
                    ] = 255
            cv2.imwrite(f"out/debug_{n}.jpg", output_image)

    if args.opti:
        stipples = path_optimization(stipples)

    return stipples

import numpy as np
from multiprocessing import Pool
from scipy.spatial import KDTree 
import cv2
import numpy as np

from .path_optimization import path_optimization

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def weighted_centroid_compute(args, w, img, points, seed_pts, idx_list):
    #TODO: Remove seed_pts args
    centroids = np.zeros((args.num_pts, 3))
    for pt, idx in zip(points, idx_list):
        x, y = pt[0], pt[1]
        i, j = int(x * img.shape[0] - 1), int(y * img.shape[1] - 1)

        centroids[idx][0] += x * (1 - img[i, j]/255.)
        centroids[idx][1] += y * (1 - img[i, j]/255.)
        centroids[idx][2] +=      1 - img[i, j]/255.

    return centroids

def rejection_sampling(num_points, img):
    #Generate random seed points lieing inside the input image using rejection sampling
    seed_pts = np.zeros((num_points, 2)) 
    for i in range(num_points):
        for j in range(500):
            x = np.random.randint(img.shape[0] - 1)
            y = np.random.randint(img.shape[1] - 1)
            seed_pts[i] = np.array([x * 1.0/img.shape[0], y * 1.0/img.shape[1]])
            if np.random.uniform() < 1 - img[x, y]/255.:
                break

    return seed_pts

def compute_points(args, img):
    #Generate a Voronoi Diagram
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
        _, idx_list = kd.query(xy_grid, workers = args.num_workers)

        func_args = []
        for pts, idxs in zip(split(xy_grid, args.num_workers), split(idx_list, args.num_workers)):
            sp = np.array([stipples[idx] for idx in idxs])
            func_args.append((args, w, img, pts, stipples, idxs))

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
                x = cen[0]/ cen[2]
                y = cen[1]/ cen[2]
            else:
                x = sp[0]
                y = sp[1]
            
            stipples[idx][0] -= (sp[0] - x) * w 
            stipples[idx][1] -= (sp[1] - y) * w

        if args.debug:
            for sp in stipples:
                if 0 < sp[0] < 1 and 0 < sp[1] < 1:
                    output_image[int(sp[0] * img.shape[0]), int(sp[1] * img.shape[1])] = 255
            cv2.imwrite(f"out/debug_{n}.jpg", output_image)

    if args.opti: 
        stipples = path_optimization(args, stipples)

    return stipples

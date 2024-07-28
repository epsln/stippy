import numpy as np

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def weighted_centroid_compute(args, w, img, points, seed_pts, idx_list):
    centroids = np.zeros((args.num_pts, 3))
    for pt, idx in zip(points, idx_list):
        x, y = pt[0], pt[1]
        i, j = int(x * img.shape[0]), int(y * img.shape[1])

        centroids[idx][0] += x * (1 - img[i, j]/255.)
        centroids[idx][1] += y * (1 - img[i, j]/255.)
        centroids[idx][2] +=      1 - img[i, j]/255.

    return centroids

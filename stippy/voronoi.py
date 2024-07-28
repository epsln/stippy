import numpy as np

def weighted_centroid_compute(args, w, img, points, seed_pts, idx_list):
    centroids = np.zeros((args.num_pts, 3))

    for pt, idx in zip(points, idx_list):
        x, y = pt[0], pt[1]
        i, j = int(x * img.shape[0]), int(y * img.shape[1])

        centroids[idx][0] += x * (1 - img[i, j]/255.)
        centroids[idx][1] += y * (1 - img[i, j]/255.)
        centroids[idx][2] +=      1 - img[i, j]/255.

    i = 0 
    output_points = seed_pts
    for sp, cen in zip(seed_pts, centroids):
        if cen[2] != 0:
            x = cen[0]/ cen[2]
            y = cen[1]/ cen[2]
        else:
            x = sp[0]
            y = sp[1]
        
        output_points[i][0] -= (sp[0] - x) * w 
        output_points[i][1] -= (sp[1] - y) * w
        
        i += 1

    return output_points


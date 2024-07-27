import argparse
import cv2
import numpy as np
from scipy.spatial import KDTree 
from pathlib import Path

parser = argparse.ArgumentParser("stippy")
parser.add_argument("input_filename", help="Input filename", type=Path)
parser.add_argument("-p", help="Total number of points using for stippling", default = 10000, dest = "num_pts", type=int)
parser.add_argument("-n", help="Number of times the Lloyds algorithm is applied. Higher is slower but yields better results.", dest = "num_iter", default = 50, type=int)
parser.add_argument("-lr", help="Learning rate used for relaxation. Lower is slower but yield better results.", dest = "learning_rate", default = 25, type=float)
args = parser.parse_args()

img = cv2.imread(args.input_filename.absolute().as_posix())
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Generate random seed points lieing inside the input image using rejection sampling
seed_pts = np.zeros((args.num_pts, 2)) 
for i in range(args.num_pts):
    for j in range(50):
        x = np.random.randint(img_gray.shape[0] - 1)
        y = np.random.randint(img_gray.shape[1] - 1)
        seed_pts[i] = np.array([x, y])
        if np.random.uniform() < 1 - img_gray[x, y]/255.:
            break

#Generate a Voronoi Diagram
x_pts = [i for i in range(img_gray.shape[0])] 
y_pts = [i for i in range(img_gray.shape[1])] 
pts = [[x, y] for x in x_pts for y in y_pts]
for n in range(args.num_iter):
    kd = KDTree(seed_pts)
    out_img = np.zeros((img_gray.shape[0], img_gray.shape[1], 3))
    centroids = np.zeros((args.num_pts, 3))
    _, idx_list = kd.query(pts, workers = 4)
    for idx in idx_list:
        i, j = int(seed_pts[idx][0]), int(seed_pts[idx][1])
        if 0 < i < img_gray.shape[0] and 0 < j < img_gray.shape[1]:
            centroids[idx][0] += i * (1 - img_gray[i , j]/255.)
            centroids[idx][1] += j * (1- img_gray[i, j]/255.)
            centroids[idx][2] += 1 - img_gray[i, j]/255.

    out = []
    w = np.power(n + 1, -0.8) * args.learning_rate 
    for sp, cen in zip(seed_pts, centroids):
        if cen[2] != 0:
            x = cen[0]/ cen[2]
            y = cen[1]/ cen[2]
        else:
            x = sp[0]
            y = sp[1]
        
        sp[0] += (sp[0] - x) * np.random.uniform(-1, 1) * w 
        sp[1] += (sp[1] - y) * np.random.uniform(-1, 1) * w

        if 0 < sp[0] < img_gray.shape[0] and 0 < sp[1] < img_gray.shape[1]:
            out_img[int(sp[0])][int(sp[1]), :] = [255, 255, 255] 

    output_filename = str(args.input_filename.with_stem(f"{args.input_filename.stem}_stippled_{n}"))
    cv2.imwrite(f"out/{output_filename}", out_img)

from multiprocessing import Pool
import argparse
import cv2
import numpy as np
from scipy.spatial import KDTree 
import svgwrite
from pathlib import Path
from .voronoi import split, weighted_centroid_compute
from svgwrite import cm, mm   
from python_tsp.heuristics import solve_tsp_simulated_annealing

def main():
    parser = argparse.ArgumentParser("stippy")
    parser.add_argument("input_filename", help="Input filename", type=Path)
    parser.add_argument("-p", help="Total number of points using for stippling", default = 10000, dest = "num_pts", type=int)
    parser.add_argument("-n", help="Number of times the Lloyds algorithm is applied. Higher is slower but yields better results.", dest = "num_iter", default = 50, type=int)
    parser.add_argument("-lr", help="Learning rate used for relaxation. Lower is slower but yield better results.", dest = "learning_rate", default = 25, type=float)
    parser.add_argument("-w", help="Number of workers used in the KDTree.", dest = "num_workers", default = 4, type=int)
    parser.add_argument("-inv", help="Invert the colors in the input image", action = "store_true", dest = "invert_img", default = False)
    parser.add_argument("-dpi", help="Set the DPI for the output svg.", dest = "dpi", default = 300, type = int)
    parser.add_argument("--debug", help="Activate debug mode", action = "store_true", dest = "debug", default = False)
    args = parser.parse_args()

    img = cv2.imread(args.input_filename.absolute().as_posix())

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.invert_img:
        img_gray = 255 - img_gray

    output_filename = args.input_filename.with_suffix("." + "svg")


    #Generate random seed points lieing inside the input image using rejection sampling
    seed_pts = np.zeros((args.num_pts, 2)) 
    for i in range(args.num_pts):
        for j in range(500):
            x = np.random.randint(img_gray.shape[0] - 1)
            y = np.random.randint(img_gray.shape[1] - 1)
            seed_pts[i] = np.array([x * 1.0/img_gray.shape[0], y * 1.0/img_gray.shape[1]])
            if np.random.uniform() < 1 - img_gray[x, y]/255.:
                break

    #Generate a Voronoi Diagram
    x_pts = np.linspace(0, 1, img_gray.shape[0] - 1) 
    y_pts = np.linspace(0, 1, img_gray.shape[1] - 1) 
    points = [[x, y] for x in x_pts for y in y_pts]

    for n in range(args.num_iter):
        if args.debug:
            output_image = np.zeros(img_gray.shape) 
        w = np.power(n + 1, -0.8) * args.learning_rate 
        kd = KDTree(seed_pts)
        centroids = np.zeros((args.num_pts, 3))
        _, idx_list = kd.query(points, workers = args.num_workers)

        func_args = []
        for pts, idxs in zip(split(points, args.num_workers), split(idx_list, args.num_workers)):
            sp = np.array([seed_pts[idx] for idx in idxs])
            func_args.append((args, w, img_gray, pts, seed_pts, idxs))

        with Pool(args.num_workers) as p:
            output = p.starmap(weighted_centroid_compute, func_args) 

        for o in output:
            for i, c in enumerate(o):
                centroids[i][0] += c[0]
                centroids[i][1] += c[1]
                centroids[i][2] += c[2]

        for idx in set(idx_list):
            cen = centroids[idx]
            sp = seed_pts[idx]
            if cen[2] != 0:
                x = cen[0]/ cen[2]
                y = cen[1]/ cen[2]
            else:
                x = sp[0]
                y = sp[1]
            
            seed_pts[idx][0] -= (sp[0] - x) * w 
            seed_pts[idx][1] -= (sp[1] - y) * w

        if args.debug:
            for sp in seed_pts:
                if 0 < sp[0] < 1 and 0 < sp[1] < 1:
                    output_image[int(sp[0] * img_gray.shape[0]), int(sp[1] * img_gray.shape[1])] = 255
            cv2.imwrite(f"out/debug_{n}.jpg", output_image)
    p = seed_pts[0] 
    exclude_list = []
    for i in range(seed_pts.shape[0] - 1):
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
                idx = np.random.randint(0, seed_pts.shape[0])
                if idx not in exclude_list:
                    exclude_list.append(idx)
                    break

        p = seed_pts[idx].tolist()
    
    svg_size = (img_gray.shape[1] / args.dpi * 2.54 * 10, img_gray.shape[0] / args.dpi * 2.54 * 10)
    dwg = svgwrite.Drawing(output_filename, size = (f'{svg_size[0]}mm', f'{svg_size[1]}mm'))
    for idx in exclude_list:
        sp = seed_pts[idx]
        #TODO: Set point size as argument
        dwg.add(dwg.circle((sp[1] * svg_size[0] * mm, sp[0] * svg_size[1] * mm), r = 0.25, fill = 'black'))

    dwg.save()

if __name__ == "__main__":
    main()

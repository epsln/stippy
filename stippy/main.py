import argparse
import cv2
import numpy as np
import svgwrite
from pathlib import Path
from .voronoi import compute_points 
from svgwrite import cm, mm   

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

    stipples = compute_points(args, img_gray)
   
    svg_size = (img_gray.shape[1] / args.dpi * 2.54 * 10, img_gray.shape[0] / args.dpi * 2.54 * 10)
    dwg = svgwrite.Drawing(output_filename, size = (f'{svg_size[0]}mm', f'{svg_size[1]}mm'))
    for sp in stipples:
        #TODO: Set point size as argument
        dwg.add(dwg.circle((sp[1] * svg_size[0] * mm, sp[0] * svg_size[1] * mm), r = 0.25, fill = 'black'))

    dwg.save()

if __name__ == "__main__":
    main()

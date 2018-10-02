# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch

from scipy import misc
from os import listdir
from os.path import isfile, join

# import cv2

def main():

    srcpath = '/Users/des/Desktop/Minions/street_views/'
    destpath = '/Users/des/Desktop/Minions/cropped/'
    onlyfiles = [f for f in listdir(srcpath) if isfile(join(srcpath, f))]
    for name in onlyfiles:
        if name == ".DS_Store":
            continue

        id = 0
        img = misc.imread(join(srcpath,name))

        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(
            img, scale=500, sigma=0.9, min_size=10)

        print(name)
        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 2000:
                continue
            # distorted rects
            x, y, w, h = r['rect']
            if w / h > 1.2 or h / w > 1.2:
                continue
            candidates.add(r['rect'])

        # draw rectangles on the original image
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(img)

        for x, y, w, h in candidates:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

            cropped = img[y:y + h, x:x + w]
            cropped_name = destpath + name.split(".")[0] + str(id) + "." + name.split(".")[1]
            misc.imsave(cropped_name, cropped)
            id = id + 1
        break
            # plt.imshow(cropped)
            # plt.show()

        # plt.show()
        # fig.savefig(join(destpath,name))

if __name__ == "__main__":
    main()

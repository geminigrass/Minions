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

def main():

    # loading astronaut image
    # img = skimage.data.astronaut()
    # img = misc.imread('/Users/des/Desktop/Minions/street_views/2514 Berryessa Rd, San Jose, CA 95132.jpg')
    # print(img.__class__)

    srcpath = '/Users/des/Desktop/Minions/street_views/'
    destpath = '/Users/des/Desktop/Minions/ss_results/'
    onlyfiles = [f for f in listdir(srcpath) if isfile(join(srcpath, f))]
    for name in onlyfiles:
        img = misc.imread(join(srcpath,name))

        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(
            img, scale=500, sigma=0.9, min_size=10)

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
            # print(x, y, w, h)
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

        plt.show()
        # fig.savefig(join(destpath,name))

if __name__ == "__main__":
    main()

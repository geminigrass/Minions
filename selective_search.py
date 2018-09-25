import skimage.data
# import selective_search
# from .selectivesearch import selective_search


from selectivesearch import selective_search

img = skimage.data.astronaut()
# print(img)
img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
# regions[:10]
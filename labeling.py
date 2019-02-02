from roipoly import RoiPoly
from matplotlib import pyplot as plt
import pickle, os
import numpy as np

n = 46

def labeling(m=3):
    '''Iterate over all training images and label m rois in each image'''
    mask_list = []
    for i in range(n):
        i += 1
        img_name = str(i)+'.png'
        img = plt.imread(os.path.join('trainset',img_name))
        masks = []
        for j in range(m):
            plt.imshow(img)
            r = RoiPoly(color='r')
            mask = r.get_mask(img[:,:,0]) # create mask of roi
            masks.append(mask)
        #combine masks
        mask_result = masks[0] | masks[1]
        mask_result = mask_result | masks[2]
        mask_list.append(mask_result)
    pickle_out = open('mask_list',"wb")
    pickle.dump(mask_list, pickle_out)
    pickle_out.close()
    return mask_list
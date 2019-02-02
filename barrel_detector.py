'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np



class BarrelDetector():
    def __init__(self):
        '''
        Initilize your blue barrel detector with the attributes you need
        eg. parameters of your classifier
        '''
    def segment_image(self, img):
        '''
        Calculate the segmented image using a classifier
        eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
        call other functions in this class if needed
        Inputs:
        img - original image
        Outputs:
        mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
        '''
        pixel_len = img.shape[0] * img.shape[1]
        x = np.ones([pixel_len,7]) # 2 color spaces and bias term
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert to HSV color space
        img_flat = img_new.reshape(pixel_len,3)
        img_flat2 = img.reshape(pixel_len,3)    
        x[:,:3] = img_flat # feature 1-3 are HSV values
        x[:,3:6] = img_flat2 # FEATURE 4-6 are BGR values
        # import trained weights for classification
        w = np.array([[-0.0067286 ],[-0.08391962],[-0.9366656 ],[ 1.85043287],[-0.7109699 ],[-1.09063044],[ 0.35853911]])
        result = np.dot(x,w) # obtain prediction results with (-1,1) labels
        y_pred = (result>=0)
        mask_img = y_pred.reshape(img.shape[0],img.shape[1]) # reshape back to 2D image dimensions
        new_mask = mask_img.astype('uint8')
        return new_mask

    def get_bounding_box(self, img):
        '''
        Find the bounding box of the blue barrel
        call other functions in this class if needed
        Inputs:
        img - original image
        Outputs:
        boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
        where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list is from left to right in the image.
        '''
        mask_img = self.segment_image(img) # get masked image using segment function from above
        contours = get_contour(mask_img) # get processed contours from the masks
        cprop, boxes = process_props(contours) # get bounding boxes
        return boxes
    
def erode_dilate(mask,e_kernel = 2,d_kernel = 10,e_iter = 5 ,d_iter = 5):
    ''' Erode and dilate the mask image to reduce noise and combine segmented regions'''
    kernel_e = np.ones((e_kernel,e_kernel), np.uint8)
    kernel_d = np.ones((d_kernel,d_kernel), np.uint8)
    img_erosion = cv2.erode(mask, kernel_e, iterations = e_iter) 
    img_dilation = cv2.dilate(img_erosion, kernel_d, iterations = d_iter)
    return img_dilation
    
def get_contour(mask):
    '''Apply the erosion and dilation, and generate contours from the processed masks'''
    new_mask = erode_dilate(mask,2,4,2,4)
    contours, hiearchy = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def normalized_l2(prop):
    ''' Calculate the normalized L2 deviation of contours' centroids from the center of bounding boxes''' 
    x1,y1,x2,y2 = prop.bbox
    b = x2 - x1 # base length
    h = y2 -y1 # height
    loc_x,loc_y = prop.local_centroid
    bb_x = b/2 # bbox center x
    bb_y = h/2 # bbox center y
    l2_dist= ((loc_x-bb_x)/b)**2+((loc_y-bb_y)/h)**2 # calculate distance
    res = 1-np.sqrt(l2_dist) # calculate score
    return res

def process_props(contours):
    '''Obtain properties of contours'''
    all_props = []
    for c in contours:
        cc = cv2.drawContours(np.zeros([800,1200]), [c], 0, (255,0,0), 2)
        c_region = cc.astype('int32')
        props = regionprops(c_region)
        all_props.append(props)
    # sort props
    prop_sort = np.zeros([len(contours),4])
    for i in range(len(contours)):
        c_prop = all_props[i][0]
        area = c_prop.filled_area # filled contour area
        r_area = c_prop.filled_area/c_prop.bbox_area # area ratio between filled and bbox
        n_L2 = normalized_l2(c_prop) # distance from center
        prop_sort[i,0] = area
        prop_sort[i,1] = r_area
        prop_sort[i,2] = n_L2
        prop_sort[i,3] = i 
        Ars = -prop_sort[:,0] # sort aacording to filled area
        idxs = Ars.argsort() 
    sorted_prop = prop_sort[idxs]
    top_area = sorted_prop[:3] # get bbox of top 3 areas 
    result = []
    bboxs = []
    for i in range(top_area.shape[0]):
        target = top_area[i]
        orig_idx = int(target[3])
        if target[1] >= 0.75 and target[2] >= 0.94: # filter by area ratio >0.75 and l2 >= 0.94
            result.append(target)
            y1,x1,y2,x2 = all_props[orig_idx][0].bbox
            bboxs.append([x1,y1,x2,y2])
    return result,bboxs    




if __name__ == '__main__':
    folder = "trainset"
    my_detector = BarrelDetector()
    for filename in os.listdir(folder):
        
    # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#Display results:
#(1) Segmented images
#mask_img = my_detector.segment_image(img)
#(2) Barrel bounding box
#boxes = my_detector.get_bounding_box(img)
#The autograder checks your answers to the functions segment_image() and get_bounding_box()
#Make sure your code runs as expected on the testset before submitting to Gradescope


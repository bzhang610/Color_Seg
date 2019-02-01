'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops

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
        x[:,:3] = img_flat
        x[:,3:6] = img_flat2
        w = np.array([[ 0.04176607],[ 0.03761931],[-0.93647226],[ 1.89223607],[-0.76342823],[-1.21911078],[ 0.35824075]])
        result = np.dot(x,w)
        y_pred = (result>=0) * 2 - 1
        mask_img = y_pred.reshape(img.shape[0],img.shape[1]) # reshape back to 2D image dimensions
		return mask_img

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		# YOUR CODE HERE
		
		return boxes
    
    def process_x(img):
        pixel_len = img.shape[0] * img.shape[1]
        x = np.ones([pixel_len,7]) # 2 color spaces and bias term
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert to HSV color space
        img_flat = img_new.reshape(pixel_len,3)
        img_flat2 = img.reshape(pixel_len,3)    
        x[:,:3] = img_flat
        x[:,3:6] = img_flat2
        result = np.dot(x,w)
        y_pred = (result>=0) * 2 - 1
        mask_img = y_pred.reshape(img.shape)

    train_num = 35 #number of samples used for training
    test_num = 46 - train_num
    x_train = x[:train_num*pixel_len]
    y_train = y[:train_num*pixel_len]
    x_test = x[train_num*pixel_len:]
    y_test = y[train_num*pixel_len:]
    return (x_train,y_train,x_test,y_test)
    


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
		#	 mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope


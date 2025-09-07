import os
import random
import cv2
import PIL  
from utils.util import get_detections
import numpy as np

#Difne paths

#Path to mofel configuration file
config_path = r'16_Semantic_Segmentation\models\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

#Path to model weights
weights_path = r'16_Semantic_Segmentation\models\frozen_inference_graph.pb'

#Path to the class.names file
class_names_path = r'16_Semantic_Segmentation\models\class.names'

#Path to the image
image_path = r'16_Semantic_Segmentation\images\cat_and_dog.png'

#Load image
image = cv2.imread(filename=image_path)

#Get the height and width of the image
H, W, C = image.shape
#H: height
#W: width
#C: channels

#Load the model
net = cv2.dnn.readNetFromTensorflow(model=weights_path, config=config_path)
#model: path to the model weights
#config: path to the model configuration file

#Convert the image to blob
#blob is a format that is used to pass the image data to the network
#blob: Binary Large Object
blob = cv2.dnn.blobFromImage(image=image) 
#image: input image

#Get detections from the model
boxes, masks = get_detections(net=net, blob=blob)
#boxes: bounding boxes
#masks: mask for the object
#net: model
#blob: image data

#Draw the bounding boxes and masks on the image

#Create random colors
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(90)]

#define an empty image
mask_img = np.zeros((H, W, C))

#Print en error message if no object is detected
if len(masks) == 0:
    print('No object detected')
    
#create a class name list
class_name_number = []

#print(len(masks)) -> With the cat image we're having 100 objects detected so we have to clean the image
det_threshold = 0.5 #Define the detection threshold
#Iterate over the boxes and masks
for i in range(len(masks)): #Every iteration is a new object detected
    bbox = boxes[0, 0, i] #Get the bounding box
    #[0, 0, i]: 0: batch size, 0: number of classes, i: number of objects detected
    
    #print(len(mask)) #Print the length of the mask: 90 
    
    #Get class name
    class_name = bbox[1] #Get the class name
    
    #Get the score
    score = bbox[2] #Get the score
    
    if score > det_threshold: #How confident the model is that the object is detected
        #print(len(masks)) -> now we only have 1 object detected
        #print(bbox.shape) -> (7,) -> 7 elements in the bounding box
        #print(mask.shape) -> (90, 15, 15) -> array of 90 elements with dimensions 15x15
        
        mask = masks[i] #Get the mask
        
        #Get the coordinates of the bounding box
        x1, x2, y1, y2 = int(bbox[3] * W), int(bbox[5] * W), int(bbox[4] * H), int(bbox[6] * H)
        
        '''
        Show the bounding box on the image
        
        img = cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
        
        cv2.imshow(winname='image', mat=img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        #The only mask that is going to make sense is the one that corresponds to the class name
        #So we have to care about the class name
        #print(class_name) #Print the class name: 16.0 -> cat
        
        # Get the important mask
        mask = mask[int(class_name)]
        
        #Get the name of the class
        class_name_str = open(class_names_path, 'r').read().split('\n')[int(class_name) - 1]
        
        #print(mask.shape) #Print the shape of the mask: (90, 15, 15)
        
        #resize the mask to the size of the bounding box, resize the 15x15 to fit the bounding box
        mask = cv2.resize(mask, (x2 - x1, y2 - y1))
        
        _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

        #Iterate over the channels of the image
        for c in range(3):
            mask_img[y1:y2, x1:x2, c] = mask * colors[int(class_name)][c]
            #mask_img[y1:y2, x1:x2, c]: mask the bounding box
            #y1:y2 -> height
            #x1:x2 -> width
            #c -> channels
            #mask * colors[int(class_name)][c]: color the mask
            
        class_name_number.append(class_name_str)
            
            
#Overlay the mask on the image with a 60% transparency
overlay = ((0.6 * mask_img) + (0.4 * image)).astype("uint8")

for i in range(len(class_name_number)):
    #Print class name number on the image
    cv2.putText(overlay, class_name_number[i], (10, 30 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


#Show the images
cv2.imshow('mask', mask_img)
cv2.imshow('img', image)
cv2.imshow('overlay', overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()
def get_detections(net, blob): #Function to get the detections from the network
    net.setInput(blob) #Set the input to the network
    boxes, masks = net.forward(["detection_out_final", "detection_masks"]) #Get the detections from the network
    return boxes, masks #Return the detections
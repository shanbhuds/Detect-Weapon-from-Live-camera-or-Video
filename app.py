import cv2
import numpy as np
import os
import torch
import time
import cv2
import sys
from skimage import io
import matplotlib.pyplot as plt
from IPython.display import clear_output
import datetime
import imutils



# Load Yolo
# Download weight file(yolov3_training_2000.weights)
sound = "Police.mp3"
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")

classes = ["Weapon","Knief"]

layer_names = net.getLayerNames()
#output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = net.getUnconnectedOutLayersNames()  
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def cv2_imshow(a, **kwargs):
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    # https://matplotlib.org/stable/gallery/showcase/mandelbrot.html#sphx-glr-gallery-showcase-mandelbrot-py
    dpi = 72
    width, height = a.shape[1], a.shape[0]
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)  # Create new figure
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)  # Add axes to figure
    ax.imshow(a, **kwargs)
    plt.axis('off')
    plt.show(block=False)  # Show image without "blocking"        



# Enter file name for example "ak47.mp4" or press "Enter" to start webcam
def value():
    val = input("Enter file name or press enter to start webcam : \n")
    if val == "":
        val = 0
    return val


# for video capture
cap = cv2.VideoCapture(value())
fps= int(cap.get(cv2.CAP_PROP_FPS))

print("This is the fps ", fps)

while True:
    _, img = cap.read()
    if img is None:
        break
	
    #cv2.imshow('out',img)
    
    #fps = fps + 10
   # i=i+1 
    img = imutils.resize(img, width=500)

    height, width, channels = img.shape
    width = 512
    height = 512
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
  
    
    # draw the text and timestamp on the frame
    cv2.putText(img, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)



    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * (width/2))
                center_y = int(detection[1] * (height/2))
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #center_x = int(detection[0] )
                #center_y = int(detection[1] )
                #w = int(detection[2] )                #h = int(detection[3] )
                

                

                # Rectangle coordinates
                x = int(center_x - (w / 4))
                y = int(center_y - (h / 4))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    if indexes == 0: print("weapon detected in frame")
    #if indexes ==0:  
     #os.system(sound) 
     #time.sleep(10)
    #io.imshow(img)
    

    #break
    
    font = cv2.FONT_HERSHEY_PLAIN
  
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (w , h), color, 2)

           # cv2.rectangle(img, (x+w, y+h), (x , y), color, 2)
           # cv2.rectangle(img, (x, y), (w, h ), color, 2)

            cv2.putText(img, label, (x, y + 30), font, 5, color, 5)

    frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    
    #time.sleep(1/(fps*2))

    cv2_imshow(img)
    #plt.pause(1/fps)


    #cv2.imshow("Output", img)
    #img = img +10000
    #io.imshow(img)

    #if indexes ==0:  os.system(sound)
    #key = cv2.waitKey(1)
    #if key == 27:
     #   break
    if cv2.waitKey(100) & 0xFF == ord('q'):
            break

#io.imshow(img)
cap.release()
cv2.destroyAllWindows()












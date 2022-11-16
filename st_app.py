import streamlit as st
import cv2 as cv
import tempfile
import streamlit as st
import numpy as np
import os
import torch
import time
import sys
from skimage import io
import matplotlib.pyplot as plt
from IPython.display import clear_output
import datetime
import imutils
#from google.colab.patches import cv2_imshow
import tempfile

sound = "/content/Police.mp3"
net = cv.dnn.readNet("yolov3_training_2000.weights","yolov3_testing.cfg")

classes = ["Weapon","Knife"]

layer_names = net.getLayerNames()
#output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = net.getUnconnectedOutLayersNames()  
colors = np.random.uniform(0, 255, size=(len(classes), 3))




def run(vf,stframe):
  while 1:
      ret, frame = vf.read()
      # if frame is read correctly ret is True
      if ret is None:
          print("Can't receive frame (stream end?). Exiting ...")
          break

      height, width, channels = frame.shape
      blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
      #blob = cv.dnn.blobFromImage(frame, 1, (416, 416), (0, 0, 0), True, crop=False)

      net.setInput(blob)
      outs = net.forward(output_layers)

      cv.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                      (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

      class_ids = []
      confidences = []
      boxes = []  
      for out in outs:
          for detection in out:
              detection = np.nan_to_num(detection, copy=True, posinf=0, neginf=0)

              scores = detection[5:]
              class_id = np.argmax(scores)
              confidence = scores[class_id]
              if confidence > 0.5:
                  center_x = int(detection[0] * width)
                  center_y = int(detection[1] * height)
                  w = int(detection[2] * width)
                  h = int(detection[3] * height)
      
                  # Rectangle coordinates
                  x = int(center_x - w / 2)
                  y = int(center_y - h / 2)
      
                  boxes.append([x, y, w, h])
                  confidences.append(float(confidence))
                  class_ids.append(class_id)

      indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      stframe.image(frame)
      
              # text
      text = 'Weapon is Detected in Video'
          
        # font
      font = cv.FONT_HERSHEY_SIMPLEX
          
        # org
      org = (00, 185)
          
        # fontScale
      fontScale = 1
           
      # Red color in BGR
      color = (0, 0, 255)
          
        # Line thickness of 2 px
      thickness = 2
           
        # Using cv2.putText() method
      #image = cv2.putText(image, text, org, font, fontScale, 
       #                  color, thickness, cv2.LINE_AA, False)

      
      print(indexes)
      if len(indexes) > 0: st.text("Weapon is detected in video")

      
      if cv.waitKey(100) & 0xFF == ord('q'):
              break

def main():
  #st.title("Weapon Detection System ")
  st.markdown("<h1 style='text-align: center; color: Blue;'>Weapon Detection System</h1>", unsafe_allow_html=True)

  st.image('Sunera.jpg')


  f = st.file_uploader("Upload file")

  tfile = tempfile.NamedTemporaryFile(delete=False)
  try:
      if tfile:   
        tfile.write(f.read())
  except:
      pass

  vf = cv.VideoCapture(tfile.name)
  stframe = st.empty()
  if st.button("Run"):
    run(vf,stframe)

if __name__ == '__main__':
  main()      
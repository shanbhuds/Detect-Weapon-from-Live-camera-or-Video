import numpy as np
import pickle
from keras.models import load_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow 
import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import streamlit as st
#from keras.utils.np_utils import to_categoricalr
import h5py
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from PIL import ImageTk, Image
from keras.models import load_model
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import imputer.transform()


classes = { 1:'Cheating in Examination',
            2:'Eating',
            3:'Indiscip',
            4:'Knife',
            5:'Long_gun',
            6:'Playing',
            7:'Proper sitting in classroom',
            8:'Short_gun',
           }

# loading the saved model
#loaded_model = pickle.load(open('C:/Users/103077/Downloads/Working/Weapon-Detection-with-yolov3-master/weapon_detection/students_activities.h5', 'rb'))

sample_data = []
sample_labels = []

model = load_model('students_activities _new.h5')

#image = Image.open('imagefile')

#displaying the image on streamlit app
#st.image(image, caption='Enter any caption here')
image = Image.open("C:/Users/103077/Downloads/Working/Weapon-Detection-with-yolov3-master/weapon_detection/knief.jpg")

#image = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
sample_im = np.resize(image,(32,32))
sample_im = np.asarray(sample_im).astype(np.float32)

sample_im = np.array(sample_im)
sample_data.append(sample_im)
sample_data = np.array(sample_data)
sample_data = tensorflow.expand_dims(sample_data, axis=-1)
pred = model.predict(sample_data)
maxindex = pred.argmax()
sign = classes[maxindex+1]
st.write("This is a  =>",sign)




#image = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

#image = Image.open("C:/Users/103077/Downloads/Working/Weapon-Detection-with-yolov3-master/weapon_detection/knief.jpg")

#sample_im = Image.open(image)

def load_image(image_file):
    try:
        img = Image.open(image_file)
    except:
        pass
    #return img




def img_processing(input_image):

    classes = { 1:'Cheating in Examination',
                2:'Eating',
                3:'Indiscip',
                4:'Knife',
                5:'Long_gun',
                6:'Playing',
                7:'Proper sitting in classroom',
                8:'Short_gun',
               }
    
               #0-cE,1-e,2-I,3-K,4-lg,5-P,6-pS,7,sg
    sample_data = []
    sample_labels = []
    #classes = 43
    #sample_im = Image.open(input_image)
    
    image = input_image 

    sample_im = np.resize(image,(32,32))
    sample_im = np.asarray(sample_im).astype(np.float32)

    sample_im = np.array(sample_im)
    sample_data.append(sample_im)
    sample_data = np.array(sample_data)
    sample_data = tensorflow.expand_dims(sample_data, axis=-1)
    pred = model.predict(sample_data)
    maxindex = pred.argmax()
    sign = classes[maxindex+1]
    st.write("This is a  =>",sign)




def main():
    
    
    # giving a title
    st.title('Student Activities Monitor System')
    
    
    # getting the input data from the user
    
    
    # code for Prediction
    Action = ''
    
    # creating a button for Prediction
    
    if st.button('Activity Detection'):
       Action = img_processing(image)
        
        
    st.success(Action)
    #st.Image.open(image,width=550)
   # st.image(load_image(image),width=250)
    st.image(load_image(image),width=250)



    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    



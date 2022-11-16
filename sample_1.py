#importing streamlit library

import streamlit as st



#displaying a local video file
#uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])


#video_file = open(uploaded_video, 'rb') #enter the filename with filepath

#video_bytes = video_file.read() #reading the file

#st.video(video_bytes) #displaying the video



#displaying a video by simply passing a Youtube link

st.video("https://youtu.be/yVV_t_Tewvs")


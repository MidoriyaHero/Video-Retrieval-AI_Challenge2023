import torch
import numpy
import torch
import clip
from PIL import Image
import numpy as np
import glob
from scipy import spatial
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json
import streamlit as st
import pandas as pd
from Transnet_model import Transnet_model
from CLIP_model import Clip_model


#header
st.header('Video Retrieval IU-LAB-513')

des_path = st.sidebar.text_input('Folder\'s path to save keyframe: ', "/data/keyframe")
video_paths = st.sidebar.text_input('Folder\'s path of raw video files: ', ('/data/video/*.mp4'))
video_paths= sorted(glob.glob(video_paths))

@st.cache_data
def create_keyframe(video_paths,des_path):
    for video_path in video_paths:
        video_preprocess = Transnet_model(video_path,des_path)
        video_preprocess.process_video()
create_keyframe(video_paths,des_path)


model = Clip_model(des_path)
search_term = st.text_input('Search (recommend English): ')


#features app settings
st.sidebar.header('Settings')
top_number = st.sidebar.slider('Number of Search Results', min_value=20, max_value=100)
picture_width = st.sidebar.slider('Picture Width', min_value=200, max_value=500)
option = st.sidebar.selectbox('Choose language',('English', 'Tiếng ziệt'))


#xét xem ngôn ngữ nào rồi encode input text
if option == 'Tiếng ziệt':
    st.sidebar.warning('GG trans hết được dùng free rồi, tự dịch qua tiếng anh đi!', icon="⚠️")
text_emb = model.encoded(search_term)


@st.cache_data
def encode_frame():
    describe_feature = model.encode_video_frame(des_path)
    return describe_feature
describe_features = encode_frame()


@st.cache_data
def preprocess_frame():
    dict_frame = model.process_frame()
    return dict_frame
dict_frame = preprocess_frame()


path_imgs, info_imgs = model.fit(text_emb,top_number,describe_features,dict_frame)
df_rank = pd.DataFrame(columns=['image_path','img_info'])

for i in range(len(path_imgs)):
    df_rank = pd.concat([df_rank,pd.DataFrame(data=[[path_imgs[i],info_imgs[i]]],columns=['image_path','img_info'])])
df_rank.reset_index(inplace=True,drop=True)

#Tạo button để lưu thành file csv
df_1 = pd.DataFrame(data = info_imgs)
df_1.reset_index(drop=True)
data = df_1.to_csv(header= False, index=False)
st.sidebar.download_button(label = 'Export CSV file', data = data, file_name = 'query_output.csv')


# display code: 3 column view
col1, col2, col3 = st.columns(3)

df_result = df_rank.head(top_number)

for i in range(top_number):
    img = Image.open(df_result.loc[i,'image_path'])
    if i % 3 == 0:
        with col1:
            st.image(img,caption=str(df_result.loc[i,'img_info']),width=picture_width)
    elif i % 3 == 1:
        with col2:
            st.image(img,caption=str(df_result.loc[i,'img_info']),width=picture_width)
    elif i % 3 == 2:
        with col3:
            st.image(img,caption=str(df_result.loc[i,'img_info']),width=picture_width)
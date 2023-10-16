import torch
import numpy
import clip
from PIL import Image
import numpy as np
import glob
import math
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json
import streamlit as st
import pandas as pd
from transnetv2 import TransNetV2
import tensorflow as tf
import cv2
import csv


class Clip_model:

    def __init__(self,des_path):
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.des_path = des_path

    def encoded(self, search_term):
        return self.model.encode([search_term])

    @staticmethod
    def __get_path_frame(dict1,value):
        return dict1[str(value)][1]
    @staticmethod
    def __get_frame_info(dict1,value):
        key = dict1[str(value)][0]
        temp = key.split('-')
        return [temp[-2], (temp[-1])]
    
    def process_frame(self):
        dict_frame ={}
        image_paths = glob.glob(self.des_path+"/*/*.jpg")
        image_paths = sorted(image_paths)

        for index,img_path in enumerate(image_paths):
        #tạo ra dict với key có dạng : keyframes_L0X-L0X_V00X-XXXX VÀ value có là số thứ tự từ 0 đến số lượng key frame trong file video
            tempo = img_path.split("/")
            temp1 = tempo[-2] +'-'+tempo[-1].replace('.jpg','')
            dict_frame[str(index)] = [temp1,img_path]
        return dict_frame
    #You can save this dictionary to json file to re-use latter
    def encode_video_frame(self,des_path):
        describe =[]
        image_paths = glob.glob(des_path+"/*/*.jpg")
        for i in image_paths:
            img_emb = self.model.encode(Image.open(i))
            describe.append(img_emb)
        return describe
    def fit(self,text_emb,k,describe,dict_frame):
        self.id_list = []
        self.score_list = []
        cos_scores = util.semantic_search(torch.tensor(text_emb), torch.tensor(describe,dtype=torch.float32),top_k = k)
        list_of_dictionaries = cos_scores[0]
        for item in list_of_dictionaries:
            self.id_list.append(item['corpus_id'])
            self.score_list.append(item['score'])

        self.path_imgs = []
        self.info_imgs =[]

        for i in self.id_list:
            self.path_imgs.append(self.__get_path_frame(dict_frame,i))
            self.info_imgs.append(self.__get_frame_info(dict_frame,i))

        return self.path_imgs, self.info_imgs

        
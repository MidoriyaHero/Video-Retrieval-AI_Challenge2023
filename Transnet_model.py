import ffmpeg
import numpy as np
import glob
import os
import cv2
import csv
from transnetv2 import TransNetV2

#format of path should be like this
#video_paths = sorted(glob.glob('*\\data\\video_test.mp4'))
#des_path = '*\\data\\key_frame'

class Transnet_model:

    def __init__(self,video_paths,des_path):
        self.model = TransNetV2()
        self.video_path = sorted(glob.glob((video_paths)))
        self.des_path = des_path

    def get_frames(self,fn, width=48, height=27):
        self.video_stream, err = (
        ffmpeg
        .input(fn)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .run(capture_stdout=True, capture_stderr=True)
        )
        self.video = np.frombuffer(self.video_stream, np.uint8).reshape([-1, height, width, 3])
        return self.video
    
    def process_video(self):
        folder_name = os.path.basename(self.video_path).replace('.mp4', '')
        folder_path = os.path.join(self.des_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        video = Transnet_model.get_frames(self.video_path, width=48, height=27)
        single_frame_predictions, _ = self.model.predict_frames(video)
        scenes = self.model.predictions_to_scenes(single_frame_predictions)

        cam = cv2.VideoCapture(self.video_path)
        currentframe = 0
        index = 0

        while True:
                ret, frame = cam.read()
                if ret:
                    currentframe += 1
                    idx_first = int(scenes[index][0])
                    idx_end = int(scenes[index][1])
                    idx_025 = int(scenes[index][0] + (scenes[index][1]-scenes[index][0])/4)
                    idx_05 = int(scenes[index][0] + (scenes[index][1]-scenes[index][0])/2)
                    idx_075 = int(scenes[index][0] + 3*(scenes[index][1]-scenes[index][0])/4)

                # Sử dụng chỉ số "n" từ last_n + 1 khi ghi dữ liệu vào file CSV
                    if currentframe - 1 == idx_first:
                        filename_first = "{}/{:0>4d}.jpg".format(folder_path, idx_first)
                    # video_save = cv2.resize(video[idx_first], (1280,720))
                        cv2.imwrite(filename_first, frame)
                    if currentframe - 1 == idx_025:
                        filename_025 = "{}/{:0>4d}.jpg".format(folder_path, idx_025)
                    # video_save = cv2.resize(video[idx_025], (1280,720))
                        cv2.imwrite(filename_025, frame)

                # #### 05 ####
                    if currentframe - 1 == idx_05:
                        filename_05 = "{}/{:0>4d}.jpg".format(folder_path, idx_05)
                    # video_save = cv2.resize(video[idx_05], (1280,720))
                        cv2.imwrite(filename_05, frame)
                # #### 075 ####
                    if currentframe - 1 == idx_075:
                        filename_075 = "{}/{:0>4d}.jpg".format(folder_path, idx_075)
                    # video_save = cv2.resize(video[idx_075], (1280,720))
                        cv2.imwrite(filename_075, frame)
                    if currentframe - 1 == idx_end:
                        filename_end = "{}/{:0>4d}.jpg".format(folder_path, idx_end)
                    # video_save = cv2.resize(video[idx_end], (1280,720))
                        cv2.imwrite(filename_end, frame)
                        index += 1
                else:
                    break

        cam.release()
        cv2.destroyAllWindows()
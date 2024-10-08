import copy
import numpy as np
import cv2
from glob import glob
import os
import argparse
import json

# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class FFProbeResult(NamedTuple): 
    return_code: int
    json: str
    error: str
#定義一個具名元組(NameTuple)，名為FFProbeResult
#FFProbe是ffmpeg裡面的其中一個功能，可以用來提取影音文件的原始數據
'''
#ffprobe 的功能：
顯示文件的格式信息：例如文件的大小、比特率、時長等。
顯示文件中的流信息：可以檢查視頻流和音頻流的編碼格式、分辨率、幀率、聲道數、語言等。
提取特定數據：可以提取例如字幕、封面圖像等元數據。
自定義輸出格式：可以選擇不同的輸出格式，如 JSON、XML 或普通文本，方便進一步的數據處理。
'''
#因此上面的類別是在定義ffprobe的輸出內容格式

def ffprobe(file_path) -> FFProbeResult: #def ffprobe(file_path) ， file_path是輸入參數 ， ->xx 表示該函數返回一個 xx， 格式就是上面定義的那樣
    command_array = ["ffprobe",                   #一個陣列，方法是"ffprobe"，ffprobe的使用方法 ffprobe[選項][文件]
                     "-v", "quiet",               #-v 操作流程訊息， 關閉
                     "-print_format", "json",     #列印格式，JSON
                     "-show_format",                #輸出
                     "-show_streams",               #輸出影片數據、音效數據等"Streams"
                     file_path]                     #路徑
    result = subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    #subprocess.run ，python執行某種外部命令的函數(GPT講的)
    #command_array ，執行方式，是上面定義的那樣，ffprobe(方法)、[選項][路徑]
    return FFProbeResult(return_code=result.returncode,
                         json=result.stdout,
                         error=result.stderr)


# openpose setup
from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def process_frame(frame, body=True, hands=True):
    canvas = copy.deepcopy(frame)
    if body:
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if hands:
        hands_list = util.handDetect(candidate, subset, frame)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(frame[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)
        canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas

# writing video with ffmpeg because cv2 writer failed
# https://stackoverflow.com/questions/61036822/opencv-videowriter-produces-cant-find-starting-number-error
import ffmpeg

# open specified video
parser = argparse.ArgumentParser(
        description="Process a video annotating poses detected.")
parser.add_argument('file', type=str, help='Video file location to process.')
parser.add_argument('--no_hands', action='store_true', help='No hand pose')
parser.add_argument('--no_body', action='store_true', help='No body pose')
args = parser.parse_args()
video_file = args.file
cap = cv2.VideoCapture(video_file)

# get video file info
ffprobe_result = ffprobe(args.file)
info = json.loads(ffprobe_result.json)
videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
input_fps = videoinfo["avg_frame_rate"]
# input_fps = float(input_fps[0])/float(input_fps[1])
input_pix_fmt = videoinfo["pix_fmt"]
input_vcodec = videoinfo["codec_name"]

# define a writer object to write to a movidified file
postfix = info["format"]["format_name"].split(",")[0]
output_file = ".".join(video_file.split(".")[:-1])+".processed." + postfix


class Writer():
    def __init__(self, output_file, input_fps, input_framesize, input_pix_fmt,
                 input_vcodec):
        if os.path.exists(output_file):
            os.remove(output_file)
        self.ff_proc = (
            ffmpeg
            .input('pipe:',
                   format='rawvideo',
                   pix_fmt="bgr24",
                   s='%sx%s'%(input_framesize[1],input_framesize[0]),
                   r=input_fps)
            .output(output_file, pix_fmt=input_pix_fmt, vcodec=input_vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def __call__(self, frame):
        self.ff_proc.stdin.write(frame.tobytes())

    def close(self):
        self.ff_proc.stdin.close()
        self.ff_proc.wait()


writer = None
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break

    posed_frame = process_frame(frame, body=not args.no_body,
                                       hands=not args.no_hands)

    if writer is None:
        input_framesize = posed_frame.shape[:2]
        writer = Writer(output_file, input_fps, input_framesize, input_pix_fmt,
                        input_vcodec)

    cv2.imshow('frame', posed_frame)

    # write the frame
    writer(posed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.close()
cv2.destroyAllWindows()



#ffmpeg -i IMG_right.mp4 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" output.mp4 酷
#==>因為不是2的倍數

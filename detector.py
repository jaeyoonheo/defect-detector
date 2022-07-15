"""
다크넷 검출기를 이용하여 입력된 이미지에서
모든 차량 객체를 검출한다.
"""


from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

from tracker import *
from utils import *


class Detector():
    """
    객체를 검출하기 위한 클래스
    """
    def __init__(self):
        '''
        객체 검출을 위한 멤버 변수 선언
        '''
        self.netMain = None
        self.metaMain = None
        self.altNames = None

        self.darknet_width = None
        self.darknet_height = None
        self.darknet_image = None

        self.scale_width = None
        self.scale_height = None
        
        self.frame_num = 0

    def convertBack(self, x, y, w, h):
        '''
        중심점가 객체의 크기를 이용하여 좌상단 좌표와
        우하단 좌표 계산
        '''
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax


    def cvDrawBoxes(self, detections, img, thresh=0.5):
        '''
        검출 결과와 이미지를 입력받아
        입력받은 이미지에 결과를 표시하여 반환
        '''
        for detection in detections:
            if detection[1] < thresh:
                continue
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            print(detection[0])
            cv2.putText(img,
                        str(detection[0]),
                        #detection[0].decode() +
                        #" [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
        return img


    def convertScale(self, detections):
        '''
        검출 결과를 입력받아
        yolo 형태의 결과를 이미지 형태의 결과로 변환하여 반환
        '''
        new_detections = []
        for detection in detections:
            c = detection[0]
            s = detection[1]
            x = detection[2][0] * self.scale_width
            y = detection[2][1] * self.scale_height
            w = detection[2][2] * self.scale_width
            h = detection[2][3] * self.scale_height
            new_detections.append((c, s, (x, y, w, h)))
        return new_detections


    def drawResults(self, writer, tracks, frame_num):
        '''
        파일 기술자와 검출 정보를 입력받아
        파일에 작성하는 함수
        '''
        for track in tracks:
            x, y, w, h = track.bboxes[-1].x,\
                track.bboxes[-1].y,\
                track.bboxes[-1].width,\
                track.bboxes[-1].height
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)

            writer.write('%d,%d,%d,%d,%d,%d\n' % (frame_num, track.id, 
                                                  pt1[0], pt1[1], pt2[0], 
                                                  pt2[1]))


    def initialize(self, configPath, weightPath, metaPath):
        '''
        검출기를 초기화 하는 함수
        '''
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath)+"`")

        if self.netMain is None:
            self.netMain = darknet.load_net_custom(configPath.encode(
                # batch size = 1
                "ascii"), weightPath.encode("ascii"), 0, 1)  
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except TypeError:
                pass
        
        # Create an image we reuse for each detect
        self.darknet_width = darknet.network_width(self.netMain)
        self.darknet_height = darknet.network_height(self.netMain)
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                        darknet.network_height(self.netMain), 3)


    def getDetectionImage(self, image):
        '''
        검출을 위해 크기 조정된 이미지를 반환하는 함수
        '''
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                    (darknet.network_width(self.netMain),
                                    darknet.network_height(self.netMain)),
                                    interpolation=cv2.INTER_LINEAR)
        return frame_resized


    def detector(self, image, thresh=0.25):
        '''
        이미지와 임계값을 전달 받아 검출하는 함수
        입력 :
            image : numpy BGR 이미지
            thresh : 검출 신뢰도 임계값(0.0~1.0)
        출력 : 
            detections : 검출 결과
                         [('class name', confidence, (cx, cy, w, h)), ...]
        '''
        #prev_time = time.time()
        frame_read = image
        height, width, _ = frame_read.shape
        self.scale_width = float(width) / float(self.darknet_width)
        self.scale_height = float(height) / float(self.darknet_height)

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                (self.darknet_width,
                                    self.darknet_height),
                                interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh)

        detections = self.convertScale(detections)

        #fps = 1/(time.time()-prev_time)

        #return detections, fps
        return detections
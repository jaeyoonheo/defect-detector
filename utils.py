"""
차량 검출, 추적 및 계수에 필요한 도구를 제공한다.
"""


import os
import enum

import cv2
import numpy as np


class TRACK_END_STATE(enum.Enum):
    """
    추적의 종료 조건
    """
    MERGE = 1
    TIME_OUT = 2
    DETECT_SHORT = 3
    ROI_OUT = 4
    NOT_VEHICLE = 5


class Rect:
    """
    사각형 영역 정보를 저장하기 위한 클래스
    """
    def __init__(self, _x, _y, _width, _height):
        '''
        x, y, width, height를 입력 받아 사각형 정보 생성
        '''
        self.x = _x
        self.y = _y
        self.width = _width
        self.height = _height

    def area(self):
        '''
        입력된 사각형의 넓이 정보 반환
        '''
        return self.width * self.height
    
    def intersection(self, rect):
        '''
        현재 사각형과 입력된 사각형의 교집합 영역을 반환
        '''
        x = max(self.x, rect.x)
        y = max(self.y, rect.y)
        w = min(self.x+self.width, rect.x+rect.width) - x
        h = min(self.y+self.height, rect.y+rect.height) - y
        if w < 0 or h < 0:
            return Rect(0, 0, 0, 0)
        else:
            return Rect(x, y, w, h)
            
    def union(self, rect):
        '''
        현재 사각형과 입력된 사각형의 합집합 영역을 반환
        '''
        x = min(self.x, rect.x)
        y = min(self.y, rect.y)
        w = max(self.x+self.width, rect.x+rect.width) - x
        h = max(self.y+self.height, rect.y+rect.height) - y
        if w < 0 or h < 0:
            return Rect(0, 0, 0, 0)
        else:
            return Rect(x, y, w, h)  
        
    def printable(self):
        '''
        현재 입력된 사각형의 정보 출력
        '''
        if type(self.x) is int:
            return([self.x, self.y, self.width, self.height])
        else:
            return([round(self.x, 2), round(self.y, 2), round(self.width, 2), 
                    round(self.height, 2)])
    
    def printable_int(self):
        '''
        현재 입력된 사각형의 정보 출력
        '''
        return([int(round(self.x, 0)), 
                int(round(self.y, 0)), 
                int(round(self.width, 0)), 
                int(round(self.height, 0))])


class Point:
    """
    좌표 정보를 저장하기 위한 클래스
    """
    def __init__(self, _x, _y):
        '''
        x와 y 좌표를 입력 받아 저장
        '''
        self.x = _x
        self.y = _y


class Size:
    """
    크기 정보를 저장하기 위한 클래스
    """
    def __init__(self, _width, _height):
        '''
        width와 height 정보를 입력 받아 저장
        '''
        self.width = _width
        self.height = _height


class Pair:
    """
    좌표 쌍을 저장하기 위한 클래스
    """
    def __init__(self):
        '''
        초기 멤버 설정
        '''
        self.first_type = None
        self.second_type = None
    
    def first(self):
        '''
        first type에 대한 정보 반환
        '''
        return self.first_type

    def second(self):
        '''
        second type에 대한 정보 반환
        '''
        return self.second_type


def convertBack(x, y, w, h):
    '''
    x, y, width, height 정보를 입력받아 좌상단 및 우하단 좌표 계산
    입력된 x, y는 중심점 의미
    '''
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    '''
    검출 결과와 이미지를 입력 받아
    입력받은 이미지에 검출결과를 표시하여 반환
    '''
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        #cv2.putText(img, detection[0].decode() + " [" + 
        #            str(round(detection[1] * 100, 2)) + "]", 
        #            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 
        #            0.5, [0, 255, 0], 1)
    return img


def cvDrawTracks(tracks, img):
    '''
    추적 결과와 이미지를 입력 받아
    입력받은 이미지에 추적 결과를 표시하여 반환
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
        cv2.rectangle(img, pt1, pt2, track.color, 1)
        cv2.putText(img, '%d : %.2f' % (track.id, 
                                        track.detection_confidences[-1]), 
                    (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    track.color, 1)

        circle_end = len(track.bboxes)
        if circle_end < 30:
            circle_start = 0
        else:
            circle_start = len(track.bboxes) - 30
        for circle_idx in range(circle_start, circle_end):
            x, y, w, h = track.bboxes[circle_idx].x,\
                track.bboxes[circle_idx].y,\
                track.bboxes[circle_idx].width,\
                track.bboxes[circle_idx].height
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            cv2.circle(
                img, 
                (xmin + int((xmax - xmin) / 2), 
                 ymin + int((ymax - ymin) / 2)), 
                 1, track.color, 2)
        #cv2.putText(img, detection[0].decode() + " [" + 
        #            str(round(detection[1] * 100, 2)) + "]", 
        #            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
        #            [0, 255, 0], 1)
    return img


def cvDrawCounters(counters, img):
    '''
    카운팅 결과와 이미지를 입력 받아
    입력받은 이미지에 카운팅 결과를 표시하여 반환
    '''
    for counter in counters:
        x, y, w, h = counter.bboxes[-1].x,\
            counter.bboxes[-1].y,\
            counter.bboxes[-1].width,\
            counter.bboxes[-1].height
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, counter.color, 1)
        cv2.putText(img, '%d' % (counter.id), (xmin, ymin - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, counter.color, 1)

        circle_end = len(counter.bboxes)
        if circle_end < 30:
            circle_start = 0
        else:
            circle_start = len(counter.bboxes) - 30
        for circle_idx in range(circle_start, circle_end):
            x, y, w, h = counter.bboxes[circle_idx].x,\
                counter.bboxes[circle_idx].y,\
                counter.bboxes[circle_idx].width,\
                counter.bboxes[circle_idx].height
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            cv2.circle(img, (xmin + int((xmax - xmin) / 2), 
                             ymin + int((ymax - ymin) / 2)), 1,
                      counter.color, 2)
        #cv2.putText(img, detection[0].decode() + " [" + 
        #            str(round(detection[1] * 100, 2)) + "]", 
        #            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 
        #            0.5, [0, 255, 0], 1)
    return img


def drawDetectionResults(writer, detections, frame_num):
    '''
    파일 discriptor, 추적 결과, 현재 프레임 번호를 입력받아
    해당 파일에 추적 결과와 프레임 번호를 저장
    '''
    for detection in detections:
        x, y, w, h = detection[2][0], \
            detection[2][1], \
            detection[2][2], \
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        pt2 = (xmax - xmin, ymax - ymin)

        writer.write('%05d,car,%d,%d,%d,%d\n' % 
                     (frame_num, pt1[0], pt1[1], pt2[0], pt2[1]))


def drawTrackResults(writer, tracks, frame_num):
    '''
    파일 discriptor, 추적 결과, 현재 프레임 번호를 입력받아
    해당 파일에 추적 결과와 프레임 번호를 저장
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
        pt2 = (xmax - xmin, ymax - ymin)

        writer.write('%05d,%d,car,%d,%d,%d,%d\n' % 
                     (frame_num, track.id, pt1[0], pt1[1], pt2[0], pt2[1]))


def drawCounterResults(writer, counts, frame_num):
    '''
    파일 discriptor, 카운팅 결과, 현재 프레임 번호를 입력받아
    해당 파일에 카운팅 결과와 프레임 번호를 저장
    '''
    for count in counts:
        x, y, w, h = count.bboxes[-1].x,\
            count.bboxes[-1].y,\
            count.bboxes[-1].width,\
            count.bboxes[-1].height
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        pt2 = (xmax - xmin, ymax - ymin)

        writer.write('%05d,%d,car,%d,%d,%d,%d\n' % 
                     (frame_num, count.id, pt1[0], pt1[1], pt2[0], pt2[1]))


def copyObject(src, dst, box):
    '''
    원본 이미지, 결과 이미지, 검출 박스를 입력받아
    원본 이미지의 검출된 객체를 결과 이미지에 복사
    '''
    box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    src[box[0]:box[0]+box[2], box[1]:box[1]+box[3]] = \
        dst[box[0]:box[0]+box[2], box[1]:box[1]+box[3]]
    return dst


def copyObjects(src, dst, detections, thresh=0.5, pad=0):
    '''
    원본 이미지, 결과 이미지, 검출된 모든 결과, 임계값, 패딩 옵션을 입력받아
    검출 신뢰도가 임계값 이상인 객체를 원본 이미지에서 결과 이미지로 복사
    '''
    for det in detections:
        if det[1] < thresh:
            continue
        x1 = int(det[2][0] - ((det[2][2] + pad) / 2))
        if x1<0:
            x1=0
        x2 = int(det[2][0] + ((det[2][2] + pad) / 2))
        y1 = int(det[2][1] - ((det[2][3] + pad) / 2))
        if y1<0:
            y1=0
        y2 = int(det[2][1] + ((det[2][3] + pad) / 2))
        dst[y1:y2, x1:x2] = \
            src[y1:y2, x1:x2]
    return dst


def writeAnnotations(file_name, detections, width, height, thresh=0.5):
    '''
    파일명, 모든 검출결과, 이미지의 높이와 너비, 임계값을 입력받아
    임계값 이상인 검출 결과를 파일에 저장
    '''
    #path = os.path.join(anno_dir, file_name.split('.')[0] + '.txt')
    path = file_name
    f = open(path, 'w')
    for det in detections:
        print(det[0])
        if det[1] < thresh:
            continue
        w = det[2][2] / width
        h = det[2][3] / height
        x = det[2][0] / width
        y = det[2][1] / height
        f.write("%s %f %f %f %f\n" % (det[0].decode(), x, y, w, h))
    f.close()

def writeAnnotations_(file_name, detections, width, height, thresh=0.5):
    '''
    파일명, 모든 검출결과, 이미지의 높이와 너비, 임계값을 입력받아
    임계값 이상인 검출 결과를 파일에 저장
    '''
    #path = os.path.join(anno_dir, file_name.split('.')[0] + '.txt')
    path = file_name
    f = open(path, 'w')
    for det in detections:
        print(det[0])
        if det[1] < thresh:
            continue
        w = det[2][2] / width
        h = det[2][3] / height
        x = det[2][0] / width
        y = det[2][1] / height
        f.write("%s %f %f %f %f %f\n" % (det[0].decode(), det[1], x, y, w, h))
    f.close()
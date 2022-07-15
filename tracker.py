"""
검출 결과를 이용하여 다중 객체를 추적한다.
추적은 IOU를 기반으로 하며 미검출 상황을 고려하여
추가 정보를 보완하는 알고리즘을 사용한다.
"""


import os
import random
import itertools
import math

import cv2
import numpy as np

import utils

import time


class DetectionInfo:
    """
    검출 결과를 저장하기 위한 클래스
    """
    def __init__(self, frame_number, bbox, center, confidence):
        '''
        프레임 번호, 검출 박스, 중심점, 신뢰도를 입력받아 검출 정보 생성
        '''
        # detection frame_number
        self.frame_number = frame_number

        # detection bbox
        self.bbox = bbox

        # detection center
        self.center = center

        # detection confidence
        self.confidence = confidence


class AdditionalInfo:
    """
    검출 실패시 지속적인 추적을 위해 생성하는 추가 정보
    """
    def __init__(self, bbox, center, distance, 
                 speed, scale, direction, confidence):
        '''
        박스 영역, 중심점, 거리, 속도, 크기 변화율, 방향, 신뢰도를 입력받아
        추가 정보 생성
        '''
        self.bbox = bbox
        self.center = center
        self.distance = distance
        self.speed = speed
        self.scale = scale
        self.direction = direction
        self.tracking_confidence = confidence


class TrackerInfo:
    """
    추적 정보를 저장하기 위한 클래스
    """
    def __init__(self):
        '''
        객체 추적 정보를 저장하기 위한 멤버 변수 선언
        '''
        # object id as integer num
        self.id = None

        # tracking flag
        self.flag = False

        # tracking information
        self.tracked_frames = []
        self.tracked_frames_count = 0
        self.continuous_tracking_count = 0

        # detection information
        self.detected_frames = []
        self.detected_frames_count = 0
        self.continuous_detection_count = 0

        # undetected information
        self.undetected_frames = []
        self.undetected_frames_count = 0
        self.continuous_undetection_count = 0

        # tracked bboxes
        self.bboxes = []
        self.centers = []

        # drawing box color as (b, g, r)
        self.color = None

        # total confidence
        self.detection_confidences = []
        self.track_confidences = []

        # euclidean distance between first box and last box
        self.distances = []

        # if current frame is n, distance is between n-1 and n-2
        self.speeds = []

        # if current frame is n, scale is between n-1 and n-2
        self.scales = []

        # direction
        self.directions = []

    def initialize(self, track_id, frame_number, color, detection_info):
        '''
        새로운 객체 발생시 추적 정보를 새로 생성하는 함수
        '''
        self.id = track_id
        self.tracked_frames.append(frame_number)
        self.tracked_frames_count += 1

        self.detected_frames.append(detection_info.frame_number)
        self.detected_frames_count += 1
        self.bboxes.append(detection_info.bbox)
        self.centers.append(detection_info.center)
        self.detection_confidences.append(detection_info.confidence)
        self.track_confidences.append(0.5)

        self.color = color

        self.continuous_tracking_count += 1

        self.distances.append(0)
        self.speeds.append(0)
        self.scales.append(0)
        self.directions.append(0)

    def update(self, frame_number, additional_info, detection_info=None):
        '''
        개체가 지속적으로 추적될 시 추적 정보를 업데이트 하는 함수
        '''
        self.tracked_frames.append(frame_number)
        self.tracked_frames_count += 1
        self.track_confidences.append(additional_info.tracking_confidence)
        self.continuous_tracking_count += 1

        self.distances.append(additional_info.distance)
        self.speeds.append(additional_info.speed)
        self.scales.append(additional_info.scale)
        self.directions.append(additional_info.direction)
        
        if detection_info != None:
            self.detected_frames.append(detection_info.frame_number)
            self.detected_frames_count += 1
            self.continuous_detection_count += 1

            self.bboxes.append(detection_info.bbox)
            self.centers.append(detection_info.center)
            self.detection_confidences.append(detection_info.confidence)

            self.continuous_undetection_count = 0
        else:
            self.undetected_frames.append(frame_number)
            self.undetected_frames_count += 1
            self.continuous_undetection_count += 1

            self.bboxes.append(additional_info.bbox)
            self.centers.append(additional_info.center)
            self.detection_confidences.append(0)

            self.continuous_detection_count =- 1

    def remove(self):
        '''
        추적 완료 및 실패시 추적 정보를 소멸하는 함수
        '''
        pass


class Tracker:
    """
    다중 객체를 추적하여 관리하는 추적기 클래스
    """
    def __init__(self):
        '''
        다중 추적을 위해 필요한 멤버 변수 선언
        '''
        self.track_num = 0
        self.track_cnt = 0
        self.frame_num = 0
        self.track_infos = []

        self.track_candidate_num = 0
        self.track_candidate_infos = []
        self.track_removed_infos = []
        self.color_list = list(range(0, 256))
        self.color = random.sample(self.color_list, 3)


    def initialize(self, detection_info):
        '''
        새로운 추적 정보 발생시 새로운 추적 정보를 생성하여 등록
        '''
        track = TrackerInfo()
        track.initialize(-1, self.frame_num, self.color, detection_info)
        track.centers.append(self.calculate_center(track.bboxes[-1]))
        self.color = random.sample(self.color_list, 3)
        self.track_candidate_infos.append(track)
        self.track_candidate_num -= 1


    def upgrade(self, candidate_info):
        '''
        객체가 지속적으로 추적될 경우 해당 추적 정보에 현재 정보 갱신
        '''
        idx = self.track_candidate_infos.index(candidate_info)

        self.track_candidate_infos[idx].id = self.track_num
        self.track_candidate_infos[idx].flag = True
        self.track_infos.append(self.track_candidate_infos[idx])

        self.track_candidate_infos.remove(self.track_candidate_infos[idx])
        self.track_cnt += 1
        self.track_num += 1


    def additionalInfo(self, tracker_info, detection_info=None, confidence=0):
        '''
        검출 실패시 추적을 보완하기 위해 추가 정보를 생성하여 객체 유지
        '''
        if detection_info == None:
            bbox, center, speed, distance, scale, direction = \
                self.compensation_box(tracker_info)
            confidence = 0
        else:
            bbox = detection_info.bbox
            speed = self.speed_estimation(tracker_info, detection_info)
            scale = self.scale_estimation(tracker_info, detection_info)
            direction = \
                self.direction_estimation(tracker_info, detection_info)
            distance = self.distance_estimation(tracker_info, detection_info)
            center = self.calculate_center(detection_info.bbox)
        additional_info = AdditionalInfo(bbox, center, distance, 
                                         speed, scale, direction, confidence)
        return additional_info
        

    def compensation_box(self, tracker_info):
        '''
        검출 실패시 추가 정보를 생성하기 위한 보완 알고리즘
        '''
        bbox = tracker_info.bboxes[-1]
        center = tracker_info.centers[-1]
        speed = tracker_info.speeds[-1]
        distance = tracker_info.distances[-1]
        scale = tracker_info.scales[-1]
        direction = tracker_info.directions[-1]
        
        if tracker_info.tracked_frames_count > 3:
            for i in range(-2, -5, -1):
                speed += tracker_info.speeds[i]
                distance += tracker_info.distances[i]
                scale += tracker_info.scales[i]
                direction += tracker_info.directions[i]

            speed /= 3
            distance /= 3
            scale /= 3
            direction /= 3
        
            #angle = self.radian(direction)

            #center.x = center.x + distance * math.cos(direction)
            #center.y = center.y + distance * math.sin(direction)

            #width = bbox.width * scale
            #height = bbox.height * scale
            width = bbox.width
            height = bbox.height

            x = center.x - width / 2
            y = center.y - height / 2

            bbox.x = int(x)
            bbox.y = int(y)
            bbox.width = int(width)
            bbox.height = int(height)
        return bbox, center, speed, distance, scale, direction


    def update(self, tracker_info, confidence, detection_info=None):
        '''
        추적중인 객체에 대하여 현재 정보를 입력 받아 정보 갱신
        '''
        # 추가 정보 계산
        additional_info = self.additionalInfo(tracker_info, 
                                              detection_info, confidence)
        # 아이디가 0보다 작으면 추적 후보
        if tracker_info.id < 0:
            idx = self.track_candidate_infos.index(tracker_info)
            self.track_candidate_infos[idx].update(self.frame_num, 
                                                   additional_info, 
                                                   detection_info)

            if self.track_candidate_infos[idx].continuous_detection_count > 5:
                self.upgrade(self.track_candidate_infos[idx])
            elif self.track_candidate_infos[idx].\
                continuous_undetection_count > 15:
                self.remove(self.track_candidate_infos[idx])
        # 아이디가 0보다 크면 추적 객체
        else:
            idx = self.track_infos.index(tracker_info)
            self.track_infos[idx].update(self.frame_num, 
                                         additional_info, 
                                         detection_info)

            if self.track_infos[idx].continuous_undetection_count > 15:
                self.remove(self.track_infos[idx])
        

    def remove(self, tracker_info):
        '''
        추적 실패 및 완료 객체에 대한 후처리
        '''
        # 추적 아이디가 0보다 작으면 후보
        if tracker_info.id < 0:
            idx = self.track_candidate_infos.index(tracker_info)
            self.track_candidate_infos.remove(self.track_candidate_infos[idx])
        # 추적 아이디가 0보다 크면 추적 객체
        else:
            idx = self.track_infos.index(tracker_info)
            self.track_removed_infos.append(self.track_infos[idx])
            self.track_infos.remove(self.track_infos[idx])
            self.track_cnt -= 1


    def calculateIOUMap(self, track_infos, detection_infos):
        '''
        추적 객체와 검출된 정보를 비교하여 IOU 행렬 계산
        '''
        # calculate iou between last bbox of track_info and 
        # current bbox of detection_info
        iou_matrix = []
        for info in track_infos:
            iou_list = []
            for detection_info in detection_infos:
                intersection_rect = \
                    info.bboxes[-1].intersection(detection_info.bbox)
                union_rect = info.bboxes[-1].union(detection_info.bbox)
                iou_list.append(intersection_rect.area() / union_rect.area())
            iou_matrix.append(iou_list)
        return np.array(iou_matrix, dtype=float)


    def calculateColorMap(self, detection_infos):
        '''
        컬러 정보를 비교하여 유사도 맵 생성
        '''
        color_matrix = []

        return np.array(color_matrix, dtype=float)

    
    def calculate_center(self, bbox):
        '''
        입력된 박스의 중심점 계산
        '''
        x = bbox.x + bbox.width / 2
        y = bbox.y + bbox.height / 2
        return utils.Point(x, y)

    
    def distance_estimation(self, tracker_info, detection_info):
        '''
        입력된 객체와 검출 결과의 중심 좌표를 이용하여 유사도 계산
        '''
        past_center = tracker_info.centers[-1]
        current_center = detection_info.center
        delta_x = current_center.x - past_center.x
        delta_y = current_center.y - past_center.y
        distance = math.sqrt(pow(abs(delta_x), 2) + pow(abs(delta_y), 2))
        return distance


    def speed_estimation(self, tracker_info, detection_info):
        '''
        입력된 객체와 검출 결과의 중심 좌표를 이용하여 속도 계산
        '''
        past_center = tracker_info.centers[-1]
        current_center = detection_info.center

        speed = math.sqrt(pow(past_center.x - current_center.x, 2) + 
                          pow(past_center.y - current_center.y, 2))

        return speed


    def scale_estimation(self, tracker_info, detection_info):
        '''
        입력된 객체와 검출 결과의 박스 크기를 이용하여 크기 변화율 계산
        '''
        #print(detection_info.frame_number)
        past_box = tracker_info.bboxes[-1]
        current_box = detection_info.bbox
        #print(past_box.x, past_box.y, past_box.width, past_box.height)
        #print(current_box.x, current_box.y, current_box.width, 
        #      current_box.height)
        scale = current_box.area() / past_box.area()
        return scale


    def radian(self, degree):
        '''
        디그리 각도를 라디안 각도로 변환
        '''
        return degree * math.pi / 180

    
    def degree(self, radian):
        '''
        라디안 각도를 디그리 각도로 변환
        '''
        return radian / math.pi * 180


    def direction_estimation(self, tracker_info, detection_info):
        '''
        추적 객체와 검출 결과의 각도 변화율 계산
        '''
        past_center = tracker_info.centers[-1]
        current_center = detection_info.center
        delta_x = current_center.x - past_center.x
        delta_y = current_center.y - past_center.y
        #delta_x = past_center.x - current_center.x
        #delta_y = past_center.y - current_center.y
        radian = math.atan2(delta_y, delta_x)
        #degree = self.degree(radian)
        #if degree  < 0:
        #    degree = 360 + degree
        
        return radian


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
    
    
    def cvDrawBoxes(self, tracks, img):
        '''
        검출 결과와 이미지를 입력받아
        입력받은 이미지에 결과를 표시하여 반환
        '''
        for track in tracks:
            x, y, w, h = track.bboxes[-1].x,\
                track.bboxes[-1].y,\
                track.bboxes[-1].width ,\
                track.bboxes[-1].height
            xmin, ymin, xmax, ymax = self.convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, track.color, 1)

            cv2.putText(img,
                        "%d : %.2f" % (track.id, track.detection_confidences[-1]),
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        track.color, 2)
            print(track.id, pt1[0], pt1[1], pt2[0], pt2[1])
        return img

    
    def getTrackInfos(self):
        '''
        추적 결과를 반환
        [(id, confidence, (cx, cy, w, h))]
        '''
        rst = []
        for track in self.track_infos:
            x, y, w, h = track.bboxes[-1].x,\
                track.bboxes[-1].y,\
                track.bboxes[-1].width ,\
                track.bboxes[-1].height
            rst.append([track.id, track.detection_confidences[-1], (x, y, w, h)])
        return rst


    def convertDetection2Tracking(self, detections, frame_num):
        '''
        검출 결과와 프레임 번호를 받아 추적 형태로 변환
        '''
        detection_infos = []
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]

            rect = utils.Rect(x, y, w, h)
            center = utils.Point(rect.x + rect.width / 2, rect.y + 
                                    rect.height / 2)
            temp_info = DetectionInfo(frame_num, rect, center, 
                                        detection[1])
            detection_infos.append(temp_info)
        return detection_infos


    def tracking(self, detection_infos, frame_num):
        '''
        검출 결과들을 입력 받아 현재 추적 정보들과 비교하여 추적 수행
        '''
        # update current frame_num of tracker
        self.frame_num = frame_num

        # dropout detection info as low detection confidence
        #detection_infos = [detection_info 
        #                   for detection_info in detection_infos 
        #                   if detection_info.confidence > 0.5]
        detection_infos = [detection_info 
                           for detection_info in detection_infos]
        tracker_infos = self.track_infos + self.track_candidate_infos

        detection_idx_list = list(range(len(detection_infos)))
        tracker_idx_list = list(range(len(tracker_infos)))

        # calculate iou
        iou_matrix = self.calculateIOUMap(tracker_infos, detection_infos)

        iou_indexes = []
        if iou_matrix.size != 0:
            for _ in range(iou_matrix.shape[0]):    # pylint: disable=E1136  # pylint/issues/3139
                idx = \
                    np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                value = np.max(iou_matrix)
                if value < 0.1:
                    break
                iou_indexes.append([idx, value])
                iou_matrix[idx[0], :] = 0
                iou_matrix[:, idx[1]] = 0

        # update for confidence pairs
        for iou_index in iou_indexes:
            self.update(tracker_infos[iou_index[0][0]], 
                        iou_index[1], detection_infos[iou_index[0][1]])
            tracker_idx_list.remove(iou_index[0][0])
            detection_idx_list.remove(iou_index[0][1])

        # initialize for residual detection infos
        for idx in detection_idx_list:
            self.initialize(detection_infos[idx])

        # update residual (fail) track infos
        for idx in tracker_idx_list:
            self.update(tracker_infos[idx], 0, None)

        #print(self.frame_num, len(self.track_candidate_infos), 
        #      len(self.track_infos), len(self.track_removed_infos))
        #ids = [track_info.id for track_info in self.track_removed_infos]
        #time.sleep(0.01)
        #print(ids)

        return self.getTrackInfos()
        
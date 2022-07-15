from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from time import sleep
import cv2
import sys
import threading
import numpy as np
import os
import webbrowser

'''
Darknet compilation required
'''
import background as bg
import detector
import tracker
import utils

class VideoMethod:
    def __init__(self, ui):
        super().__init__()
        self._state = 0
        self.th = 0.1
        self.tracked_id = []

    def load_video(self, file_name, cfg_path, weight_path, data_path):
        self.enable_pause = True

        if not file_name:
            ui.lineEdit.settext(self.file_name)
            return

        self.cap = cv2.VideoCapture(file_name)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_of_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(self.num_of_frame)
        ui.lineEdit.setText(file_name)

        ui.videoProgress.setRange(0, int(self.num_of_frame))


        '''
        Darknet compilation required
        '''
        self.detector = detector.Detector()
        self.detector.initialize(cfg_path, weight_path, data_path)
        self.tracker = tracker.Tracker()

    def play_video(self):
        self.enable_pause = False

    def pause_video(self):
        self.enable_pause = True

    def get_frame(self):   # get only one frame
        '''
        사용자가 선택한 비디오로 부터 하나의 프레임을 입력받아
        BGR -> RGB로 변환하여 반환
        '''
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            self.frame_count = int(
                self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def prev_frame(self):
        self.enable_pause = True
        if self._state == 1:
            self.frame_count -=2
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count-2)
        self.play_once()
    
    def next_frame(self):
        self.enable_pause = True
        self.play_once()
        
    def pressed_video(self):
        self.prev_status = self.enable_pause
        self.enable_pause = True

    def moved_slider(self):
        if self._state == 1:
            self.frame_count = ui.videoProgress.value()-1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, ui.videoProgress.value()-1)
        self.play_once()
        self.enable_pause = self.prev_status

    def play_once(self):
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.num_of_frame:
            return
        ret, frame = self.get_frame()

        rst_frame = np.zeros([100,100,3],dtype=np.uint8)
        rst_frame.fill(240)


        '''
        Darknet compilation required
        검출된 객체 수만큼 detections와 track_infos 반환.
        track_infos[n][0]이 tracking ID (같은 tracking ID면 같은 객체)
        '''
        if ui.dt_chk.isChecked():
            detections = self.detector.detector(frame)
            rst_frame = self.detector.cvDrawBoxes(detections, frame, self.th)
            if ui.tk_chk.isChecked():
                detection_infos = self.tracker.convertDetection2Tracking(
                    detections, self.frame_count)
                track_infos = self.tracker.tracking(
                    detection_infos, self.frame_count)
                rst_frame = self.tracker.cvDrawBoxes(
                    self.tracker.track_infos, frame)
                for i in range(0,len(self.tracker.track_infos-1)):
                    if not self.tracker.track_infos[i].id in self.tracked_id:
                        self.tracked_id.append(self.tracker.track_infos[i].id)
                    # 새로운 tracking 대상이 들어오면 list에 추가, 지도에 추가 요청 코드 작성 필요
                    # 현재 프레임 카운트 정보 self.frame_count


        img4Qt = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)

        ui.pixmap = QPixmap(img4Qt)
        ui.p = ui.pixmap.scaled(720,405,QtCore.Qt.IgnoreAspectRatio)
        ui.leftView.setPixmap(ui.p)
        ui.leftView.update()

        rst_img4Qt = QImage(rst_frame, rst_frame.shape[1], rst_frame.shape[0], QImage.Format_RGB888)

        ui.rst_pixmap = QPixmap(rst_img4Qt)
        ui.rst_p = ui.rst_pixmap.scaled(720,405,QtCore.Qt.IgnoreAspectRatio)
        ui.rightView.setPixmap(ui.rst_p)
        ui.rightView.update()

        ui.videoProgress.setValue(self.frame_count)
        ui.frame_cnt_rate.setText(str(self.frame_count) + '/' + str(self.num_of_frame))

    def Video_to_frame(self, MainWindow):
        self.enable_pause = True
        while True:
            while not self.enable_pause:
                self.play_once()
                sleep(0.01)
    
    def video_thread(self,MainWindow):
        thread = threading.Thread(target=self.Video_to_frame, args=(self,))
        thread.daemon = True
        thread.start()    

class Ui_MainWindow(object):
    def setupUi(self, MainWindow,b_video):

# Create Windows
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1508, 733)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        
        self.msgBox = QMessageBox()
        self.dir_flag = False

    # Menu Bar Design
        new_action = QAction('&New', MainWindow)
        new_action.setShortcut('Ctrl+N')
        new_action.setStatusTip('새 프로젝트 생성')     
        new_action.triggered.connect(self.menu_newcall)   

        open_action = QAction('&Open', MainWindow)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('프로젝트 불러오기')
        open_action.triggered.connect(self.menu_opencall)

        close_action = QAction('&Exit', MainWindow)
        close_action.setShortcut('Ctrl+Q')
        close_action.setStatusTip('종료')
        close_action.triggered.connect(qApp.quit)


        path_setting = QAction('&Set Path', MainWindow)
        path_setting.triggered.connect(self.set_path)

        menu_bar = MainWindow.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        file_menu.addAction(close_action)

    # Display the progress of video
        self.videoProgress = QtWidgets.QSlider(self.centralwidget)
        self.videoProgress.setGeometry(QtCore.QRect(22, 509, 1441, 22))
        self.videoProgress.setOrientation(QtCore.Qt.Horizontal)
        self.videoProgress.setObjectName("videoProgress")
        self.videoProgress.sliderReleased.connect(b_video.moved_slider)       # grab slider event
        self.videoProgress.sliderPressed.connect(b_video.pressed_video)

        self.frame_cnt_rate = QtWidgets.QLabel(self.centralwidget)
        self.frame_cnt_rate.setGeometry(QtCore.QRect(930,650,50,20))
        self.frame_cnt_rate.setText('0/0')
    
    # Group by locate
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(22, 539, 521, 58))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

    # Button design
        self.loadBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.loadBtn.setObjectName("loadVideo")
        self.loadBtn.clicked.connect(self.load)          # load Video
        self.horizontalLayout.addWidget(self.loadBtn)
        self.playBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.playBtn.setObjectName("playVideo")
        self.playBtn.clicked.connect(b_video.play_video)        # play Video
        self.horizontalLayout.addWidget(self.playBtn)
        self.pauseBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.pauseBtn.setObjectName("pauseBtn")
        self.pauseBtn.clicked.connect(b_video.pause_video)       # pause Video
        self.horizontalLayout.addWidget(self.pauseBtn)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.prevBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.prevBtn.setObjectName("prevBtn")
        self.prevBtn.clicked.connect(b_video.prev_frame)
        self.horizontalLayout_2.addWidget(self.prevBtn)
        self.nextBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.nextBtn.setObjectName("nextBtn")
        self.nextBtn.clicked.connect(b_video.next_frame)
        self.openURL_btn = QtWidgets.QPushButton(self.centralwidget)
        self.openURL_btn.setObjectName("OpenURL")
        self.openURL_btn.setGeometry(QtCore.QRect(1265,540,200,23))
        self.openURL_btn.clicked.connect(self.openURL)

    # Group by locate
        self.horizontalLayout_2.addWidget(self.nextBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(20, 10, 1441, 491))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

    # The other Widgets
        self.leftView = QtWidgets.QLabel(self.layoutWidget1)
        self.leftView.setObjectName("leftView")
        self.horizontalLayout_4.addWidget(self.leftView)
        self.rightView = QtWidgets.QLabel(self.layoutWidget1)
        self.rightView.setObjectName("rightView")
        self.horizontalLayout_4.addWidget(self.rightView)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(23, 610, 521, 22))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.layoutWidget2)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_3.addWidget(self.lineEdit)
        
        self.project_path = QtWidgets.QLabel(self.centralwidget)
        self.project_path.setGeometry(QtCore.QRect(23,645,70,22))
        self.project_path.setObjectName("label")
        self.project_path.setText("Project")
        self.path_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.path_edit.setObjectName("pathEdit")
        self.path_edit.setGeometry(QtCore.QRect(90,645,454,20))

    # Check Boxes
        self.chk_label = QtWidgets.QLabel(self.centralwidget)
        self.chk_label.setObjectName("checkLabel")
        self.chk_label.setGeometry(QtCore.QRect(620,545,91,16))
        self.chk_label.setText("Select Options : ")

        self.trsh_label = QtWidgets.QLabel(self.centralwidget)
        self.trsh_label.setObjectName("inputThreshold")
        self.trsh_label.setGeometry(QtCore.QRect(620,650,91,16))
        self.trsh_label.setText("Set Threshold : ")

        self.trsh = QtWidgets.QLineEdit(self.centralwidget)
        self.trsh.setText("0.1")
        self.trsh.setGeometry(QtCore.QRect(740,650,60,16))
        self.trsh.setObjectName("threshold")

        self.dt_chk = QtWidgets.QCheckBox(self.centralwidget)
        self.dt_chk.setGeometry(QtCore.QRect(740, 545, 91, 16))
        self.dt_chk.setObjectName("checkDetected")
        self.dt_chk.stateChanged.connect(self.dt_chk_change)

        self.tk_chk = QtWidgets.QCheckBox(self.centralwidget)
        self.tk_chk.setGeometry(QtCore.QRect(740, 570, 91, 16))
        self.tk_chk.setObjectName("checkTracker")
        self.tk_chk.stateChanged.connect(self.tk_chk_change)
        
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.dir_flag = False

# Funtion Definition
    def load(self):
        self.file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', 'C:\\', 'AVI files(*.avi);; MP4 files(*.mp4)')
        if not self.file_name:
            ui.lineEdit.settext(self.file_name)
            return


        '''
        Darknet compilation required
        cfg, weight, data 파일 입력하는 부분
        '''
        cfg_path = "C:\\anno_ws\AutoAnnotation_option\yolov4-tiny.cfg"
        weight_path = "C:\\anno_ws\AutoAnnotation_option\yolov4-tiny.weights"
        data_path = "C:\\anno_ws\AutoAnnotation_option\yolov4-tiny.data"

        b_video.load_video(self.file_name, cfg_path, weight_path, data_path)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadBtn.setText(_translate("MainWindow", "Load Video"))
        self.playBtn.setText(_translate("MainWindow", "Play"))
        self.pauseBtn.setText(_translate("MainWindow", "Pause"))
        self.prevBtn.setText(_translate("MainWindow", "<"))
        self.nextBtn.setText(_translate("MainWindow", ">"))
        self.openURL_btn.setText(_translate("MainWindow", "Marking"))
        self.label.setText(_translate("MainWindow", "Video Path"))
        self.dt_chk.setText(_translate("MainWindow", "Detector"))
        self.tk_chk.setText(_translate("MainWindow", "Tracker"))

    def set_path(self):
        text, ok = QInputDialog.getText(None, 'Input Dialog', 'Enter the train directory')
        if ok:
            self.path = text
            print(self.path)

    def trsh_btn_clicked(self):
        input_num = float(self.trsh.text())
        if 0< input_num < 1:
            b_video.th = input_num
        else:
            self.disp_error('Error Message', 'Input threshold is out of range')
        
    def dt_chk_change(self):
        if self.dt_chk.isChecked():
            print("check detection")
        else:
            self.tk_chk.setChecked(False)
    
    def tk_chk_change(self):
        if self.tk_chk.isChecked():
            self.dt_chk.setChecked(True)

    def menu_newcall(self):
        workspace = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory")
        if not workspace:
            return

        text, ok = QInputDialog.getText(None, 'Make Directory', 'Enter Project Name:')
        
        if not ok:
            self.disp_error('Error message', 'No value entered')
            self.dir_flag = False
            return

        if os.path.isdir(workspace+'/'+text):
            self.disp_error('Error message', 'Path that already exists')
            self.dir_flag = False
            return

        self.workspace = workspace
        self.path_edit.setText(self.workspace)

        self.workspace = self.workspace+'/'+text
        os.makedirs(os.path.join(self.workspace+'/annotation'))
        os.makedirs(os.path.join(self.workspace+'/background/select'))
        os.makedirs(os.path.join(self.workspace+'/result/merge'))
        os.makedirs(os.path.join(self.workspace+'/result/boxdrawing'))
        os.makedirs(os.path.join(self.workspace+'/frame'))
        self.path_edit.setText(self.workspace)
        self.dir_flag = True

    def menu_opencall(self):
        workspace = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory")

        if not workspace:
            return

        if not (os.path.isdir(workspace+'/annotation')) or not (os.path.isdir(workspace+'/background/select')) or not (os.path.isdir(workspace+'/result/merge')) or not (os.path.isdir(workspace+'/result/boxdrawing')):
            self.disp_error('Error message', 'Path is worng')
            self.dir_flag = False
            self.path_edit.setText('')
            return
        self.path_edit.setText(workspace)
        self.workspace = workspace
        self.dir_flag = True

    def openURL(self):
        '''
        입력할 html을 filepath에 입력
        '''
        message = """<html>
        <head></head>
        <body><p>Hello World!</p></body>
        </html>"""
 
        filepath = "hello.html"
        with open(filepath, 'w') as f:
            f.write(message)
            f.close()
 
        webbrowser.open_new_tab(filepath)
        
    def disp_error(self, title, message):
        self.msgBox.setWindowTitle(title)
        self.msgBox.setText(message)
        self.msgBox.setStandardButtons(QMessageBox.Ok)
        self.msgBox.exec_()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    b_video = VideoMethod(ui)
    ui.setupUi(MainWindow,b_video)
    b_video.video_thread(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

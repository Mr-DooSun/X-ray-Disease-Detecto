# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'narsha_main.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets  
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import argparse
from tkinter import filedialog
from tkinter import *
import pydicom as dicom
from matplotlib import pyplot as plt
from time import sleep

image_path=""
image=None
picture=None

class search():
    ap_config="yolo-obj-medical.cfg"
    ap_weight="yolo-obj-medical_nonContrast.weights"
    ap_classes="yolo-obj-medical.txt"
    image_picture=None

    def Check_bottle(self):
        if str(self.bottle) == "[0]":
            return "Atelectasis / 폐확장부전"
        elif str(self.bottle) == "[1]":
            return "Consolidation / 폐 경화"
        elif str(self.bottle) == "[2]":
            return "Infiltrate / 폐 침윤물"
        elif str(self.bottle) == "[3]":
            return "Pneumothorax / 기흉"
        elif str(self.bottle) == "[4]":
            return "Edema/부종"
        elif str(self.bottle) == "[5]":
            return "Emphysema / 폐기종"
        elif str(self.bottle) == "[6]":
            return "Fibrosis / 섬유증"
        elif str(self.bottle) == "[7]":
            return "Effusion / 가슴막삼출"
        elif str(self.bottle) == "[8]":
            return "Pneumonia/폐렴"
        elif str(self.bottle) == "[9]":
            return "Pleural_thickening / 가슴막 비대화"
        elif str(self.bottle) == "[10]":
            return "Cardiomegaly / 심장 비대"
        elif str(self.bottle) == "[11]":
            return "Nodule / 폐종괴"
        elif str(self.bottle) == "[12]":
            return "Mass / 양성종양"
        elif str(self.bottle) == "[13]":
            return "Hernia / 탈장"
        elif str(self.bottle) == "[14]":
            return ""
        elif str(self.bottle) == "[15]":
            return ""

        # return self.bottle
    def Per(self):
        print(str(self.per))

        if str(self.per) == "":
            return 

        else :
            self.per=str(self.per)
            self.per+=" %"
            return self.per

    def DcmToPng(self):
        self.d = dicom.read_file(image_path)
        self.Mate=plt.imsave("defalt.png",self.d.pixel_array,cmap=plt.cm.bone)

    def get_output_layers(self,net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers


    def draw_prediction(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        print(class_id)
        print(self.classes)
        self.label = str(self.classes[class_id])
        self.color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), self.color, 2)
        cv2.putText(img, self.label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

    def Analysis(self):
        self.per=""
        self.bottle=""
        self.DcmToPng()
        global picture
        picture=None
        image = cv2.imread("defalt.png")

        self.Width = image.shape[1]
        self.Height = image.shape[0]
        self.scale = 0.00392

        self.classes = None

        with open(self.ap_classes, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.net = cv2.dnn.readNet(self.ap_weight, self.ap_config)

        self.blob = cv2.dnn.blobFromImage(image, self.scale, (416,416), (0,0,0), True, crop=False)

        self.net.setInput(self.blob)

        self.outs = self.net.forward(self.get_output_layers(self.net))

        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4


        for self.out in self.outs:
            for self.detection in self.out:
                self.scores = self.detection[5:]
                self.class_id = np.argmax(self.scores)
                self.confidence = self.scores[self.class_id]
                if self.confidence > 0.5:
                    self.center_x = int(self.detection[0] * self.Width)
                    self.center_y = int(self.detection[1] * self.Height)
                    self.w = int(self.detection[2] * self.Width)
                    self.h = int(self.detection[3] * self.Height)
                    self.x = self.center_x - self.w / 2
                    self.y = self.center_y - self.h / 2
                    self.class_ids.append(self.class_id)
                    self.confidences.append(float(self.confidence))
                    self.boxes.append([self.x, self.y, self.w, self.h])

        self.bottle=self.class_ids
        try :
            self.per=int(self.confidences[0]*100)
        except :
            print("None")

        print(self.class_ids)

        self.indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.conf_threshold, self.nms_threshold)

        for self.i in self.indices:
            self.i = self.i[0]
            self.box = self.boxes[self.i]
            self.x = self.box[0]
            self.y = self.box[1]
            self.w = self.box[2]
            self.h = self.box[3]
            self.draw_prediction(image, self.class_ids[self.i], self.confidences[self.i], round(self.x), round(self.y), round(self.x+self.w), round(self.y+self.h))

        self.image_picture=image

        self.convertToQtFormat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888) 
        self.p = QPixmap(self.convertToQtFormat)
        picture = self.p.scaled(440, 440, QtCore.Qt.IgnoreAspectRatio)

    def Save_image(self,):
        cv2.imwrite(image_path[0:len(image_path) - 4] + ".png",self.image_picture)

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
#============================================================================================================================

class Ui_MainWindow(object):
    sh=search()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 710)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")

        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.AnalysisButton = QtWidgets.QPushButton(self.centralwidget)
        self.AnalysisButton.setGeometry(QtCore.QRect(100, 480, 210, 40))
        self.AnalysisButton.setFont(font)
        self.AnalysisButton.setStyleSheet("background-color: rgb(58, 116, 154);\ncolor: rgb(255, 255, 255);\nborder:2px solid #ffffff\n\n")
        self.AnalysisButton.setObjectName("AnalysisButton")
        
        self.FolderButton = QtWidgets.QPushButton(self.centralwidget)
        self.FolderButton.setGeometry(QtCore.QRect(100, 550, 210, 40))
        self.FolderButton.setFont(font)
        self.FolderButton.setStyleSheet("background-color: rgb(58, 116, 154);\ncolor: rgb(255, 255, 255);\nborder:2px solid #ffffff\n")
        self.FolderButton.setObjectName("FolderButton")

        self.SaveButton = QtWidgets.QPushButton(self.centralwidget)
        self.SaveButton.setGeometry(QtCore.QRect(100, 620, 210, 40))
        self.SaveButton.setFont(font)
        self.SaveButton.setStyleSheet("background-color: rgb(58, 116, 154);\ncolor: rgb(255, 255, 255);\nborder:2px solid #ffffff\n")
        self.SaveButton.setObjectName("SaveButton")

        self.dicom_frame = QtWidgets.QFrame(self.centralwidget)
        self.dicom_frame.setGeometry(QtCore.QRect(410, 10, 671, 420))
        self.dicom_frame.setStyleSheet("background-color: rgb(72, 103, 154);")
        self.dicom_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.dicom_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.dicom_frame.setObjectName("dicom_frame")

        self.picture_label = QtWidgets.QLabel(self.dicom_frame)
        self.picture_label.setGeometry(QtCore.QRect(135, 10, 400, 400))
        self.picture_label.setObjectName("picture_label")
        self.p = QtGui.QPixmap("load_image.png")
        self.picture = self.p.scaled(400, 400, QtCore.Qt.IgnoreAspectRatio)
        self.picture_label.setPixmap(self.picture)

        self.logo_label = QtWidgets.QLabel(self.centralwidget)
        self.logo_label.setGeometry(QtCore.QRect(90, 20, 311, 400))
        self.logo_label.setObjectName("logo_label")
        self.logo_label.setPixmap(QtGui.QPixmap("logo.png"))

        self.list_label_1 = QtWidgets.QLabel(self.centralwidget)
        self.list_label_1.setGeometry(QtCore.QRect(410, 510, 411, 60))
        self.list_label_1.setStyleSheet("border:2px solid rgb(72, 103, 154);")
        self.list_label_1.setTextFormat(QtCore.Qt.PlainText)
        self.list_label_1.setObjectName("list_label_1")

        self.list_label_name_1 = QtWidgets.QLabel(self.centralwidget)
        self.list_label_name_1.setGeometry(QtCore.QRect(410, 485, 70, 20))
        self.list_label_name_1.setObjectName("list_label_name_1")
        self.list_label_name_1.setFont(QtGui.QFont("맑은 고딕",11))
        self.list_label_name_1.setText("File path")

        self.list_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.list_label_2.setGeometry(QtCore.QRect(410, 600, 411, 60))
        self.list_label_2.setStyleSheet("border:2px solid rgb(72, 103, 154);")
        self.list_label_2.setTextFormat(QtCore.Qt.PlainText)
        self.list_label_2.setObjectName("list_label_2")

        self.list_label_name_2 = QtWidgets.QLabel(self.centralwidget)
        self.list_label_name_2.setGeometry(QtCore.QRect(410, 575, 70, 20))
        self.list_label_name_2.setObjectName("list_label_name_2")
        self.list_label_name_2.setFont(QtGui.QFont("맑은 고딕",11))
        self.list_label_name_2.setText("Log")

        #병이름 퍼센트
        self.result_label = QtWidgets.QLabel(self.centralwidget)
        self.result_label.setGeometry(QtCore.QRect(830, 510, 250, 150))
        self.result_label.setStyleSheet("border:2px solid rgb(72, 103, 154);")
        self.result_label.setTextFormat(QtCore.Qt.PlainText)
        self.result_label.setObjectName("result_label")

        self.result_label_name = QtWidgets.QLabel(self.centralwidget)
        self.result_label_name.setGeometry(QtCore.QRect(830, 485, 250, 20))
        self.result_label_name.setObjectName("result_label_name")
        self.result_label_name.setFont(QtGui.QFont("맑은 고딕",11))
        self.result_label_name.setText("Disease Name / Percentage")

        self.result_label_1 = QtWidgets.QLabel(self.centralwidget)
        self.result_label_1.setGeometry(QtCore.QRect(850, 540, 120, 12))
        self.result_label_1.setObjectName("result_label_1")

        self.result_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.result_label_2.setGeometry(QtCore.QRect(925, 620, 56, 20))
        self.result_label_2.setObjectName("result_label_2")

        
        self.title_label = QtWidgets.QLabel(self.centralwidget)
        self.title_label.setGeometry(QtCore.QRect(45, 380, 321, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(16)
        self.title_label.setFont(font)
        self.title_label.setTextFormat(QtCore.Qt.PlainText)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setObjectName("title_label")

        self.title_label.setText("X-ray DICOM image Analyzer")
        self.title_label.setPixmap(QtGui.QPixmap("name.png"))
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.chart_label = QtWidgets.QLabel(self.dicom_frame)
        self.chart_label.setGeometry(QtCore.QRect(420, 5, 250, 415))
        self.chart_label.setObjectName("chart_label")
        self.chart_label.setPixmap(QtGui.QPixmap("chart.png"))
        self.chart_label.setVisible(False) #라벨을 껐다 켰다 할수있다.

        self.chart_label_1 = QtWidgets.QLabel(self.chart_label)
        self.chart_label_1.setGeometry(QtCore.QRect(130, 30, 100, 20))
        self.chart_label_1.setObjectName("chart_label_1")
        self.chart_label_1.setAlignment(Qt.AlignCenter)

        self.chart_label_2 = QtWidgets.QLabel(self.chart_label)
        self.chart_label_2.setGeometry(QtCore.QRect(130, 96, 100, 20))
        self.chart_label_2.setObjectName("chart_label_1")
        self.chart_label_2.setAlignment(Qt.AlignCenter)

        self.chart_label_3 = QtWidgets.QLabel(self.chart_label)
        self.chart_label_3.setGeometry(QtCore.QRect(130, 162, 100, 20))
        self.chart_label_3.setObjectName("chart_label_1")
        self.chart_label_3.setAlignment(Qt.AlignCenter)

        self.chart_label_4 = QtWidgets.QLabel(self.chart_label)
        self.chart_label_4.setGeometry(QtCore.QRect(130, 228, 100, 20))
        self.chart_label_4.setObjectName("chart_label_1")
        self.chart_label_4.setAlignment(Qt.AlignCenter)

        self.chart_label_5 = QtWidgets.QLabel(self.chart_label)
        self.chart_label_5.setGeometry(QtCore.QRect(130, 294, 100, 20))
        self.chart_label_5.setObjectName("chart_label_1")
        self.chart_label_5.setAlignment(Qt.AlignCenter)

        self.chart_label_6 = QtWidgets.QLabel(self.chart_label)
        self.chart_label_6.setGeometry(QtCore.QRect(130, 360, 100, 20))
        self.chart_label_6.setObjectName("chart_label_1")
        self.chart_label_6.setAlignment(Qt.AlignCenter)

        self.p = QtGui.QPixmap("load_image.png")
        self.picture = self.p.scaled(400, 400, QtCore.Qt.IgnoreAspectRatio)
        self.picture_label.setPixmap(self.picture)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1100, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "X-ray Decomposition Analysis"))
        self.AnalysisButton.setText(_translate("MainWindow", "Analysis"))
        self.FolderButton.setText(_translate("MainWindow", "Load"))
        self.SaveButton.setText(_translate("MainWindow", "Save"))
        self.title_label.setText(_translate("MainWindow","X-ray Decomposition Analyzer"))

    def Analysis(self,MainWindow):
        if image_path=="":
            self.list_label_2.setText("Not find file")

        else :
            self.sh.Analysis()
            try :
                self.picture_label.setGeometry(QtCore.QRect(10, 10, 400, 400))
                self.picture_label.setPixmap(picture)
                self.picture_label.update()

                self.list_label_2.setText("Success Detecting file")
                bottle=self.sh.Check_bottle()
                per=self.sh.Per()
                self.result_label_1.setText(str(bottle))
                self.result_label_2.setText(str(per))
            except :
                self.list_label_2.setText("hello")
                self.result_label_1.setText("No Problem")

            self.chart_label.setVisible(True)
            self.chart_label_1.setText(self.d[0x0010,0x0040].value)
            self.chart_label_2.setText(self.d[0x0010,0x1010].value)
            self.chart_label_3.setText(self.d[0x0018,0x0015].value)
            self.chart_label_4.setText(str(self.d[0x0028,0x0010].value)+" * "+str(self.d[0x0028,0x0011].value))
            self.chart_label_5.setText(self.d[0x0028,0x2114].value)
            self.chart_label_6.setText(str(len(self.d[0x7fe0,0x0010].value)))

    def Dir_Path(self,MainWindow):
        global image_path
        image_path=""
        image=None
        picture=None
        try :
            root=Tk()
            root.filename = filedialog.askopenfilename(filetypes = (("dcm files","*.dcm"),))
            image_path=root.filename

            self.d = dicom.read_file(root.filename)

            self.list_label_1.setText(image_path)
            root.destroy()
            self.sh.DcmToPng()

            self.file = QtGui.QPixmap("defalt.png")
            self.file = self.file.scaled(400, 400, QtCore.Qt.IgnoreAspectRatio)
            self.picture_label.setPixmap(self.file)

            self.list_label_2.setText("Success load file")
        except :
            self.list_label_2.setText("Error")
            root.destroy()

    def Save_Dir(self,MainWindow):
        if image_path=="":
            self.list_label_2.setText("Not find file")
        else :
            self.sh.Save_image()
            self.list_label_2.setText("Success save")

    def button_clicked(self,MainWindow):
        self.AnalysisButton.clicked.connect(self.Analysis)
        self.FolderButton.clicked.connect(self.Dir_Path) 
        self.SaveButton.clicked.connect(self.Save_Dir)

if __name__=="__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()

    ui.setupUi(MainWindow)
    ui.button_clicked(MainWindow)
    
    MainWindow.show()

    sys.exit(app.exec_())
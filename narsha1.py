from PyQt5 import QtCore, QtGui, QtWidgets  
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import argparse
from tkinter import filedialog
from tkinter import *

image_path=""
image=None
picture=None

class search():
    ap_config="yolo-obj-medical.cfg"
    ap_weight="yolo-obj-medical_nonContrast.weights"
    ap_classes="yolo-obj-medical.txt"
    image_picture=None
    def get_output_layers(self,net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers


    def draw_prediction(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        self.label = str(self.classes[class_id])
        self.color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), self.color, 2)
        cv2.putText(img, self.label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

    def Analysis(self):
        global picture
        image = cv2.imread(image_path)

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
        picture = self.p.scaled(650, 440, QtCore.Qt.IgnoreAspectRatio)

    def Save_image(self,):
        cv2.imwrite("test.png",self.image_picture)

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
#============================================================================================================================

class Ui_MainWindow(object):
    sh=search()

    def setupUi(self, MainWindow):
        font = QtGui.QFont()
        font.setFamily("나눔스퀘어")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1109, 713)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.AnalysisButton = QtWidgets.QPushButton(self.centralwidget)
        self.AnalysisButton.setGeometry(QtCore.QRect(120, 380, 170, 50))
        self.AnalysisButton.setFont(font)
        self.AnalysisButton.setStyleSheet("background-color: rgb(255, 255, 255);\nborder:2px solid #222dff")
        Analysis_icon = QtGui.QIcon()
        Analysis_icon.addPixmap(QtGui.QPixmap("search.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.AnalysisButton.setIcon(Analysis_icon)
        self.AnalysisButton.setIconSize(QtCore.QSize(20, 20))
        self.AnalysisButton.setObjectName("AnalysisButton")

        self.FolderButton = QtWidgets.QPushButton(self.centralwidget)
        self.FolderButton.setGeometry(QtCore.QRect(120, 480, 170, 50))
        self.FolderButton.setFont(font)
        self.FolderButton.setStyleSheet("background-color: rgb(255, 255, 255);\nborder:2px solid #222dff")
        Folder_icon = QtGui.QIcon()
        Folder_icon.addPixmap(QtGui.QPixmap("folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.FolderButton.setIcon(Folder_icon)
        self.FolderButton.setIconSize(QtCore.QSize(20, 20))
        self.FolderButton.setObjectName("FolderButton")

        self.SaveButton = QtWidgets.QPushButton(self.centralwidget)
        self.SaveButton.setGeometry(QtCore.QRect(120, 580, 170, 50))
        self.SaveButton.setFont(font)
        self.SaveButton.setStyleSheet("background-color: rgb(255, 255, 255);\nborder:2px solid #222dff")
        Save_icon = QtGui.QIcon()
        Save_icon.addPixmap(QtGui.QPixmap("save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.SaveButton.setIcon(Save_icon)
        self.SaveButton.setIconSize(QtCore.QSize(20, 20))
        self.SaveButton.setObjectName("SaveButton")
        
        self.dicom_frame = QtWidgets.QFrame(self.centralwidget)
        self.dicom_frame.setGeometry(QtCore.QRect(410, 10, 671, 461))
        self.dicom_frame.setStyleSheet("background-color: rgb(34, 45, 255);")
        self.dicom_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.dicom_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.dicom_frame.setObjectName("dicom_frame")
        
        self.picture_label = QtWidgets.QLabel(self.dicom_frame)
        self.picture_label.setGeometry(QtCore.QRect(10, 10, 650, 440))
        self.picture_label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.picture_label.setObjectName("picture_label")
        # self.p = QtGui.QPixmap("00000756_001.png")
        # self.picture = self.p.scaled(650, 440, QtCore.Qt.IgnoreAspectRatio)
        # self.picture_label.setPixmap(self.picture)
        
        self.log_label = QtWidgets.QLabel(self.centralwidget)
        self.log_label.setGeometry(QtCore.QRect(415, 510, 661, 151))
        self.log_label.setStyleSheet("border:2px solid #222dff")
        self.log_label.setTextFormat(QtCore.Qt.PlainText)
        self.log_label.setObjectName("log_label")

        self.logo_label= QtWidgets.QLabel(self.centralwidget)
        self.logo_label.setGeometry(QtCore.QRect(50, 20, 311, 281))
        self.logo_label.setObjectName("logo_label")
        self.logo_label.setPixmap(QtGui.QPixmap("logo.png"))

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1109, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.AnalysisButton.setText(_translate("MainWindow", "Analysis"))
        self.FolderButton.setText(_translate("MainWindow", "Load"))
        self.SaveButton.setText(_translate("MainWindow", "Save"))

    def Analysis(self,MainWindow):
        if image_path=="":
            print("not find file")

        else :
            self.sh.Analysis()
            self.picture_label.setPixmap(picture)
            self.picture_label.update()

    def Dir_Path(self,MainWindow):
        global image_path
        root=Tk()
        root.filename = filedialog.askopenfilename(filetypes = (("png files","*.png"),("jpg files","*.jpg"),("all files","*.*")))
        
        image_path=root.filename
        self.log_label.setText(image_path)
        root.destroy()

    def Save_Dir(self,MainWindow):
        if image_path=="":
            print("not find file")
        else :
            self.sh.Save_image
            self.log_label.setText(image_path+" Success Save")

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
import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsTextItem, \
    QMainWindow, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QPointF, QCoreApplication, QRectF
import cv2 as cv
import pandas as pd
import sys
from PyQt5.QtCore import QFile, QTextStream
from PyQt5.QtWidgets import QApplication
import qdarkstyle
from qt_material import  apply_stylesheet



class CustomCircle(QGraphicsEllipseItem):
    def __init__(self, x, y, r, shape, name, parent, idx, nbr, conf):
        self.conf = conf
        self.idx = idx
        self.nbr = nbr
        super().__init__(0, 0, r, r)
        self.r = r
        self.w, self.h = shape
        self.setPos(x - self.r, y - self.r)
        #self.setBrush(Qt.green)
        self.setBrush(QColor(170,255,150))
        self.setAcceptHoverEvents(True)
        self.text = QGraphicsTextItem(name, parent=self)
        #self.text.setDefaultTextColor(Qt.blue)
        self.text.setDefaultTextColor(QColor(255,64,129))
        self.text.setPos(10,-3)
        self.parent = parent
        self.move = False

    def hoverEnterEvent(self, event):
        app.instance().setOverrideCursor(Qt.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        app.instance().restoreOverrideCursor()

    def mousePressEvent(self, event):
        self.parent.refreshJoints()
        #self.setBrush(Qt.red)
        self.setBrush(QColor(197,96,96))
        self.parent.selectedJoint = self.nbr

    def mouseMoveEvent(self, event):
        if self.parent.move:
            circle_og_pos = self.scenePos()

            cursor_og_pos = event.lastScenePos()
            cursor_new_pos = event.scenePos()

            cursor_new_x = circle_og_pos.x() + cursor_new_pos.x() - cursor_og_pos.x()
            cursor_new_y = circle_og_pos.y() + cursor_new_pos.y() - cursor_og_pos.y()
            self.setPos(QPointF(max(-self.r / 2, min(cursor_new_x, self.w - (self.r / 2))),
                                max(-self.r / 2, min(cursor_new_y, self.h - (self.r / 2)))))
            self.move = True

    def mouseReleaseEvent(self, event):
        if self.move:
            print('x: {0}, y: {1}'.format(self.pos().x() + self.r / 2, self.pos().y() + self.r / 2))
            self.move = False

    def getPos(self):
        return (self.pos().x() + self.r, self.pos().y() + self.r)


class MergeCorrector(QGraphicsView):
    def __init__(self, screen_width, scalefactor=None):
        super().__init__()
        ################################################################################################################
        ################################################################################################################
        if scalefactor is None:
            if screen_width == 1080:
                scalefactor = 0.5
            elif screen_width == 2560:
                scalefactor = 0.33
            else:
                scalefactor = 0.25
        self.scalefactor = scalefactor
        session = 4
        cam = 2
        video_path = "E:/Nurse_videos/S" + str(session) + "/southampton " + str(session) + " cam " + str(cam) + '.mp4'
        self.data_path = "E:\\Nurse_Data/S4/" + "OpenPose_southampton_4_cam_2_fixed_labeled.csv"
        ################################################################################################################
        ################################################################################################################
        """Load Remaining Duplicates"""
        ################################################################################################################
        self.df = pd.read_csv(self.data_path, engine='python')

        df1 = self.df[self.df[' Person_ID'] % 2 == 0]
        self.duplicate = df1[df1.duplicated(['# Image_ID', ' Person_ID'], keep=False)]
        self.duplicate = self.duplicate.sort_values(by=['# Image_ID', ' Person_ID'])
        self.duplicate['# Image_ID'] = [int(i[:-4][(len(i[:-4]) - 8):]) for i in self.duplicate.iloc[:, 0]]
        self.sceleton_idx = 0

        self.row = self.duplicate.iloc[0].copy()
        self.result = self.duplicate.iloc[0].copy()
        self.usedcols = []
        ################################################################################################################
        ################################################################################################################

        self.cap = cv.VideoCapture(video_path)

        w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.w = w
        self.h = h
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.cb = QtWidgets.QComboBox()
        self.cb.setGeometry(w, 50, 200, 50)
        self.move = False
        self.changed = False
        self.joints = [self.df.columns[k][:-2] for k in range(3, len(self.df.columns) - 8, 3)]
        self.cb.addItems(self.joints)

        self.b1 = QtWidgets.QPushButton()
        self.b1.setText("apply")
        self.b1.clicked.connect(self.apply)
        self.b1.setGeometry(w, 0, 200, 50)

        self.b2 = QtWidgets.QPushButton()
        self.b2.setText("next")
        self.b2.clicked.connect(self.nextSkeleton)
        self.b2.setGeometry(w, 350, 200, 50)

        self.b3 = QtWidgets.QPushButton()
        self.b3.setText("save")
        self.b3.clicked.connect(self.save)
        self.b3.setGeometry(w, 400, 200, 50)

        self.b4 = QtWidgets.QPushButton()
        self.b4.setText("move")
        self.b4.clicked.connect(self.button_clicked)
        self.b4.setGeometry(w, 500, 200, 50)

        self.b5 = QtWidgets.QPushButton()
        self.b5.setText("Back")
        self.b5.clicked.connect(self.back_button)
        self.b5.setGeometry(w, 550, 200, 50)

        self.b6 = QtWidgets.QPushButton()
        self.b6.setText("add Joint")
        self.b6.clicked.connect(self.addJoint)
        self.b6.setGeometry(w, 300, 200, 50)

        self.testlbl = QtWidgets.QLabel()
        self.testlbl.setText(f'Frame Number: {self.row[0]}')
        self.testlbl.move(w, h-50)

        self.cap.set(1, self.row[0])
        ret, frame = self.cap.read()

        self.drawImageBackground(frame, w, h)

        self.setSceneRect(0, 0, w * 1, h)
        self.scene.addWidget(self.cb)
        self.scene.addWidget(self.b1)
        self.scene.addWidget(self.b2)
        self.scene.addWidget(self.b3)
        self.scene.addWidget(self.b4)
        self.scene.addWidget(self.b5)
        self.scene.addWidget(self.b6)
        self.scene.addWidget(self.testlbl)
        self.b5.setEnabled(False)

        self.rowJointColLeangth = len(self.df.columns) - 9

        self.jointList = []
        self.updateSkeleton()
        self.showMaximized()
        QCoreApplication.processEvents()

    def refreshJoints(self):
        for i in self.jointList:
            #i.setBrush(Qt.green)
            i.setBrush(QColor(170,255,150))

    def drawImageBackground(self, img, w, h):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB, img)
        self.scene.clear()
        self.imgQ = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pixMap = QPixmap(self.imgQ)
        self.pm = self.scene.addPixmap(pixMap)

        self.fitInView(QRectF(0, 0, self.scalefactor * w, self.scalefactor * h), Qt.KeepAspectRatio)
        self.scene.update()

    def redrawImageBackground(self, img, w, h):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB, img)
        self.imgQ = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        self.pm.setPixmap(QPixmap(self.imgQ))

    def updateSkeleton(self):
        for i in self.jointList:
            self.scene.removeItem(i)

        x_s = [int(self.row[k]) for k in range(2, len(self.df.columns) - 9, 3)]
        y_s = [int(self.row[k]) for k in range(3, len(self.df.columns) - 8, 3)]
        conf = [self.row[k] for k in range(4, len(self.df.columns) - 7, 3)]
        idxs = [k for k in range(4, len(self.df.columns) - 7, 3)]
        self.jointList = []
        i = 0
        for x, y, c, name, idx in zip(x_s, y_s, conf, self.joints, idxs):
            if x != 0 and y != 0:
                b = CustomCircle(x, y, 15, shape=(self.w, self.h), name=name, parent=self, idx=idx, nbr=i, conf=c)
                self.scene.addItem(b)
                self.jointList.append(b)
                self.usedcols.append(idx)
                i += 1
        self.selectedJoint = 0

        self.tomany = {}
        for i in idxs:
            self.tomany[i] = 0
        for i in self.usedcols:
            self.tomany[i] += 1

        #self.jointList[0].setBrush(Qt.red)
        self.jointList[0].setBrush(QColor(197,96,96))
        self.b6.setEnabled(len(self.jointList) < 25)

    def addJoint(self):
        self.changed = True
        x = self.w / 2
        y = self.h / 2
        name = self.joints[0]
        idx = 4
        b = CustomCircle(x, y, 15, shape=(self.w, self.h), name=name, parent=self, idx=idx, nbr=len(self.jointList),
                         conf=1)
        self.scene.addItem(b)
        self.jointList.append(b)
        self.tomany[idx] += 1

        flag = True
        for _, item in self.tomany.items():
            if item > 1:
                flag = False
        self.b3.setEnabled(flag)
        self.b2.setEnabled(flag)
        self.b5.setEnabled(flag)
        if self.sceleton_idx + 1 == len(self.duplicate):
            self.b3.setEnabled(False)
        if self.sceleton_idx == 0:
            self.b5.setEnabled(False)
        self.b6.setEnabled(len(self.jointList) < 25)

    def button_clicked(self):
        self.move = not self.move

    def apply(self):
        self.changed = True
        current = 4 + (self.cb.currentIndex() * 3)
        self.jointList[self.selectedJoint].text.setPlainText(self.df.columns[current][:-11])
        if current != self.jointList[self.selectedJoint].idx:
            self.tomany[self.jointList[self.selectedJoint].idx] = min(0, self.tomany[
                self.jointList[self.selectedJoint].idx] - 1)

            #self.tomany[self.jointList[self.selectedJoint].idx] -= 1
            self.jointList[self.selectedJoint].idx = current
            self.tomany[current] += 1
        flag = True
        for _, item in self.tomany.items():
            if item > 1:
                flag = False
        self.b3.setEnabled(flag)
        self.b2.setEnabled(flag)
        self.b5.setEnabled(flag)
        if self.sceleton_idx + 1 == len(self.duplicate):
            self.b3.setEnabled(False)
        if self.sceleton_idx == 0:
            self.b5.setEnabled(False)

    def nextSkeleton(self):
        if self.changed:
            ret_row = [0 for i in range(self.rowJointColLeangth)]
            adjust = 2
            for joint in self.jointList:
                ret_row[joint.idx - adjust] = joint.conf
                ret_row[joint.idx - 2 - adjust], ret_row[joint.idx - 1 - adjust] = joint.getPos()
            self.df.loc[self.row.name] = self.df.loc[self.row.name][:2].tolist() + ret_row + self.df.loc[self.row.name][
                                                                                             -7:].tolist()
            self.duplicate.iloc[self.sceleton_idx] = self.duplicate.iloc[self.sceleton_idx][:2].tolist() + ret_row + \
                                                     self.duplicate.iloc[self.sceleton_idx][
                                                     -7:].tolist()
        self.sceleton_idx += 1
        self.row = self.duplicate.iloc[self.sceleton_idx].copy()
        self.result = self.duplicate.iloc[self.sceleton_idx].copy()

        self.usedcols = []
        self.testlbl.setText(f'Frame Number: {self.row[0]}')
        self.cap.set(1, self.row[0])
        ret, frame = self.cap.read()
        self.redrawImageBackground(frame, self.w, self.h)
        self.updateSkeleton()
        self.changed = False
        self.b5.setEnabled(True)
        if self.sceleton_idx + 1 == len(self.duplicate):
            self.b3.setEnabled(False)

    def back_button(self):
        if self.changed:
            ret_row = [0 for i in range(self.rowJointColLeangth)]
            adjust = 2
            for joint in self.jointList:
                ret_row[joint.idx - adjust] = joint.conf
                ret_row[joint.idx - 2 - adjust], ret_row[joint.idx - 1 - adjust] = joint.getPos()
            self.df.loc[self.row.name] = self.df.loc[self.row.name][:2].tolist() + ret_row + self.df.loc[self.row.name][
                                                                                             -7:].tolist()
            self.duplicate.iloc[self.sceleton_idx] = self.duplicate.iloc[self.sceleton_idx][:2].tolist() + ret_row + \
                                                     self.duplicate.iloc[self.sceleton_idx][
                                                     -7:].tolist()
        self.sceleton_idx -= 1
        self.row = self.duplicate.iloc[self.sceleton_idx].copy()
        self.result = self.duplicate.iloc[self.sceleton_idx].copy()

        self.usedcols = []
        self.testlbl.setText(f'Frame Number: {self.row[0]}')
        self.cap.set(1, self.row[0])
        ret, frame = self.cap.read()
        self.redrawImageBackground(frame, self.w, self.h)
        self.updateSkeleton()
        self.changed = False
        self.b2.setEnabled(True)
        if self.sceleton_idx == 0:
            self.b5.setEnabled(False)

    def save(self):
        if self.changed:
            ret_row = [0 for i in range(self.rowJointColLeangth)]
            for joint in self.jointList:
                ret_row[joint.idx] = joint.conf
                ret_row[joint.idx - 2], ret_row[joint.idx - 1] = joint.getPos()
            self.df.loc[self.row.name] = self.df.loc[self.row.name][:2].tolist() + ret_row + self.df.loc[self.row.name][
                                                                                             -7:].tolist()
            self.df.to_csv(self.data_path[:4] + 't.csv', index=False)
        pass


app = QApplication(sys.argv)
win = QMainWindow()


app.setStyle('GTK')

#app.setStyleSheet(qdarkstyle.load_stylesheet())
#apply_stylesheet(app, theme='test.xml')
app
scale = 0.25
view = MergeCorrector(screen_width=app.primaryScreen().size().width())

view.show()
sys.exit(app.exec_())

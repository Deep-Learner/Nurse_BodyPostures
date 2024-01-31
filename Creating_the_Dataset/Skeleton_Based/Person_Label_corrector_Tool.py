from tkinter import *
from PIL import ImageTk, Image
import cv2 as cv
import pandas as pd
from Creating_the_Dataset.Skeleton_Based.Correct_misclassification import run_correction

"""def resize_bb(df):
    for index, row in df.iterrows():
        x_values = []
        y_values = []

        for k in range(4, len(df.columns) - 5, 3):
            if row[k] > 0:
                x_values.append(row[k - 2])
                y_values.append(row[k - 1])
        df.at[index, 'x1'] = min(x_values)
        df.at[index, 'y1'] = min(y_values)
        df.at[index, 'x2'] = max(x_values)
        df.at[index, 'y2'] = max(y_values)
    return df
"""


class MainWindow():

    def __init__(self, main, path, video_path, historyName):
        self.main = main
        self.pairs = [(0, 1), (15, 0), (16, 0), (17, 15), (18, 16), (1, 8), (2, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                      (8, 9),
                      (9, 10), (10, 11), (11, 24), (11, 22), (22, 23), (8, 12), (12, 13), (13, 14), (14, 21), (14, 19),
                      (19, 20)]
        self.images = []
        from pathlib import Path
        self.histPath = historyName + '.txt'
        if Path(self.histPath).is_file():
            with open(self.histPath, 'r') as f:
                self.history = set([int(l) for l in f])
        else:
            self.history = set()

        self.path = path
        self.dict = {0: "Blue Button",
                     1: "Dummy",
                     2: "Green Button",
                     3: "NoPerson",
                     4: "Red Button",
                     5: "Others"
                     }
        self.class_col = {0: 'blue',
                          1: 'magenta',
                          2: 'green',
                          2: 'green',
                          3: 'gray',
                          4: 'red',
                          5: 'gold'
                          }
        self.df = pd.read_csv(path, engine='python')
        self.duplicate = self.df[self.df.duplicated(['# Image_ID', ' Person_ID'], keep=False)]
        self.duplicate = self.duplicate.sort_values(by=['# Image_ID', ' Person_ID'])
        self.duplicate = self.duplicate[self.duplicate[' Person_ID'] != 5]
        self.duplicate = self.duplicate.drop(list(set(self.duplicate.index).intersection(set(self.history))))
        self.checked = 0
        self.total = len(self.duplicate)
        self.as_var = BooleanVar()
        self.as_var.set(False)

        self.current = 0
        self.color = 'magenta'

        # video
        self.cap = cv.VideoCapture(video_path)
        ret, self.image = self.cap.read()
        self.count = 0

        self.select_next_img()

        # canvas for image
        self.canvas = Canvas(main, width=1280 + 372, height=700)
        self.canvas.grid(row=0, column=0, columnspan=7)

        # set first image on canvas
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.img)
        self.create_transparent_rectangle(self.row[-4], self.row[-3], self.row[-2], self.row[-1],
                                          fill=self.color, alpha=.25, tags="rect")

        x_values = [self.row[k - 2] for k in range(4, len(self.row) - 5, 3) if self.row[k] > 0]
        y_values = [self.row[k - 1] for k in range(4, len(self.row) - 5, 3) if self.row[k] > 0]

        self.create_transparent_rectangle(min(x_values), min(y_values), max(x_values), max(y_values), fill=self.color,
                                          alpha=.3,
                                          tags="rect")

        x_s = [int(self.row[k]) for k in range(2, len(self.row) - 6, 3)]
        y_s = [int(self.row[k]) for k in range(3, len(self.row) - 5, 3)]

        for i, j in self.pairs:
            if x_s[i] != 0 and y_s[i] != 0 and x_s[j] != 0 and y_s[j] != 0:
                self.canvas.create_line(x_s[i], y_s[i], x_s[j], y_s[j], width=5, fill='black', tags="rect")

        self.manual = PhotoImage(file="button.png")
        self.image_on_canvas2 = self.canvas.create_image(1280, 0, anchor=NW, image=self.manual)

        self.canvas.bind("w", self.onArrowB)  # "Blue Button"
        self.canvas.bind("s", self.onArrowR)  # "Red Button"
        self.canvas.bind("a", self.onArrowG)  # "Green Button"
        self.canvas.bind("d", self.onArrowD)  # "Dummy"
        self.canvas.bind("e", self.onArrowNP)  # Other
        self.canvas.bind("q", self.onArrowO)  # No Person

        # self.canvas.bind("<Up>", self.onArrowB)  # "Blue Button"
        # self.canvas.bind("<Down>", self.onArrowR)  # "Red Button"

        self.canvas.bind("<Left>", self.onArrowBack)  # "Back"
        self.canvas.bind("<Right>", self.onArrowS)  # "Skip"

        # self.canvas.bind("<space>", self.onArrowS)  # skip
        self.canvas.focus_set()

        self.button0 = Button(main, text="Blue Button", command=self.onButton0, fg="blue")
        self.button0.grid(row=1, column=1)
        self.button1 = Button(main, text="Dummy", command=self.onButton1)
        self.button1.grid(row=1, column=4)
        self.button2 = Button(main, text="Green Button", command=self.onButton2, fg="green")
        self.button2.grid(row=1, column=2)
        self.button3 = Button(main, text="NoPerson", command=self.onButton3)
        self.button3.grid(row=1, column=5)
        self.button4 = Button(main, text="Red Button", command=self.onButton4, fg="red")
        self.button4.grid(row=1, column=3)
        self.button5 = Button(main, text="Others", command=self.onButton5)
        self.button5.grid(row=1, column=6)
        self.saveb = Button(main, text="save", command=self.save)
        self.saveb.grid(row=2, column=3)
        self.skip = Button(main, text="skip", command=self.skip)
        self.skip.grid(row=2, column=4)
        self.aMerge = Button(main, text="Auto Merge", command=self.autoMerge)
        self.aMerge.grid(row=2, column=5)
        self.back = Button(main, text="back", command=self.onBack)
        self.back.grid(row=2, column=6)
        c1 = Checkbutton(main, text="Autosave", variable=self.as_var)
        c1.grid(row=2, column=0, sticky=W, columnspan=2)

    def pop(self, dup):
        return dup.iloc[self.count]

    def select_next_img(self):
        self.checked += 1
        if self.count <= self.total - 1:
            self.row = self.pop(self.duplicate)
            self.idx = self.row.name
            self.color = self.class_col[int(self.row[' Person_ID'])]
            frameNbr = int(self.row[0][:-4][(len(self.row[0][:-4]) - 8):])
            if self.current != frameNbr:
                self.cap.set(1, frameNbr)
                ret, self.image = self.cap.read()
                self.current = frameNbr
                self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
                self.image = Image.fromarray(self.image)
            self.img = ImageTk.PhotoImage(self.image)
            self.main.title(
                "Frame: " + str(frameNbr) + " | Done : " + str(self.count) + " | Total: " + str(len(self.duplicate)))
        else:
            self.save()
            print('Fertig')
            exit(0)

    def save(self):
        self.df.to_csv(self.path, index=False)
        with open(self.histPath, 'w') as f:
            for i in self.history:
                f.writelines(str(i) + "\n")

    # ----------------

    def create_transparent_rectangle(self, x1, y1, x2, y2, alpha, fill, **kwargs):
        alpha *= 255
        fill = root.winfo_rgb(fill) + (int(alpha),)
        image = Image.new('RGBA', (int(x2) - int(x1), int(y2) - int(y1)), fill)
        self.images.append(ImageTk.PhotoImage(image))
        self.canvas.create_image(x1, y1, image=self.images[-1], anchor='nw')
        self.bbox = self.canvas.create_rectangle(x1, y1, x2, y2, **kwargs)

    def updateFrame(self):
        self.count += 1
        self.select_next_img()
        # change image
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)
        self.images = []
        self.canvas.delete("rect")

        ################################################################################################################
        """Draw the yolo box associated with the Skeleton"""
        ################################################################################################################
        self.create_transparent_rectangle(self.row[-4], self.row[-3], self.row[-2], self.row[-1], fill=self.color,
                                          alpha=.1,
                                          tags="rect")
        ################################################################################################################

        ################################################################################################################
        """Draw the Skeleton bounding box """
        ################################################################################################################
        # store the x and y values of all joints in two sperate lists to later determin the min and max values for the bb
        x_values = [self.row[k - 2] for k in range(4, len(self.row) - 5, 3) if self.row[k] > 0]
        y_values = [self.row[k - 1] for k in range(4, len(self.row) - 5, 3) if self.row[k] > 0]

        self.create_transparent_rectangle(min(x_values), min(y_values), max(x_values), max(y_values), fill=self.color,
                                          alpha=.3,
                                          tags="rect")
        ################################################################################################################

        ################################################################################################################
        """Draw Skeleton Lines"""
        ################################################################################################################
        x_s = [int(self.row[k]) for k in range(2, len(self.row) - 6, 3)]
        y_s = [int(self.row[k]) for k in range(3, len(self.row) - 5, 3)]
        # joints = [self.df.columns[k] for k in range(3, len(self.df.columns) - 6 + 1, 3)]
        """
        for i, j, k in zip(x_s, y_s, joints):
            if i != 0 and j != 0:
                self.canvas.create_text(int(i), int(j) + 10, fill="darkblue", font="Times 20 italic bold", text=k,
                                        tags="rect")
        """
        for i, j in self.pairs:
            if x_s[i] != 0 and y_s[i] != 0 and x_s[j] != 0 and y_s[j] != 0:
                self.canvas.create_line(x_s[i], y_s[i], x_s[j], y_s[j], width=5, fill='black', tags="rect")
        ################################################################################################################
        if self.as_var.get() and self.count % 10 == 0:
            self.save()

    def onButton0(self):
        self.df.at[self.idx, ' Person_ID'] = 0.0
        self.history.add(self.idx)
        self.updateFrame()

    def onButton1(self):
        self.df.at[self.idx, ' Person_ID'] = 1.0
        self.history.add(self.idx)
        self.updateFrame()

    def onButton2(self):
        self.df.at[self.idx, ' Person_ID'] = 2.0
        self.history.add(self.idx)
        self.updateFrame()

    def onButton3(self):
        self.df.at[self.idx, ' Person_ID'] = 3.0
        self.history.add(self.idx)
        self.updateFrame()

    def onButton4(self):
        self.df.at[self.idx, ' Person_ID'] = 4.0
        self.history.add(self.idx)
        self.updateFrame()

    def onButton5(self):
        self.df.at[self.idx, ' Person_ID'] = 5.0
        self.history.add(self.idx)
        self.updateFrame()

    def onBack(self):
        self.count = max(0, self.count - 2)
        self.updateFrame()

    def autoMerge(self):
        self.save()
        self.df = run_correction(self.path)
        self.duplicate = self.df[self.df.duplicated(['# Image_ID', ' Person_ID'], keep=False)]
        self.duplicate = self.duplicate.sort_values(by=['# Image_ID', ' Person_ID'])
        self.duplicate = self.duplicate[self.duplicate[' Person_ID'] != 5]
        self.duplicate = self.duplicate.drop(self.history, errors='ignore')
        self.updateFrame()

    def skip(self):
        self.updateFrame()

    def onArrowD(self, *args):
        self.df.at[self.idx, ' Person_ID'] = 1.0
        self.history.add(self.idx)
        self.updateFrame()

    def onArrowG(self, *args):
        self.df.at[self.idx, ' Person_ID'] = 2.0
        self.history.add(self.idx)
        self.updateFrame()

    def onArrowB(self, *args):
        self.df.at[self.idx, ' Person_ID'] = 0.0
        self.history.add(self.idx)
        self.updateFrame()

    def onArrowR(self, *args):
        self.df.at[self.idx, ' Person_ID'] = 4.0
        self.history.add(self.idx)
        self.updateFrame()

    def onArrowNP(self, *args):
        self.df.at[self.idx, ' Person_ID'] = 3.0
        self.history.add(self.idx)
        self.updateFrame()

    def onArrowO(self, *args):
        self.df.at[self.idx, ' Person_ID'] = 5.0
        self.history.add(self.idx)
        self.updateFrame()

    def onArrowS(self, *args):
        self.history.add(self.idx)
        self.updateFrame()

    def onArrowBack(self, *args):
        self.count = max(0, self.count - 2)
        self.updateFrame()


# ----------------------------------------------------------------------


if __name__ == "__main__":
    session = 4
    cam = 2
    historyName = "S" + str(session) + "C" + str(cam)
    path = "E:\\Nurse_Data/s" + str(session) + "/OpenPose_southampton_" + str(session) + "_cam_" + str(
        cam) + "_fixed.csv"
    path = "E:\\Nurse_Data/s" + str(session) + "/OpenPose_southampton_" + str(session) + "_cam_" + str(
        cam) + ".csv"
    video_path = "E:/Nurse_videos/S" + str(session) + "/southampton " + str(session) + " cam " + str(cam) + ".mp4"
    root = Tk()
    MainWindow(root, path=path, video_path=video_path, historyName=historyName)
    root.mainloop()

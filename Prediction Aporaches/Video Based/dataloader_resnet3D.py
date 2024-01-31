import collections
import glob
import os
import cv2
import numpy
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import pickle as pkl
import albumentations as A
import random


class NurseDataset(Dataset):
    def __init__(self, image_path, window_path, yolo_path, set_type, session_list, transformation=None):
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        ################################################################################################################
        #                                                   INIT VARS                                                  #
        ################################################################################################################
        self.image_path = image_path  # Dataset/images
        self.window_path = window_path  # Dataset/images

        if transformation is None:
            self.transformation = A.Compose(
                [
                    A.CLAHE(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Lambda(p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.2),
                    A.RandomGamma(p=0.2),
                    A.HueSaturationValue(p=0.1),
                    A.Rotate(limit=2),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255
                    )
                ],
                additional_targets={'image' + str(i): 'image' for i in range(1, 25)})

        else:
            self.transformation = transformation  # toTensor
        self.norm = A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255
                )
            ],
            additional_targets={'image' + str(i): 'image' for i in range(1, 25)})
        self.set_type = set_type
        self.yolo_path = yolo_path
        self.new_shape = (112, 112)
        self.expand_yolobox = 30
        """transform = A.Compose(
        [A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.5),
         A.CLAHE(p=0.5),
         A.RandomBrightnessContrast(p=0.5),
         A.Lambda(p=0.5),
         A.MedianBlur(blur_limit=3, p=0.2),
         A.RandomGamma(p=0.2),
    
         A.ColorJitter(p=0.1),
         A.RGBShift(p=0.1),
         A.HueSaturationValue(p=0.1),
    
         A.ToGray(p=0),
         A.Blur(p=0),
         A.RandomCrop(width=int(np.size(image_in, 1) * 0.5), height=int(np.size(image_in, 0) * 0.5), p=0)]
         )"""
        ################################################################################################################
        #                                                     Paths                                                    #
        ################################################################################################################

        # train_window_paths = glob.glob(self.window_path + os.sep + 'train' + os.sep + 's*_cam*_pers*.txt')
        # val_window_paths = glob.glob(self.window_path + os.sep + 'validation' + os.sep + 's*_cam*_pers*.txt')
        # test_window_paths = glob.glob(self.window_path + os.sep + 'test' + os.sep + 's*_cam*_pers*.txt')

        self.all_imgs = {}
        first = session_list.pop(0)
        self.window_paths = glob.glob(self.window_path + os.sep + f's{first}_cam*_pers*.txt')
        all_imgs_file = open(self.image_path + f"s{first}_all_imgs.pkl", "rb")
        self.all_imgs[str(first)] = pkl.load(all_imgs_file)
        all_imgs_file.close()
        for i in session_list:
            self.window_paths += glob.glob(self.window_path + os.sep + f's{i}_cam*_pers*.txt')
            all_imgs_file = open(self.image_path + f"s{i}_all_imgs.pkl", "rb")
            self.all_imgs[str(i)] = pkl.load(all_imgs_file)
            all_imgs_file.close()

        ################################################################################################################
        #                                                   Load Data                                                  #
        ################################################################################################################
        self.windows_dict, self.lbl_dict, self.session_dict, self.crops_dict = self.load_data()
        ################################################################################################################

    def load_data(self):
        changelbl = {
            0: 2,
            1: 0,
            2: 1,
            3: 2,
        }
        windows_dict = {}
        lbl_dict = {}
        session_dict = {}
        crops_dict = {}

        idx = 0
        classes = []

        for session in self.window_paths:
            s = os.path.split(session)[1][1]
            c = os.path.split(session)[1][7]
            p = os.path.split(session)[1][-5]
            windows = np.loadtxt(session, delimiter=',')

            df = pd.read_csv(self.yolo_path + os.path.split(session)[1])
            df.set_index("frame", inplace=True)
            new_idx = np.arange(0, df.index[-1] + 1, 1)
            df = df.reindex(new_idx)
            df = df[df.columns[[0, 1, 2, 3, 7]]]
            pers = np.array(df.values)

            count = idx
            for start, end, _, _, _, _, _, _ in windows:
                crops_dict[count] = [[i, pers[i][0], pers[i][1], pers[i][2], pers[i][3]] for i in
                                     range(int(start), int(end) + 1) if not np.isnan(pers[i][0])]
                count += 1

            for window in windows:
                windows_dict[idx] = [window[0:6].astype(int), window[6]]
                lbl_dict[idx] = changelbl[int(window[7])]
                classes.append(changelbl[int(window[7])])
                session_dict[idx] = [s, c, p]
                idx += 1
        if self.set_type == 'train':
            counter = collections.Counter(classes)
            orderd_counter = collections.OrderedDict(sorted(counter.items()))
            class_sample_count = np.array(list(orderd_counter.values()))
            weight = 1. / class_sample_count
            weight = np.divide(weight, np.sum(weight))
            self.counter = torch.tensor(weight)

        return windows_dict, lbl_dict, session_dict, crops_dict

    def get_cw(self):
        if self.set_type == 'train':
            return self.counter.float()
        else:
            return np.nan

    def __len__(self):
        return len(self.windows_dict)

    def create_ausgabe(self, y_pred_list, name):
        if self.set_type == 'test':
            ret = []
            for idx, pred in enumerate(y_pred_list):
                start, end, _, _, _, _ = self.windows_dict[idx][0]
                session, cam, person = self.session_dict[idx]
                ret.append([int(person), int(start), int(end), int(pred)])
            path = os.getcwd() + os.sep + name + "_pred.txt"
            ret_np = np.array(ret, dtype=np.int8)
            np.savetxt(path, ret_np, delimiter=',')
            np.savetxt()

    def __getitem__(self, idx):
        start, end, minx, maxx, miny, maxy = self.windows_dict[idx][0]
        scale = self.windows_dict[idx][1]
        yolo_data = self.crops_dict[idx]
        labels = self.lbl_dict[idx]
        session, cam, person = self.session_dict[idx]

        frame_out = np.zeros((3, len(yolo_data), self.new_shape[1], self.new_shape[0]), dtype=np.uint8)
        """ imgs = {}
            for i in range(len(yolo_data)):
            image = self.all_imgs[session][f"s{session}_cam{cam}_pers{person}_{start + i}.jpg"]
            # image = Image.fromarray(image)
            if i > 0:
                imgs['image' + str(i)] = image.copy()
            else:
                imgs['image'] = image.copy()"""
        imgs = {('image' + str(i) if i > 0 else 'image'): self.all_imgs[session][
            f"s{session}_cam{cam}_pers{person}_{start + i}.jpg"].copy() for i in range(len(yolo_data))}
        # imgsdict = self.transformation(**{('image' + str(i) if i > 0 else 'image'): img for i, img in enumerate(imgs)})
        if self.set_type == 'test':
            imgsdict = self.norm(**imgs)
        else:
            imgsdict = self.transformation(**imgs)
        flip = False
        if random.random() < .5:
            flip = True
        for i, (key, img) in enumerate(imgsdict.items()):
            h, w, c, = img.shape
            y = max(0, int(yolo_data[i][2]) - self.expand_yolobox)
            x = max(0, int(yolo_data[i][1]) - self.expand_yolobox)
            y_1 = int((y - miny) * scale)
            x_1 = int((x - minx) * scale)
            #img_i = np.moveaxis(img, -1, 0)
            img_o = self.preprocess(np.uint8(img))
            frame_out[:, i, y_1:y_1 + h, x_1:x_1 + w] = img_o
            if flip:
                frame_out[:, i, y_1:y_1 + h, x_1:x_1 + w] = cv2.flip(frame_out[:, i, y_1:y_1 + h, x_1:x_1 + w], 1)
        data = torch.tensor(frame_out)
        sample = data, torch.tensor(labels)
        return sample

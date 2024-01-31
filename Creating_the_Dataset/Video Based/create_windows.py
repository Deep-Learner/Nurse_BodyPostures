import collections
import csv

import cv2
import numpy as np
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm


def yoloPos(yolo_pos_path, session, cam):
    yolo_pos_arr = [pd.read_csv(yolo_pos_path + f'\s{session}_cam_{cam}_pers_{pers}.txt') for pers in [0, 2, 4]]
    return_arr = []
    for i in range(3):
        df = yolo_pos_arr[i]
        df.set_index("frame", inplace=True)
        new_idx = np.arange(0, df.index[-1] + 1, 1)
        df = df.reindex(new_idx)
        df = df[df.columns[[0, 1, 2, 3, 7]]]
        return_arr.append(np.array(df.values))

    return return_arr[0], return_arr[1], return_arr[2]


def compute_windows(pers, window_size):
    count_no_detect = 0
    num_img_in_window = 0
    windowArray = []
    start_of_window = 0
    for frame_id, pos in enumerate(pers):
        if np.isnan(pos[0]):
            count_no_detect += 1
        if count_no_detect > 0:  # int(window_size/4):
            start_of_window = frame_id + 1
            count_no_detect = 0
            num_img_in_window = 0
        else:
            num_img_in_window += 1
            if num_img_in_window == window_size:
                count_no_detect = 0
                windowArray.append([start_of_window, frame_id])
                start_of_window = frame_id + 1
                num_img_in_window = 0
    windowArray = np.array(windowArray)
    # np.savetxt(f'E:\\Dataset\\Windows\\s{session}_cam{cam}_pers{person_ID}.txt', windowArray, delimiter=',')
    return windowArray


def compute_scale_per_window(windowArray, pers, width, height, person_ID, session, cam, new_shape, expand_yolobox, val,
                             dataset_loc, isAA):
    scaleList = []
    min_max_List = []
    window_info_array = []
    new_windowArray = []
    for start, end in windowArray:
        minx = np.inf
        maxx = 0
        miny = np.inf
        maxy = 0
        labels = []
        for i in range(start, end + 1):
            minx = min(minx, max(0, int(pers[i][0]) - expand_yolobox))
            maxx = max(maxx, min(width - 1, int(pers[i][2]) + expand_yolobox))
            miny = min(miny, max(0, int(pers[i][1]) - expand_yolobox))
            maxy = max(maxy, min(height - 1, int(pers[i][3]) + expand_yolobox))
            labels.append(pers[i][4])
        if (maxx - minx) > new_shape[0] and (maxy - miny) > new_shape[1]:
            scale = min(new_shape[0] / (maxx - minx), new_shape[1] / (maxy - miny))
        elif (maxx - minx) > new_shape[0]:
            scale = new_shape[0] / (maxx - minx)
        elif (maxy - miny) > new_shape[1]:
            scale = new_shape[1] / (maxy - miny)
        else:
            scale = 1
        if mode(labels)[0][0] == 10 and isAA:
            continue
        scaleList.append(scale)
        min_max_List.append([minx, maxx, miny, maxy])
        window_info_array.append(
            [int(start), int(end), int(minx), int(maxx), int(miny), int(maxy), scale, mode(labels)[0][0], ])
        new_windowArray.append([start, end])
    folder = 'Train'
    if session == val:
        folder = 'Val'
    np.savetxt(dataset_loc + f'Windows\\{folder}\\s{session}_cam_{cam}_pers_{person_ID}.txt', window_info_array,
               delimiter=',', fmt="%.8g")
    return scaleList, min_max_List, np.array(new_windowArray)


def creat_and_save_crpos(vidcap, windowArray, scaleP, pers, height, width, person_ID, expand_yolobox, dataset_loc,
                         session, cam, cut=False):
    pbar = tqdm(range(len(windowArray)), total=len(windowArray))

    for (start, end), scale in zip(windowArray, scaleP):
        pbar.update()
        if not cut:
            vidcap.set(1, start)
            if scale == 1:
                for i in range(start, end + 1):
                    _, image = vidcap.read()
                    y = max(0, int(pers[i][1]) - expand_yolobox)
                    yh = min(height - 1, int(pers[i][3]) + expand_yolobox)
                    x = max(0, int(pers[i][0]) - expand_yolobox)
                    xh = min(width - 1, int(pers[i][2]) + expand_yolobox)
                    img_out = image[y:yh, x:xh]
                    cv2.imwrite(dataset_loc + f'Crops\\S{session}\\s{session}_cam{cam}_pers{person_ID}_{i}.jpg',
                                img_out)

            else:
                for i in range(start, end + 1):
                    _, image = vidcap.read()

                    y = max(0, int(pers[i][1]) - expand_yolobox)
                    yh = min(height - 1, int(pers[i][3]) + expand_yolobox)
                    x = max(0, int(pers[i][0]) - expand_yolobox)
                    xh = min(width - 1, int(pers[i][2]) + expand_yolobox)
                    img_out = image[y:yh, x:xh]
                    img_out = cv2.resize(img_out, (0, 0), fx=scale, fy=scale)
                    cv2.imwrite(dataset_loc + f'Crops\\S{session}\\s{session}_cam{cam}_pers{person_ID}_{i}.jpg',
                                img_out)
    pbar.close()


def main(video_path, dataset_loc, yolo_pos_path, session, cam, expand_yolobox, new_shape, val, window_size, isAA):
    vidcap = cv2.VideoCapture(video_path)
    width = int(vidcap.get(3))  # float `width`
    height = int(vidcap.get(4))  # float `height

    pers0, pers2, pers4 = yoloPos(yolo_pos_path, session, cam)

    windowArrayP0 = compute_windows(pers0, window_size)
    windowArrayP2 = compute_windows(pers2, window_size)
    windowArrayP4 = compute_windows(pers4, window_size)

    scaleP0, min_max_P0, windowArrayP0 = compute_scale_per_window(windowArrayP0, pers0, width, height,
                                                                                  person_ID=0,
                                                                                  session=session, cam=cam,
                                                                                  new_shape=new_shape,
                                                                                  expand_yolobox=expand_yolobox,
                                                                                  val=val,
                                                                                  dataset_loc=dataset_loc, isAA=isAA)
    scaleP2, min_max_P2, windowArrayP2 = compute_scale_per_window(windowArrayP2, pers2, width, height,
                                                                                  person_ID=2,
                                                                                  session=session, cam=cam,
                                                                                  new_shape=new_shape,
                                                                                  expand_yolobox=expand_yolobox,
                                                                                  val=val,
                                                                                  dataset_loc=dataset_loc, isAA=isAA)
    scaleP4, min_max_P4, windowArrayP4 = compute_scale_per_window(windowArrayP4, pers4, width, height,
                                                                                  person_ID=4,
                                                                                  session=session, cam=cam,
                                                                                  new_shape=new_shape,
                                                                                  expand_yolobox=expand_yolobox,
                                                                                  val=val,
                                                                                  dataset_loc=dataset_loc, isAA=isAA)

    creat_and_save_crpos(vidcap, windowArrayP0, scaleP0, pers0, height, width, person_ID=0,
                         expand_yolobox=expand_yolobox, dataset_loc=dataset_loc, session=session, cam=cam)
    creat_and_save_crpos(vidcap, windowArrayP2, scaleP2, pers2, height, width, person_ID=2,
                         expand_yolobox=expand_yolobox, dataset_loc=dataset_loc, session=session, cam=cam)
    creat_and_save_crpos(vidcap, windowArrayP4, scaleP4, pers4, height, width, person_ID=4,
                         expand_yolobox=expand_yolobox, dataset_loc=dataset_loc, session=session, cam=cam)

    cv2.destroyAllWindows()


if __name__ == "__main__":

    for session in range(3, 7):
        # session = 7
        cam = 2
        video_path = f"E:\\Nurse_videos\S{session}\southampton {session} cam {cam}.MP4"
        # dataset_loc = 'D:\\Dataset\\'
        dataset_loc = 'D:\\DSTAA\\'
        # dataset_loc = 'D:\\D3\\'
        # yolo_pos_path = f"D:\Dataset\YOLO"
        yolo_pos_path = dataset_loc + "YOLO"

        cut = False
        start_frame = 11698
        end_frame = 36700
        window_size = 25
        expand_yolobox = 30
        new_shape = (112, 112)  # (x,y)
        val = 7
        if session == val:
            train_or_val = "val"  # "train" or "val"
        else:
            train_or_val = "train"  # "train" or "val
        main(video_path, dataset_loc, yolo_pos_path, session, cam, expand_yolobox, new_shape, val,
                           window_size,
                           isAA=True)

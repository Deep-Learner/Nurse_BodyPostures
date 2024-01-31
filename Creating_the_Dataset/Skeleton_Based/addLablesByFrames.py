import cv2 as cv
import pandas as pd
import openpyxl
import sys
import os
import json
from collections import Counter


def read_labels(path):
    df_dict = pd.read_excel(path, sheet_name=None, header=1, engine='openpyxl')
    for idx, key in df_dict.items():
        key = key.reindex(columns=['Frame', "position", 'body posture', 'AA'])
        key = key.dropna(subset=['Frame'], how='any')
        key = key.dropna(subset=["position", "body posture", "AA"], how='all')
        key = key.reset_index(drop=True)
        df_dict[idx] = key
    return df_dict['Blue'], df_dict['Green'], df_dict['Red']


def add_label(pose_data_path, labels_path, correction_value):
    """
    This function Takes the position data path, the labels path and the fps as its input
    Writes the position into the according columns and returns the new data Frame
    :param pose_data_path:
    :param labels_path:
    :param fps:
    :return:
    """
    df_pose = pd.read_csv(pose_data_path, engine='python')
    df_pose.assign(position="", bp="", AA="")
    df_pose.rename(index={'bp': 'body posture'})
    dfBlue, dfGreen, dfRed = read_labels(labels_path)

    for id, dfLabel in zip([0, 2, 4], [dfBlue, dfGreen, dfRed]):
        print(id)
        df = df_pose[df_pose[' Person_ID'] == id].copy()
        df['# Image_ID'] = [int(i[:-4][(len(i[:-4]) - 8):]) for i in df["# Image_ID"].values]
        start_frame = dfLabel.iloc[0]['Frame']
        end_frame = dfLabel.iloc[(len(dfLabel) - 1)]['Frame']
        current_frame = 0
        for idx, row in df.iterrows():
            frame_nr = row["# Image_ID"] + correction_value
            if frame_nr >= start_frame:
                if (frame_nr < end_frame):
                    while (frame_nr >= dfLabel.iloc[current_frame + 1]['Frame'] and frame_nr < end_frame):
                        current_frame += 1
                df_pose.at[idx, 'position'] = dfLabel.iloc[current_frame]['position']
                df_pose.at[idx, 'body posture'] = dfLabel.iloc[current_frame]['body posture']
                df_pose.at[idx, 'AA'] = dfLabel.iloc[current_frame]['AA']
    print(Counter(df_pose['body posture'].dropna()))
    df_pose.to_csv(pose_data_path[:-4] + '_labeled.csv', index=False)
    print(pose_data_path[:-4] + '_labeled.csv' + ' has been saved')
    return df_pose


def get_correction_value(path, cam):
    with open(path) as f:
        data = json.load(f)
    return data[str(cam)] - data[str(2)]


def start():
    session = 7

    cam = 2
    cap = cv.VideoCapture(
        "E:/Nurse_videos/S" + str(session) + "/southampton " + str(session) + " cam " + str(cam) + ".mp4")
    pose_data_path = "E:\\Nurse_Data/S" + str(session) + "/OpenPose_southampton_" + str(session) + "_cam_" + str(
        cam)+"_short" + ".csv"   #_fixed
    labels_path =r"E:\Nurse_Labels\Labels_session" + str(session) + '.xlsx'
    fps = cap.get(cv.CAP_PROP_FPS)
    print(fps)
    sync_path = 'SyncFiles/session_' + str(session) + '_SyncFile.json'
    correction_value = get_correction_value(sync_path, cam)
    add_label(pose_data_path, labels_path, correction_value)


if __name__ == "__main__":
    start()


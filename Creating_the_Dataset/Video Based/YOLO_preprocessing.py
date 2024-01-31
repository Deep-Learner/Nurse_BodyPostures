import json

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

"""session = 1
cam = 3
video_path = f"E:\\Nurse_videos\\S{session}\\southampton {session} cam {cam}.MP4"
yolo_data_path = f"E:\\YOLO\\S{session}\\southampton {session} cam {cam}_reclassified.txt"
labels_path = f"E:\\Nurse_Labels_2\\Labels_session{session}.xlsx"
#save_path= f'D:\\Dataset\\YOLO\\'
save_path= f'D:\\Cam3\\YOLO\\'
# yolo_pos_path = f"E:\Dataset\YOLO"
save_directory = 'D'"""


def get_correction_value(session, cam):
    path = f'E:\\Resnet3D\\SyncFiles\\session_{session}_SyncFile.json'
    with open(path) as f:
        data = json.load(f)
    return data[str(cam)] - data[str(2)]


def bb_intersection_over_union(M, N):
    area_M = (M[2] - M[0]) * (M[3] - M[1])
    area_N = (N[2] - N[0]) * (N[3] - N[1])
    area_of_intersection = max(0, min(M[2], N[2]) - max(M[0], N[0])) * max(0, min(M[3], N[3]) - max(M[1], N[1]))
    return float(area_of_intersection) / (area_M + area_N - area_of_intersection)

def yolo_fix2222():
    df = pd.read_csv(yolo_data_path, skiprows=1, header=None, sep=' ')
    df.columns = ["x1", "y1", "x2", "y2", "ID", "conf_1", "conf_2", "frame"]
    duplicate_grouped = df[df['ID'] % 2 == 0].groupby(['frame', 'ID'])
    for _, frame_df in duplicate_grouped:
        if len(frame_df) > 1:
            while (len(frame_df) > 1 and min(frame_df['conf_1'].values) < 0.6):
                pos = frame_df['conf_1'].argmin()
                idx = frame_df.iloc[pos].name
                df.at[idx, 'ID'] = 5
                frame_df = frame_df.drop(idx)
    duplicate_grouped = df[df['ID'] % 2 == 0].groupby(['frame', 'ID'])
    for _, frame_df in duplicate_grouped:
        if len(frame_df) == 2:
            # ioU = bb_intersection_over_union(frame_df.iloc[0][:4].values, frame_df.iloc[1][:4].values)
            # if ioU == 0 or ioU >= 0.90:
            pos = frame_df['conf_2'].argmin()
            idx = frame_df.iloc[pos].name
            df.at[idx, 'ID'] = 5
    df.to_csv(yolo_data_path[:-4] + "_fixed.txt", index=False)

def yolo_fix():
    df = pd.read_csv(yolo_data_path, skiprows=1, header=None, sep=' ')
    df.columns = ["x1", "y1", "x2", "y2", "ID", "conf_1", "conf_2", "frame"]
    duplicate_grouped = df[df['ID'] % 2 == 0].groupby(['frame', 'ID'])
    for _, frame_df in duplicate_grouped:
        if len(frame_df) > 1:
            while (len(frame_df) > 1 and min(frame_df['conf_1'].values) < 0.6):
                pos = frame_df['conf_1'].argmin()
                idx = frame_df.iloc[pos].name
                df.at[idx, 'ID'] = 5
                frame_df = frame_df.drop(idx)
    duplicate_grouped = df[df['ID'] % 2 == 0].groupby(['frame', 'ID'])
    for _, frame_df in duplicate_grouped:
        while len(frame_df) > 1:
            # ioU = bb_intersection_over_union(frame_df.iloc[0][:4].values, frame_df.iloc[1][:4].values)
            # if ioU == 0 or ioU >= 0.90:
            pos = frame_df['conf_2'].argmin()
            idx = frame_df.iloc[pos].name
            frame_df.drop(idx, inplace=True)
            df.at[idx, 'ID'] = 5
    df.to_csv(yolo_data_path[:-4] + "_fixed.txt", index=False)

def boxFilter():
    path = yolo_data_path[:-4] + "_fixed.txt"
    df = pd.read_csv(path, skiprows=1, header=None, sep=' |,')
    df.columns = ["x1", "y1", "x2", "y2", "ID", "conf_1", "conf_2", "frame"]
    window = 3
    columns = list(range(4))
    for j in [0, 2, 4]:
        df2 = df[df['ID'] == j].copy()
        print('person ', j)
        for idx, row in df2.iterrows():
            avg_window = df2[df2['frame'].between(row['frame'] - window, row['frame'] + window)]
            for column in columns:
                col = df2.columns[column]
                avg_value = avg_window[col].mean()
                df.at[idx, col] = avg_value
    df.to_csv(path[:-4] + '_bf.txt', index=False)


def smart_interpol():
    max_leangth = 3
    path = yolo_data_path[:-4] + "_fixed_bf.txt"
    df = pd.read_csv(path, skiprows=1, header=None, sep=' |,')
    df.columns = ["x1", "y1", "x2", "y2", "ID", "conf_1", "conf_2", "frame"]
    for person in [0, 2, 4]:
        df2 = df[df['ID'] == person].copy()
        df2 = df2[df2['frame'] < 38000]

        idxs = list(set(np.arange(df2['frame'].min(), 38000)) - set(df2['frame'].unique()))
        idxs.sort()
        consequtive_list = []
        last = idxs[0] - 1
        currentl = []
        for i in idxs:
            if i == last + 1:
                currentl.append(i)
            else:
                if len(currentl) <= max_leangth:
                    consequtive_list.extend(currentl.copy())
                currentl = [i]
            last = i
        if 0 < len(currentl) <= max_leangth:
            consequtive_list.extend(currentl.copy())

        f_x1 = interp1d(df2['frame'].values, df2['x1'].values, kind='linear', axis=- 1)
        f_x2 = interp1d(df2['frame'].values, df2['x2'].values, kind='linear', axis=- 1)
        f_y1 = interp1d(df2['frame'].values, df2['y1'].values, kind='linear', axis=- 1)
        f_y2 = interp1d(df2['frame'].values, df2['y2'].values, kind='linear', axis=- 1)

        ones = np.ones((len(consequtive_list)))
        dict = {"x1": f_x1(consequtive_list),
                "y1": f_y1(consequtive_list),
                "x2": f_x2(consequtive_list),
                "y2": f_y2(consequtive_list),
                "ID": ones * person,
                "conf_1": ones,
                "conf_2": ones,
                "frame": consequtive_list
                }
        df3 = pd.DataFrame(dict)
        df = pd.concat([df, df3], ignore_index=True)

    df.sort_values(by=['frame'], ignore_index=True, inplace=True)
    df.to_csv(path[:-4] + '_I.txt', index=False)


def split_yolo_and_add_label():
    path = yolo_data_path[:-4] + "_fixed_bf_I.txt"
    df = pd.read_csv(path, skiprows=1, header=None, sep=' |,')
    df.columns = ["x1", "y1", "x2", "y2", "ID", "conf_1", "conf_2", "frame"]

    df_dict = pd.read_excel(labels_path, sheet_name=None, header=1, engine='openpyxl')

    class_to_idx = {
        'turn': 0,
        'walk': 1,
        'bendover': 2,
        'stand': 3,
    }
    correction_value = get_correction_value(session, cam)
    for idx, key in df_dict.items():
        key = key.reindex(columns=['Frame', 'body posture'])
        key = key.dropna(subset=['Frame', "body posture"], how='any')
        key = key.reset_index(drop=True)
        key['body posture'].replace(class_to_idx, inplace=True)
        df_dict[idx] = key
    labels_df = [df_dict['Blue'], df_dict['Green'], df_dict['Red']]
    for person, person_labels in zip([0, 2, 4], labels_df):
        person_df = df[df['ID'] == person].copy()
        start_frame = person_labels.iloc[0]['Frame'] + correction_value
        end_frame = person_labels.iloc[(len(person_labels) - 1)]['Frame'] + correction_value
        current_frame = 0
        person_df = person_df[person_df['frame'].between(start_frame, end_frame - 1)]
        for idx, row in person_df.iterrows():
            frame_nr = row['frame']
            while (frame_nr >= (
                    person_labels.iloc[current_frame + 1]['Frame'] + correction_value) and frame_nr < end_frame - 1):
                current_frame += 1
            person_df.at[idx, 'body posture'] = person_labels.iloc[current_frame]['body posture']
        path = save_path + f's{session}_cam_{cam}_pers_{person}.txt'
        path = save_directory + path[1:]
        person_df.dropna(subset=["body posture"], how='any', inplace=True)
        person_df.to_csv(path, index=False)


if __name__ == "__main__":
    save_path = f'D:\\Cam3\\YOLO\\'
    save_directory = 'D'
    cam = 3
    # session = 1
    for session in range(1, 8):
        video_path = f"E:\\Nurse_videos\\S{session}\\southampton {session} cam {cam}.MP4"
        yolo_data_path = f"E:\\YOLO\\S{session}\\southampton {session} cam {cam}_reclassified.txt"
        labels_path = f"E:\\Nurse_Labels_2\\Labels_session{session}.xlsx"
        yolo_fix()
        boxFilter()
        smart_interpol()
        split_yolo_and_add_label()

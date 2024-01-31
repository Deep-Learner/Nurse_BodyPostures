import pandas as pd
import cv2 as cv


def load_data(path):
    df = pd.read_csv(path, engine='python')
    return df


def resize_bb(df):
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


def get_duplicates(df):
    duplicates = df[df.duplicated(['# Image_ID', ' Person_ID'], keep=False)]
    duplicates = duplicates.sort_values(by=['# Image_ID', ' Person_ID'])
    return duplicates


def pop(dup):
    return dup.drop(labels=dup.index[0], axis=0), dup.iloc[0]


def extract_features(row):
    idx = row.name
    frame_nbr = row['# Image_ID']
    pers_id = row[' Person_ID']

    position = [row['x1'], row['y1'], row['x2'], row['y2']]
    return idx, frame_nbr, pers_id, position


def cv_visualization(df, write=False, session=5, cam=2):
    df['# Image_ID'] = [int(i[:-4][(len(i[:-4]) - 8):]) for i in df.iloc[:, 0]]
    video_path = "D:/Nurse_videos/cam_" + str(session) + "/southampton " + str(session) + " cam " + str(cam) + ".mp4"
    color = {
        0.0: (255, 0, 0),
        1.0: (136, 0, 136),
        2.0: (0, 255, 0),
        3.0: (255, 255, 0),
        4.0: (0, 0, 255),
        5.0: (0, 127, 255)
    }
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    if write:
        out = cv.VideoWriter(video_path[:-4] + '_predictied_bodyposture_cnn.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps,
                             (int(cap.get(3)), int(cap.get(4))))
    first_frame = (df.iloc[0]["# Image_ID"])

    current_frame_data = first_frame
    current_frame_data_pos = 0
    current_video_frame = 0

    last_frame = df.iloc[len(df) - 1]["# Image_ID"]

    flag = False
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            while (current_video_frame > current_frame_data and current_video_frame < last_frame):
                current_frame_data_pos += 1
                current_frame_data = df.iloc[current_frame_data_pos]["# Image_ID"]
            while current_video_frame == current_frame_data and not flag:
                x = int(float(df.iloc[current_frame_data_pos]["x1"]))
                y = int(float(df.iloc[current_frame_data_pos]["y1"]))
                z = int(float(df.iloc[current_frame_data_pos]["x2"]))
                w = int(float(df.iloc[current_frame_data_pos]["y2"]))
                col = color[df.iloc[current_frame_data_pos][' Person_ID']]
                cv.rectangle(frame, (x, y), (z, w), col)
                if current_video_frame < last_frame:
                    current_frame_data_pos += 1
                    current_frame_data = df.iloc[current_frame_data_pos]["# Image_ID"]
                else:
                    flag = True
            if first_frame <= current_video_frame <= last_frame and write:
                out.write(frame)
            cv.imshow('Frames', frame)
            current_video_frame += 1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    if write:
        out.release()
    cv.destroyAllWindows()


def bb_intersection_over_union(M, N):
    area_M = (M[2] - M[0]) * (M[3] - M[1])
    area_N = (N[2] - N[0]) * (N[3] - N[1])
    area_of_intersection = max(0, min(M[2], N[2]) - max(M[0], N[0])) * max(0, min(M[3], N[3]) - max(M[1], N[1]))
    return float(area_of_intersection) / (area_M + area_N - area_of_intersection)


def run_correction(path):
    df = load_data(path)
    store_df = df.copy()
    df['# Image_ID'] = [int(i[:-4][(len(i[:-4]) - 8):]) for i in df.iloc[:, 0]]
    df = resize_bb(df)
    duplicates = get_duplicates(df)
    duplicates = duplicates[duplicates[' Person_ID'] != 5]
    duplicates, row = pop(duplicates)
    dup_idx, dup_frame_nbr, dup_pers_id, dup_position = extract_features(row)
    long_term_memory = []
    short_term_memory = []
    frame = 0

    for current_idx, row in df.iterrows():
        _, current_frame_nbr, current_pers_id, current_position = extract_features(row)
        if frame != current_frame_nbr:
            if len(long_term_memory) >= 8:
                long_term_memory.pop(0)
            if len(short_term_memory) > 0:
                long_term_memory.append(short_term_memory.copy())
            frame = current_frame_nbr
            short_term_memory = []

        if current_frame_nbr == dup_frame_nbr:
            if len(long_term_memory) > 0:
                memory = [item for frame in long_term_memory for item in frame]
                while current_frame_nbr == dup_frame_nbr:
                    IoUs = [bb_intersection_over_union(pos[1], dup_position) for pos in memory]
                    max_val = max(IoUs)
                    if max_val >= 0.75:
                        max_val_id = memory[IoUs.index(max_val)][0]
                        store_df.at[dup_idx, ' Person_ID'] = max_val_id
                        if current_idx == dup_idx:
                            current_pers_id = max_val_id
                    if len(duplicates) == 0:
                        return store_df
                    duplicates, dup_row = pop(duplicates)
                    dup_idx, dup_frame_nbr, dup_pers_id, dup_position = extract_features(dup_row)
            else:
                if len(duplicates) == 0:
                    return store_df
                duplicates, dup_row = pop(duplicates)
                dup_idx, dup_frame_nbr, dup_pers_id, dup_position = extract_features(dup_row)
        short_term_memory.append([current_pers_id, current_position])


def run_correction_CC(path):
    df = load_data(path)
    store_df = df.copy()
    df['# Image_ID'] = [int(i[:-4][(len(i[:-4]) - 8):]) for i in df.iloc[:, 0]]
    df = resize_bb(df)
    duplicates = get_duplicates(df)
    duplicates = duplicates[duplicates[' Person_ID'] != 5]
    duplicates, row = pop(duplicates)
    dup_idx, dup_frame_nbr, dup_pers_id, dup_position = extract_features(row)
    long_term_memory = []
    short_term_memory = []
    frame = 0

    for current_idx, row in df.iterrows():
        _, current_frame_nbr, current_pers_id, current_position = extract_features(row)
        if frame != current_frame_nbr:
            if len(long_term_memory) >= 8:
                long_term_memory.pop(0)
            long_term_memory.append(short_term_memory.copy())
            frame = current_frame_nbr
            short_term_memory = []

        if current_frame_nbr == dup_frame_nbr and len(long_term_memory) > 0:
            memory = [item for frame in long_term_memory for item in frame]
            while current_frame_nbr == dup_frame_nbr:
                distances = [bb_intersection_over_union(pos[1], dup_position) for pos in memory]
                max_val = max(distances)
                if max_val >= 0.75:
                    max_val_id = memory[distances.index(max_val)][0]
                    df.at[dup_idx, ' Person_ID'] = max_val_id
                    store_df.at[dup_idx, ' Person_ID'] = max_val_id
                    if current_idx == dup_idx:
                        current_pers_id = max_val_id
                if len(duplicates) == 0:
                    return df
                duplicates, dup_row = pop(duplicates)
                dup_idx, dup_frame_nbr, dup_pers_id, dup_position = extract_features(dup_row)

        short_term_memory.append([current_pers_id, current_position])


if __name__ == "__main__":
    session = 4
    cam = 2
    vis = False
    write = True
    update = False
    path = "D:\\Nurse_Data/s" + str(session) + "/OpenPose_southampton_" + str(session) + "_cam_" + str(cam) + ".csv"
    if update:
        path = "D:\\Nurse_Data/s" + str(session) + "/OpenPose_southampton_" + str(session) + "_cam_" + str(
            cam) + "_fixed_m2.csv"
    df = run_correction(path=path)
    if vis:
        cv_visualization(df, write=False, session=session, cam=cam)
    if write:
        if update:
            df.to_csv(path, index=False)
        else:
            df.to_csv(path[:-4] + '_fixed_m2.csv', index=False)

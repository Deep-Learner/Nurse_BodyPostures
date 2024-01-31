import np as np
import pandas as pd
import operator

def merge(path):
    print("Load Csv")
    print("-----------------------------------------------")
    df = pd.read_csv(path, engine='python')
    print("Find Duplicates")
    print("-----------------------------------------------")
    print("Merge Duplicates")
    print("-----------------------------------------------")
    conf_col = [k for k in range(4, len(df.columns) - 5, 3)]
    duplicate_grouped = df[df[' Person_ID'] % 2 == 0].groupby(['# Image_ID', ' Person_ID'])
    for _, frame_df in duplicate_grouped:
        if len(frame_df) > 1:
            new_row = frame_df.iloc[0][:2].values.tolist()
            for col in conf_col:
                row = np.argmax(frame_df.iloc[:, col].values)
                new_row.extend(frame_df.iloc[row][col - 2:col + 1].values)
            new_row.extend(frame_df.iloc[0][-7:])
            df.at[frame_df.index[0]] = new_row
            df.drop(frame_df.index[1:], inplace=True)
    print("Update orignial df")
    print("-----------------------------------------------")
    df.to_csv(path[:-4] + '_merged_new.csv', index=False)

if __name__ == "__main__":
    session = 4
    cam = 2
    path = "E:\\Nurse_Data/s" + str(session) + "/OpenPose_southampton_" + str(session) + "_cam_" + str(
        cam) + "_fixed_labeled_interpol.csv"# "_short_labeled.csv" #
    merge(path=path)

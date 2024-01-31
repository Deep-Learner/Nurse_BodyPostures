from collections import Counter

from matplotlib import pyplot as plt
from sklearn import preprocessing

import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import sys
from scipy.stats import mode
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn.functional
import seaborn as sns
import SyncFiles


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3))
        self.mp_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.bn_1 = nn.BatchNorm2d(2)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.mp_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.bn_2 = nn.BatchNorm2d(16)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.mp_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn_3 = nn.BatchNorm2d(32)

        self.fc_1 = nn.Linear(64, 16)
        self.bn_4 = nn.BatchNorm2d(64)
        self.bn_5 = nn.BatchNorm1d(16)
        self.fc_2 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn_1(x)
        x = self.conv_1(x)
        x = self.mp_1(x)
        x = self.relu(x)

        x = self.bn_2(x)
        x = self.conv_2(x)
        x = self.mp_2(x)
        x = self.relu(x)

        x = self.bn_3(x)
        x = self.conv_3(x)
        x = self.mp_3(x)

        x = self.relu(x)
        x = self.bn_4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.bn_5(x)
        x = self.fc_2(x)
        return x


def acc_per_epoch(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return torch.round(acc * 100)


def load_all_dfs(cam):
    df_s1 = pd.read_csv(
        "E:\\Nurse_Data/s1/OpenPose_southampton_1_cam_" + str(cam) + "_short_labeled_merged_new.csv",
        engine='python')
    df_s2 = pd.read_csv(
        "E:\\Nurse_Data/s2/OpenPose_southampton_2_cam_" + str(cam) + "_short_labeled_merged_new.csv",
        engine='python')
    df_s3 = pd.read_csv(
        "E:\\Nurse_Data/s3/OpenPose_southampton_3_cam_" + str(cam) + "_fixed_labeled_merged_new.csv",
        engine='python')
    df_s4 = pd.read_csv(
        "E:\\Nurse_Data/s4/OpenPose_southampton_4_cam_" + str(cam) + "_fixed_labeled_merged_new.csv",
        engine='python')
    df_s5 = pd.read_csv(
        "E:\\Nurse_Data/s5/OpenPose_southampton_5_cam_" + str(cam) + "_fixed_labeled_merged_new.csv",
        engine='python')
    df_s6 = pd.read_csv(
        "E:\\Nurse_Data/s6/OpenPose_southampton_6_cam_" + str(cam) + "_fixed_labeled_merged_new.csv",
        engine='python')
    df_s7 = pd.read_csv(
        "E:\\Nurse_Data/s7/OpenPose_southampton_7_cam_" + str(cam) + "_short_labeled_merged_new.csv",
        engine='python')
    return df_s1, df_s2, df_s3, df_s4, df_s5, df_s6, df_s7


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


def normalize_bb_center(df):
    center_x = [max(1, int(abs(i + j) / 2)) for i, j in zip(df.loc[:, 'x1'].values, df.loc[:, 'x2'].values)]
    center_y = [max(1, int(abs(i + j) / 2)) for i, j in zip(df.loc[:, 'y1'].values, df.loc[:, 'y2'].values)]
    center = [center_x, center_y]
    len_x = [max(1, int(abs(j - i) / 2)) for i, j in zip(df.loc[:, 'x1'].values, center_x)]
    len_y = [max(1, int(abs(j - i) / 2)) for i, j in zip(df.loc[:, 'y1'].values, center_y)]
    len_xy = [len_x, len_y]
    for i in range(2, len(df.columns) - 8):
        if (i - 2) % 3 != 2:
            df[df.columns[i]] = (df.iloc[:, i].values - center[((i - 2) % 3)]) / (len_xy[((i - 2) % 3)])
    return df


def preprocessing(df, norm):
    resize_bb(df)
    df = df[df[' Person_ID'] % 2 == 0]
    df = df[df["body posture"] != 6]
    df = df.dropna(subset=["body posture"], how='any')

    df['# Image_ID'] = [int(i[:-4][(len(i[:-4]) - 8):]) for i in df.iloc[:, 0]]
    df = df.drop(['position', 'AA'], axis=1)
    if norm:
        df = normalize_bb_center(df)

    cols = df.columns.tolist()[:-5] + df.columns.tolist()[-1:]
    df = df[cols]
    return df


def conv_to_window(df, window_start=0, normalize=""):
    window_leangth = 25
    flag = True
    count = window_start
    window_to_sample = {window_start: []}
    input = []
    label = []
    window_to_idx = {window_start: []}
    feat = int((len(df.columns) - 3) / 3)
    frames_to_persons = [df.loc[df[' Person_ID'] == j] for j in df[' Person_ID'].unique()]
    for person_idx, person in enumerate(frames_to_persons):
        current_start_frame = int(person.iloc[0, 0]) - (int(person.iloc[0, 0]) % window_leangth)
        current_window = np.zeros((2, feat, window_leangth))
        window_count = 0
        nn_input = []
        nn_label = []
        min_x = []
        min_y = []
        for row_idx, row in person.iterrows():
            flag = True
            if window_count < window_leangth:
                window_to_sample[count].append(row.loc['body posture'])
                window_to_idx[count].append(row_idx)
                flag = False
            else:
                if (mode(window_to_sample[count])[1][0] / len(window_to_sample[count])) >= 0.75:
                    # Normalize the window
                    if normalize == "window":
                        min_x = min(min_x)
                        min_y = min(min_y)
                        current_window[0, :, :] = (current_window[0, :, :] - min_x) / (
                                np.amax(current_window[0, :, :]) - min_x)
                        current_window[1, :, :] = (current_window[1, :, :] - min_y) / (
                                np.amax(current_window[1, :, :]) - min_y)
                    elif normalize == "total size":
                        current_window[0, :, :] = (current_window[0, :, :] / 1280)
                        current_window[1, :, :] = (current_window[1, :, :] / 720)
                    ###########################################################################
                    nn_input.append([count, current_window])
                    nn_label.append([count, mode(window_to_sample[count])[0][0]])
                    count += 1
                    window_count = 0
                    current_window = np.zeros((2, feat, window_leangth))
                    current_start_frame += window_leangth
                    window_to_sample[count] = [row.loc['body posture']]
                    window_to_idx[count] = [row_idx]
                    min_x = []
                    min_y = []
                else:
                    window_to_sample[count] = window_to_sample[count][1:]
                    window_to_idx[count] = window_to_idx[count][1:]
                    min_x = min_x[1:]
                    min_y = min_y[1:]

                    current_window = current_window[:, :, 1:]
                    column_to_be_added = np.zeros((2, feat, 1))
                    current_window = np.append(current_window, column_to_be_added, axis=2)
                    window_count -= 1

            min_x.append(min([row[k] for k in range(2, len(df.columns) - 2, 3) if row[k + 2] > 0]))
            min_y.append(min([row[k] for k in range(3, len(df.columns) - 1, 3) if row[k + 1] > 0]))
            current_window[0, :, window_count] = [row[k] for k in range(2, len(df.columns) - 2, 3)]
            current_window[1, :, window_count] = [row[k] for k in range(3, len(df.columns) - 1, 3)]
            window_count += 1
        if not flag:
            if (mode(window_to_sample[count])[1][0] / len(window_to_sample[count])) >= 0.75:
                # Normalize the window
                if normalize == "window":
                    min_x = min(min_x)
                    min_y = min(min_y)
                    current_window[0, :, :] = (current_window[0, :, :] - min_x) / (
                            np.amax(current_window[0, :, :]) - min_x)
                    current_window[1, :, :] = (current_window[1, :, :] - min_y) / (
                            np.amax(current_window[1, :, :]) - min_y)
                elif normalize == "total size":
                    current_window[0, :, :] = (current_window[0, :, :] / 1280)
                    current_window[1, :, :] = (current_window[1, :, :] / 720)
                ###########################################################################
                nn_input.append([count, current_window])
                nn_label.append([count, mode(window_to_sample[count])[0][0]])
                count += 1
            window_to_sample[count] = []
            window_to_idx[count] = []
        input.append(nn_input.copy())
        label.append(nn_label.copy())
    return input, label, window_to_sample, window_to_idx

def create_data_results(class_to_idx, cam=2, normalize=""):

    df1, df2, df3, df4, df5, df6, df7 = load_all_dfs(cam=cam)
    # Replace class by numeric value
    ###################################################################################################################
    ################################################ Pre-processing ###################################################
    ###################################################################################################################
    df1['body posture'].replace(class_to_idx, inplace=True)
    df2['body posture'].replace(class_to_idx, inplace=True)
    df3['body posture'].replace(class_to_idx, inplace=True)
    df4['body posture'].replace(class_to_idx, inplace=True)
    df5['body posture'].replace(class_to_idx, inplace=True)
    df6['body posture'].replace(class_to_idx, inplace=True)
    df7['body posture'].replace(class_to_idx, inplace=True)

    df1 = preprocessing(df1, normalize == "center")
    df2 = preprocessing(df2, normalize == "center")
    df3 = preprocessing(df3, normalize == "center")
    df4 = preprocessing(df4, normalize == "center")
    df5 = preprocessing(df5, normalize == "center")
    df6 = preprocessing(df6, normalize == "center")
    df7 = preprocessing(df7, normalize == "center")
    ###################################################################################################################
    ################################################ split into sets ###################################################
    ###################################################################################################################
    d1_input, d1_label, d1_windows, _ = conv_to_window(df1, normalize=normalize)
    d2_input, d2_label, d2_windows, _ = conv_to_window(df2, d1_input[2][-1][0] + 1, normalize=normalize)
    d3_input, d3_label, d3_windows, _ = conv_to_window(df3, d2_input[2][-1][0] + 1, normalize=normalize)
    d4_input, d4_label, d4_windows, _ = conv_to_window(df4, d3_input[2][-1][0] + 1, normalize=normalize)
    d5_input, d5_label, d5_windows, _ = conv_to_window(df5, d4_input[2][-1][0] + 1, normalize=normalize)
    d6_input, d6_label, d6_windows, _ = conv_to_window(df6, d5_input[2][-1][0] + 1, normalize=normalize)
    d7_input, d7_label, d7_windows, _ = conv_to_window(df7, d6_input[2][-1][0] + 1, normalize=normalize)

    window_dict = d1_windows.copy()
    windows = [d2_windows, d3_windows, d4_windows, d5_windows, d6_windows, d7_windows]
    for w in windows:
        window_dict.update(w)

    sets = [[d1_input, d1_label], [d3_input, d3_label], [d4_input, d4_label], [d5_input, d5_label],
            [d6_input, d6_label]]

    test_input = []
    test_label = []
    val_input = []
    val_label = []
    for i in d2_input: test_input.extend(i)
    for i in d2_label: test_label.extend(i)

    for i in d7_input: val_input.extend(i)
    for i in d7_label: val_label.extend(i)
    train_input = []
    train_label = []

    for i, l in sets:
        for j in i: train_input.extend(j)
        for j in l: train_label.extend(j)
    p = np.random.permutation(len(train_input))
    train_input = [train_input[i] for i in p]
    train_label = [train_label[i] for i in p]
    ret = [[train_input, train_label, test_input, test_label, val_input, val_label, window_dict]]
    return ret


def create_loader(train_input, train_label, test_input, test_label, val_input, val_label, num_classes, batch_size,
                  scale=True, wrs=False):
    ###################################################################################################################
    ############################################## Create data Loader #################################################
    ###################################################################################################################

    train_x = np.array([j for i, j in train_input])

    val_x = [j for i, j in val_input]
    test_x = [j for i, j in test_input]

    train_y = np.array([j for i, j in train_label])
    val_y = [j for i, j in val_label]
    test_y = [j for i, j in test_label]

    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y).long()
    val_x = torch.Tensor(val_x)
    val_y = torch.Tensor(val_y).long()
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y).long()

    ###################################################################################################################
    ############################################## weighted sampler ###################################################
    ###################################################################################################################
    if wrs:
        target_list = []
        for t in train_y:
            target_list.append(t)

        class_distribution = Counter(train_y)
        target_list = torch.tensor(target_list)
        target_list = target_list[torch.randperm(len(target_list))]
        class_count = [i for i in class_distribution.values()]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        class_weights_all = class_weights[target_list]
        weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )

    ###################################################################################################################
    class_sample_count = np.array(
        [max(len(np.where(train_y == t)[0]), 1) for t in range(num_classes)])
    print(class_sample_count)
    weight = 1. / class_sample_count
    weight = np.divide(weight, np.sum(weight))
    weight = torch.Tensor(weight)
    ###################################################################################################################
    train_dataset = TensorDataset(train_x, train_y)
    if wrs:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return train_loader, test_loader, val_loader, weight


def train(model, train_loader, val_loader, name, weight, epochs=200, learning_rate=10e-4,
          l2_lambda=0.0005, use_scheduler=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###################################################################################################################
    ###############################################  Create the model  ################################################
    ###################################################################################################################

    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    ######## Adam w !!!!!
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.95, step_size=5)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, patience=5, verbose=True)
    ###################################################################################################################
    ##################################################### Train ########################################################
    ###################################################################################################################
    print("###################### Start of training phase ######################")
    min_val_loss = np.inf
    patience = 25

    for e in range(1, epochs + 1):
        losses = []
        # beginning of the training
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = acc_per_epoch(y_train_pred, y_train_batch)
            if use_scheduler:
                losses.append(train_loss.item())

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = acc_per_epoch(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        print("#####################################################################################")
        print(f'Epoch count: {e + 0:03} ')
        print("#####################################################################################")
        print(f'Training Accuracy: {train_epoch_acc / len(train_loader):.3f}')
        print(f'Validation Accuracy: {val_epoch_acc / len(val_loader):.3f}')
        print(f'Training Loss: {train_epoch_loss / len(train_loader):.5f}')
        print(f'Validation Loss: {val_epoch_loss / len(val_loader):.5f}')
        print("#####################################################################################")
        ################################################################################################################
        ############################################### LR scheduler  ##################################################
        ################################################################################################################
        if use_scheduler:
            mean_loss = sum(losses) / len(losses)
            scheduler.step(mean_loss)
        ################################################################################################################
        ############################################### Early sopping ##################################################
        ################################################################################################################
        if val_epoch_loss < min_val_loss:
            torch.save(model.state_dict(), r'net/' + name + '_net.pt')
            no_improvent = 0
            min_val_loss = val_epoch_loss
        else:
            no_improvent += 1
        if no_improvent == patience:
            print('Early stopping!')
            break
    return model


def test(test_loader, name, num_classes, idx_to_class, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(r'net/' + name + '_net.pt'))
    ###################################################################################################################
    ##################################################### Train #######################################################
    ###################################################################################################################
    print("########################################### Start of test phase ###########################################")
    y_pred_list = []
    y_actual_list = []
    cm = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        model.eval()
        val_epoch_acc = 0
        for X_batch, y_test_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_batch = y_test_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.nn.functional.softmax(y_test_pred, dim=1)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy()[0])
            y_actual_list.append(int(y_test_batch[0]))
            cm[int(y_pred_tags), int(y_test_batch[0])] += 1

            val_acc = acc_per_epoch(y_test_pred, y_test_batch)
            val_epoch_acc += val_acc.item()

    print('#########################')
    print(val_epoch_acc / len(test_loader))
    print('#########################')

    y_actual = [idx_to_class[int(i)] for i in y_actual_list]
    y_pred = [idx_to_class[int(i)] for i in y_pred_list]
    labels = ['walk', 'bendover', 'stand']

    confusion_matrix_heatmap(y_actual, y_pred, labels, name=name, show=True, plt_report=False)


def confusion_matrix_heatmap(y_actual, y_pred, labels, name="conf_matrtix", show=False, plt_report=True):
    f={"fontsize":22}
    if plt_report:
        fig, axs = plt.subplots(2, figsize=(8, 8), gridspec_kw={'height_ratios': [4, 1]})
        ax = axs[0]
        ax_table = axs[1]
        ax_table.axis("off")
    else:
        fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.axis('equal')

    ax.grid(False)
    conf_matrix = confusion_matrix(y_actual, y_pred, normalize="true", labels=labels)
    conf_matrix2 = confusion_matrix(y_actual, y_pred, normalize=None, labels=labels)

    conf_matrix3 = [[str(round(conf_matrix[i][j] * 100, 2)) + "%\n(" + str(conf_matrix2[i][j]) + ")" for j in
                     range(conf_matrix.shape[1])] for i in range(conf_matrix.shape[0])]
    conf_matrix_plot = sns.heatmap(conf_matrix, annot=conf_matrix3, cmap=plt.cm.viridis, fmt='', vmin=0, vmax=1,
                                       square=True, ax=ax, annot_kws=f,cbar=False)
    conf_matrix_plot.set_xticklabels(labels,fontdict=f, va="center")
    conf_matrix_plot.set_yticklabels(labels,fontdict=f, va="center")

    ax.set_xlabel("Predicted",fontdict=f)
    ax.set_ylabel("Actual",fontdict=f)
    #legend = ax.collections[0].colorbar
    #legend.ax.tick_params(labelsize=f['fontsize'])

    if plt_report:
        cr = classification_report(y_actual, y_pred, output_dict=True)
        cr = pd.DataFrame(cr).transpose()
        cr = cr.round(2)
        report_table = ax_table.table(cellText=cr.values,
                                      rowLabels=cr.index,
                                      colLabels=cr.columns, rowColours=["#F5EACB"] * len(cr.index),
                                      colColours=["#BDD3D2"] * len(cr.columns),
                                      cellLoc='center', rowLoc='center',
                                      loc='center')
        for i in range(3):
            report_table[(len(cr.index) - 2, i)].set_facecolor("#B2B2B2")
            report_table[(len(cr.index) - 2, i)]._text.set_text("")
    ax.autoscale()
    fig.tight_layout()
    plt.savefig('Plots/' + name + '_2.pdf')
    if show:
        plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    class_to_idx = {
        'turn': 2,
        'walk': 0,
        'bendover': 1,
        'stand': 2,
    }

    idx_to_class = {
        0: 'walk',
        1: 'bendover',
        2: 'stand',
    }

    epochs = 200
    learning_rate = 10e-4
    l2_lambda = 0.0005
    num_classes = len(idx_to_class)
    batch_size = 32
    normalize = "center"  # either "window" , "total size" or "center"
    #normalize = "window"  # either "window" , "total size" or "center"
    name = 'CNN_Resnet'
    name = name + "_" + normalize

    ###################################################################################################################
    ################################################ -------------- ###################################################
    ###################################################################################################################

    wsets = create_data_results(class_to_idx, normalize=normalize)
    idx = 0
    for wset in wsets:
        train_input, train_label, test_input, test_label, val_input, val_label, window_dict = wset
        test_y = [i for i, j in test_label]
        train_loader, test_loader, val_loader, weight = create_loader(train_input, train_label, test_input, test_label,
                                                                      val_input, val_label, num_classes, batch_size,
                                                                      wrs=True)

        useResnet = True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if useResnet:
            model = torchvision.models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            model = CustomCNN(num_classes=num_classes)

        model.to(device)

        train(model, train_loader, val_loader, name + str(idx), weight, l2_lambda=l2_lambda,learning_rate=learning_rate)
        win_to_idx = 0
        test(test_loader, name + str(idx), num_classes, idx_to_class, model)
        idx += 1

from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from Resnet3DModel import my3DResNet
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import glob
import sys
import os

name = "r3D_results_64batch"
window_path = os.getcwd() + os.sep + "dataset" + os.sep + "windows"  # + os.sep
# video_path = os.sep + "ds" + os.sep + "videos" + os.sep + "Nurse" + os.sep + "David" + os.sep
image_path = os.sep + "ds" + os.sep + "images" + os.sep + "Nurse" + os.sep + "David" + os.sep + 'fixed' + os.sep
yolo_path = os.getcwd() + os.sep + "dataset" + os.sep + "yolo_files" + os.sep

from dataloader_resnet3D import NurseDataset


def confusion_matrix_heatmap(y_actual, y_pred, labels, file_name, show=False, plt_report=True):
    fig, axs = plt.subplots(2, figsize=(8, 8), gridspec_kw={'height_ratios': [4, 1]})
    ax = axs[0]
    ax_table = axs[1]

    ax.axis('equal')
    ax_table.axis("off")
    ax.grid(False)
    conf_matrix = confusion_matrix(y_actual, y_pred, normalize="true", labels=labels)
    conf_matrix2 = confusion_matrix(y_actual, y_pred, normalize=None, labels=labels)

    conf_matrix3 = [[str(round(conf_matrix[i][j] * 100, 2)) + "%\n(" + str(conf_matrix2[i][j]) + ")" for j in
                     range(conf_matrix.shape[1])] for i in range(conf_matrix.shape[0])]
    conf_matrix_plot = sns.heatmap(conf_matrix, annot=conf_matrix3, cmap=plt.cm.viridis, fmt='', vmin=0, vmax=1,
                                   square=True, ax=ax)
    conf_matrix_plot.set_xticklabels(labels)
    conf_matrix_plot.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

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
    #plt.savefig('Plots/' + name + '.pdf')
    plt.savefig(os.getcwd() + os.sep + "results_" + file_name + '.png')
    plt.savefig(os.getcwd() + os.sep + "results_" + file_name + '.pdf')
    if show:
        plt.show()

def train_full():
    learning_rate = 0.0002504212685672718
    l2_lambda = 0.14798787204153013
    gamma = 0.5150646365196545
    batch_size = 64
    np.random.seed(12)
    torch.manual_seed(14)
    use_scheduler = True
    epochs = 1000
    #learning_rate = 10e-4
    #l2_lambda = 0.05
    #batch_size = 32
    num_classes = 3
    train_sessions = [1, 3, 4, 5, 6]
    val_sessions = [7]
    test_sessions = [2]
    #gamma = 0.95

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = my3DResNet(numOut=num_classes)

    model.to(device)

    train_Dataset = NurseDataset(image_path, window_path, yolo_path, set_type='train', session_list=train_sessions)
    class_weights = train_Dataset.get_cw()
    class_weights = class_weights.to(device)
    train_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)

    val_Dataset = NurseDataset(image_path, window_path, yolo_path, set_type='val', session_list=val_sessions)
    val_loader = torch.utils.data.DataLoader(val_Dataset, batch_size=1)

    test_Dataset = NurseDataset(image_path, window_path, yolo_path, set_type='test', session_list=test_sessions)
    test_loader = torch.utils.data.DataLoader(test_Dataset, batch_size=1)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=5)

    accumulate = 2
    min_val_loss = np.inf
    patience = 25
    for e in range(epochs + 1):
        train_epoch_loss = 0
        train_epoch_acc = 0
        losses = []
        model.train()
        for i, (train_batch_input, train_batch_label) in enumerate(train_loader):
            train_batch_input = train_batch_input.float()
            train_batch_input, train_batch_label = train_batch_input.to(device), train_batch_label.to(device)

            train_pred = model(train_batch_input)
            train_loss = criterion(train_pred, train_batch_label)
            train_acc = acc_per_epoch(train_pred, train_batch_label)
            if use_scheduler:
                losses.append(train_loss.item())
            train_loss.backward()

            if i % accumulate == 0 and i > 0:
                optimizer.step()
                optimizer.zero_grad()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            for val_batch_input, val_batch_label in val_loader:
                val_batch_input = val_batch_input.float()
                val_batch_input, val_batch_label = val_batch_input.to(device), val_batch_label.to(device)

                y_val_pred = model(val_batch_input)

                val_loss = criterion(y_val_pred, val_batch_label)
                val_acc = acc_per_epoch(y_val_pred, val_batch_label)

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
        ############################################### Early sopping ##################################################
        ################################################################################################################
        if use_scheduler:
            mean_loss = sum(losses) / len(losses)
            scheduler.step(mean_loss)
        if val_epoch_loss < min_val_loss:
            torch.save(model.state_dict(), os.getcwd() + f'{name}_net.pt')
            no_improvent = 0
            min_val_loss = val_epoch_loss
        else:
            no_improvent += 1
        if no_improvent == patience:
            print('Early stopping!')
            break
    model.load_state_dict(torch.load(os.getcwd() + f'{name}_net.pt'))
    y_pred_list = []
    y_actual_list = []
    with torch.no_grad():
        model.eval()
        val_epoch_acc = 0

        for test_batch_input, test_batch_label in test_loader:
            test_batch_input = test_batch_input.float()
            test_batch_input = test_batch_input.to(device)
            test_pred = model(test_batch_input)
            test_pred = torch.nn.functional.softmax(test_pred, dim=1)
            test_pred = test_pred.detach().cpu()
            _, y_pred_tags = torch.max(test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy()[0])
            y_actual_list.append(int(test_batch_label[0]))
            val_acc = acc_per_epoch(test_pred, test_batch_label)
            val_epoch_acc += val_acc.item()
    """class_to_idx = {
        'turn': 0,
        'walk': 1,
        'bendover': 2,
        'stand': 3,
    }"""
    idx_to_class = {
        0: 'walk',
        1: 'bendover',
        2: 'stand',
    }
    y_actual = [idx_to_class[int(i)] for i in y_actual_list]
    y_pred = [idx_to_class[int(i)] for i in y_pred_list]

    labels = ['walk', 'bendover', 'stand']
    confusion_matrix_heatmap(y_actual, y_pred, labels, file_name=name, show=False, plt_report=True)


def acc_per_epoch(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return torch.round(acc * 100)


train_full()

from sklearn.metrics import classification_report
from scipy.optimize import differential_evolution
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

name = "run_all_sessions_no_turn_s2_test_2"
window_path = os.getcwd() + os.sep + "dataset" + os.sep + "windows"  # + os.sep
# video_path = os.sep + "ds" + os.sep + "videos" + os.sep + "Nurse" + os.sep + "David" + os.sep
image_path = os.sep + "ds" + os.sep + "images" + os.sep + "Nurse" + os.sep + "David" + os.sep + 'fixed' + os.sep
yolo_path = os.getcwd() + os.sep + "dataset" + os.sep + "yolo_files" + os.sep

from dataloader_resnet3D import NurseDataset


def train_full(hyper_parameter):
    learning_rate, l2_lambda = hyper_parameter
    epochs = 1000

    batch_size = 32
    num_classes = 3
    train_sessions = [1, 3, 4, 5, 6]
    val_sessions = [7]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = my3DResNet(numOut=num_classes)

    model.to(device)

    train_Dataset = NurseDataset(image_path, window_path, yolo_path, set_type='train', session_list=train_sessions)
    class_weights = train_Dataset.get_cw()
    class_weights = class_weights.to(device)
    train_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)

    val_Dataset = NurseDataset(image_path, window_path, yolo_path, set_type='val', session_list=val_sessions)
    val_loader = torch.utils.data.DataLoader(val_Dataset, batch_size=1)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    accumulate = 2
    min_val_loss = np.inf
    patience = 25
    for e in range(epochs + 1):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for i, (train_batch_input, train_batch_label) in enumerate(train_loader):
            train_batch_input = train_batch_input.float()
            train_batch_input, train_batch_label = train_batch_input.to(device), train_batch_label.to(device)

            train_pred = model(train_batch_input)
            train_loss = criterion(train_pred, train_batch_label)
            train_acc = acc_per_epoch(train_pred, train_batch_label)
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

        sys.stdout.write("[%-60s] %d%%" % ('=' * int(60 * (e) / epochs), (100 * (e) / epochs)))
        sys.stdout.flush()
        print()
        print(
            f'Epoch {e + 0:03}: | Training Loss: {train_epoch_loss / len(train_loader):.5f} | Validation Loss: {val_epoch_loss / len(val_loader):.5f}')
        print(
            f'Training Acc: {train_epoch_acc / len(train_loader):.3f}| Validation Acc: {val_epoch_acc / len(val_loader):.3f}')
        ################################################################################################################
        ############################################### Early sopping ##################################################
        ################################################################################################################
        if val_epoch_loss < min_val_loss:
            torch.save(model.state_dict(), os.getcwd() + f'{name}_net.pt')
            no_improvent = 0
            min_val_loss = val_epoch_loss
        else:
            no_improvent += 1
        if no_improvent == patience:
            print('Early stopping!')
            break
    print('done')
    return min_val_loss


def hyperparameter_search():
    learning_rate = (0.000001, 0.001)
    l2_lambda = (0.00005, 0.5)
    boundaries = [learning_rate] + [l2_lambda]
    solver = differential_evolution(train_full, boundaries, strategy='best1bin', popsize=15, mutation=0.5,
                                    recombination=0.7, tol=0.01, seed=2020)
    best_hyperparams = solver.x
    min_loss = solver.fun

    print(f"Converged hyperparameters: learning_rate={best_hyperparams[0]},l2_lambda = {best_hyperparams[1]}")
    print(f"Minimum Loss: {min_loss}")


def acc_per_epoch(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return torch.round(acc * 100)


hyperparameter_search()

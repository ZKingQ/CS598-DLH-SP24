import numpy as np
import pandas as pd
import time
import json
import datetime
import math

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

matplotlib.use('agg')
seed = 42

label_dict = {
    # Controls
    'n': 0,
    # Chirrhosis
    'cirrhosis': 1,
    # Colorectal Cancer
    'cancer': 1, 'small_adenoma': 0,
    # IBD
    'ibd_ulcerative_colitis': 1, 'ibd_crohn_disease': 1,
    # T2D and WT2D
    't2d': 1,
    # Obesity
    'leaness': 0, 'obesity': 1,
}


def loadData(data_dir, filename, dtype=None):
    feature_string = ''
    if filename.split('_')[0] == 'abundance':
        feature_string = "k__"
    if filename.split('_')[0] == 'marker':
        feature_string = "gi|"
    # read file
    filename = data_dir + filename
    if not os.path.isfile(filename):
        print("FileNotFoundError: File {} does not exist".format(filename))
        exit()
    raw = pd.read_csv(filename, sep='\t', index_col=0, header=None)

    # select rows having feature index identifier string
    X = raw.loc[raw.index.str.contains(feature_string, regex=False)].T

    # get class labels
    Y = raw.loc['disease']
    Y = Y.replace(label_dict)

    # train and test split
    X_train, X_test, y_train, y_test = train_test_split(X.values.astype(dtype), Y.values.astype('int'), test_size=0.2,
                                                        random_state=seed, stratify=Y.values)
    print("Train data shape: ", X_train.shape)
    print("Test data shape: ", X_test.shape)
    return X_train, X_test, y_train, y_test


data_dir = '../data/marker/'
data_string = 'marker_Cirrhosis.txt'
X_train, X_test, y_train, y_test = loadData(data_dir, data_string, dtype='float32')

trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                                          batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                                         batch_size=32, shuffle=False)





optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

losses = []
for epoch in range(200):

    """ model training """
    model.train()
    cur_rec_loss = []
    for batch_idx, (data, _) in enumerate(trainloader):
        data = data.to(device)
        rec = model(data)
        loss = model.loss_func(rec, data.reshape(data.shape[0], -1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_rec_loss.append(loss.item())
    losses.append(np.mean(cur_rec_loss))

    """ model evaluation """
    with torch.no_grad():
        test_loss = []
        for batch_idx, (data, _) in enumerate(testloader):
            data = data.to(device)
            rec = model(data)
            loss = model.loss_func(rec, data)
            test_loss.append(loss.item())

    if epoch % 10 == 0:
        print(f"-- epoch {epoch} --, train MSE: {np.mean(cur_rec_loss)}, test MSE: {np.mean(test_loss)}")

model = AutoEncoder(input_dim=X_train.shape[1]).to(device)

X_train = torch.tensor(X_train).to(device)
X_test = torch.tensor(X_test).to(device)

X_train_encoded = model.encoder(X_train).cpu().detach().numpy()
X_test_encoded = model.encoder(X_test).cpu().detach().numpy()


def get_metrics(clf, is_svm=False):
    y_true, y_pred = y_test, clf.predict(X_test_encoded)
    y_prob = 0
    y_prob = clf.predict_proba(X_test_encoded)
    # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
    metrics = {
        'AUC': roc_auc_score(y_true, y_prob),
        'ACC': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'F1_score': f1_score(y_true, y_pred),
    }
    return metrics


# SVM

clf = SVC(kernel='linear', probability=True, random_state=seed)
clf.fit(X_train_encoded, y_train)
print(get_metrics(clf, is_svm=True))
# Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_encoded, y_train)
print(get_metrics(clf))
# Multi-layer Perceptron
clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
clf.fit(X_train_encoded, y_train)
print(get_metrics(clf))

import numpy as np
import pandas as pd
import time
import datetime
import math
import os
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
from torch.utils.data import DataLoader, TensorDataset
from ae_models import AutoEncoder, CAE, VAE, ShallowAutoEncoder, EarlyStopper

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


def load_data(data_dir, filename, dtype=None):
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


def train_vae(model, train_loader, test_loader, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rec_loss, kl_loss = [], []
    early_stopper = EarlyStopper(patience=5, delta=0.0)
    for epoch in range(1, max_epoch_num + 1):

        """ model training """
        model.train()
        cur_rec_loss, cur_kl_loss = [], []
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            rec, mu, std = model(data)
            loss, err, kl = model.loss_func(rec, data.reshape(-1, model.input_dim), mu, std)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_rec_loss.append(err.item())
            cur_kl_loss.append(kl.item())

        rec_loss.append(np.mean(cur_rec_loss))
        kl_loss.append(np.mean(cur_kl_loss))

        """ model evaluation """
        with torch.no_grad():
            test_loss = []
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(device)
                rec, mu, std = model(data)
                _, mse, _ = model.loss_func(rec, data.reshape(data.shape[0], -1), mu, std)
                test_loss.append(mse.item())
            if early_stopper(np.mean(test_loss)):
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"-- epoch {epoch} --, train MSE: {np.mean(cur_rec_loss)}, train KL: {np.mean(cur_kl_loss)}, test MSE: {np.mean(test_loss)}")
    return model


def train_ae(model, train_loader, test_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

    losses = []
    early_stopper = EarlyStopper(patience=5, delta=0.0)
    for epoch in range(1, max_epoch_num + 1):

        """ model training """
        model.train()
        cur_rec_loss = []
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            rec = model(data)
            loss = model.loss_func(rec, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_rec_loss.append(loss.item())
        losses.append(np.mean(cur_rec_loss))

        """ model evaluation """
        with torch.no_grad():
            test_loss = []
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(device)
                rec = model(data)
                loss = model.loss_func(rec, data)
                test_loss.append(loss.item())
            if early_stopper(np.mean(test_loss)):
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch == 1 or epoch % 10 == 0:
            print(f"-- epoch {epoch} --, train MSE: {np.mean(cur_rec_loss)}, test MSE: {np.mean(test_loss)}")

    return model


def get_metrics(clf, X_test_encoded, y_test):
    y_true, y_pred = y_test, clf.predict(X_test_encoded)
    y_prob = clf.predict_proba(X_test_encoded)[:, 1]
    # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
    metrics = {
        'AUC': roc_auc_score(y_true, y_prob),
        'ACC': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'F1_score': f1_score(y_true, y_pred),
    }
    return metrics


def classification_with_pytorch_model(model, X_train, y_train, X_test, y_test, device, is_vae=False):
    model = model.to(device)
    # check if X_train is tensor already
    if not torch.is_tensor(X_train):
        X_train = torch.tensor(X_train).to(device)
        X_test = torch.tensor(X_test).to(device)

    if is_vae:
        X_train_encoded = model.encoder(X_train)[0].cpu().detach().numpy()
        X_test_encoded = model.encoder(X_test)[0].cpu().detach().numpy()
    else:
        X_train_encoded = model.encoder(X_train).cpu().detach().numpy()
        X_test_encoded = model.encoder(X_test).cpu().detach().numpy()
    X_train_encoded = np.reshape(X_train_encoded, (X_train_encoded.shape[0], -1))
    X_test_encoded = np.reshape(X_test_encoded, (X_test_encoded.shape[0], -1))
    return classification(X_train_encoded, y_train, X_test_encoded, y_test)


def run_sub_classification(method, X_train_encoded, y_train, X_test_encoded, y_test):
    clf = ""
    hyper_parameters = {}
    # hyper-parameter grids for classifiers
    rf_hyper_parameters = [{'n_estimators': [s for s in range(100, 1001, 200)],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_leaf': [1, 2, 3, 4, 5],
                            'criterion': ['gini', 'entropy']
                            }, ]
    svm_hyper_parameters = [{'C': [2 ** s for s in range(-5, 6, 2)], 'kernel': ['linear']},
                            {'C': [2 ** s for s in range(-5, 6, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],
                             'kernel': ['rbf']}]
    mlp_hyper_parameters = [{'hidden_layer_sizes': [(64, 64), (128, 128), (256, 256), (512, 512)],
                             'alpha': [2 ** s for s in range(-15, 1, 2)]}]

    if method == 'SVM':
        clf = SVC(kernel='linear', probability=True)
        hyper_parameters = svm_hyper_parameters
    elif method == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=100)
        hyper_parameters = rf_hyper_parameters
    elif method == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=1000)
        hyper_parameters = mlp_hyper_parameters
    clf = GridSearchCV(clf, param_grid=hyper_parameters,
                                cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                                verbose=1, n_jobs=2)
    clf.fit(X_train_encoded, y_train)
    metrics = get_metrics(clf, X_test_encoded, y_test)
    metrics['classification_method'] = method
    print(metrics)
    print("Best hyper-parameters: ", clf.best_params_)
    metrics['hyper_parameters'] = clf.best_params_
    return metrics


# Classification by using the transformed data
def classification(X_train_encoded, y_train, X_test_encoded, y_test):
    methods = ['MLP', 'SVM', 'Random Forest']
    metrics_list = []
    for method in methods:
        classification_start_time = time.time()
        metrics = run_sub_classification(method, X_train_encoded.copy(), y_train.copy(), X_test_encoded.copy(),
                                         y_test.copy())
        classification_end_time = time.time()
        print(method, "Classification time: ", classification_end_time - classification_start_time)
        metrics['classification_time'] = classification_end_time - classification_start_time
        metrics_list.append(metrics)
    return metrics_list


def run_model_training_and_classification(model_type, model, X_train, X_test, y_train, y_test, train_loader, test_loader, device):
    path = './cached_model/' + global_data_identifier + '_' + model_type + '_model.pth'
    print("Checking model path: ", path)
    if cache_trained_model and os.path.exists(path):
        model = torch.load(path, map_location=device)
    else:
        if model_type == 'vae':
            model = train_vae(model, train_loader, test_loader, device)
        else:
            model = train_ae(model, train_loader, test_loader, device)
        if cache_trained_model:
            torch.save(model, path)
    if train_only:
        return []
    return classification_with_pytorch_model(model, X_train, y_train, X_test, y_test, device)


def run_shallow_ae_experiment(X_train, X_test, y_train, y_test, train_loader, test_loader, device):
    model = ShallowAutoEncoder(input_dim=X_train.shape[1]).to(device)
    return run_model_training_and_classification('shallow_ae', model, X_train, X_test, y_train, y_test, train_loader, test_loader, device)


def run_ae_experiment(X_train, X_test, y_train, y_test, train_loader, test_loader, device):
    model = AutoEncoder(input_dim=X_train.shape[1]).to(device)
    return run_model_training_and_classification('ae', model, X_train, X_test, y_train, y_test, train_loader, test_loader, device)


def run_cae_experiment(X_train, X_test, y_train, y_test, device):
    # one_side_dim = int(math.sqrt(X_train.shape[1])) + 1
    one_side_dim = 348
    enlarged_dim = one_side_dim ** 2
    X_train = np.column_stack((X_train, np.zeros((X_train.shape[0], enlarged_dim - X_train.shape[1]))))
    X_test = np.column_stack((X_test, np.zeros((X_test.shape[0], enlarged_dim - X_test.shape[1]))))
    # reshape
    X_train = np.reshape(X_train, (len(X_train), one_side_dim, one_side_dim, 1))
    X_test = np.reshape(X_test, (len(X_test), one_side_dim, one_side_dim, 1))
    # to float32
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = torch.tensor(X_train).to(device)
    X_test = torch.tensor(X_test).to(device)
    X_train = X_train.transpose(1, 3)
    X_test = X_test.transpose(1, 3)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32, shuffle=False)
    model = CAE().to(device)
    return run_model_training_and_classification('cae', model, X_train, X_test, y_train, y_test, train_loader, test_loader, device)


def run_vae_experiment(X_train, X_test, y_train, y_test, train_loader, test_loader, device):
    model = VAE(input_dim=X_train.shape[1]).to(device)
    return run_model_training_and_classification('vae', model, X_train, X_test, y_train, y_test, train_loader, test_loader, device)


def run_pca_experiment(X_train, X_test, y_train, y_test):
    if train_only:
        return []
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return classification(X_train_pca, y_train, X_test_pca, y_test)


def run_rp_experiment(X_train, X_test, y_train, y_test):
    if train_only:
        return []
    rp = GaussianRandomProjection(eps=0.5)
    rp.fit(X_train)
    X_train_rp = rp.transform(X_train)
    X_test_rp = rp.transform(X_test)
    return classification(X_train_rp, y_train, X_test_rp, y_test)


def run_plain_experiment(X_train, X_test, y_train, y_test):
    if train_only:
        return []
    return classification(X_train, y_train, X_test, y_test)


def run_sub_experiment_on_data(model_type, X_train, X_test, y_train, y_test, train_loader, test_loader, device):
    metrics_list = []
    if model_type == 'shallow_ae':
        metrics_list = run_shallow_ae_experiment(X_train, X_test, y_train, y_test, train_loader, test_loader, device)
    elif model_type == 'ae':
        metrics_list = run_ae_experiment(X_train, X_test, y_train, y_test, train_loader, test_loader, device)
    elif model_type == 'cae':
        metrics_list = run_cae_experiment(X_train, X_test, y_train, y_test, device)
    elif model_type == 'vae':
        metrics_list = run_vae_experiment(X_train, X_test, y_train, y_test, train_loader, test_loader, device)
    elif model_type == 'pca':
        metrics_list = run_pca_experiment(X_train, X_test, y_train, y_test)
    elif model_type == 'rp':
        metrics_list = run_rp_experiment(X_train, X_test, y_train, y_test)
    elif model_type == 'plain':
        metrics_list = run_plain_experiment(X_train, X_test, y_train, y_test)
    return metrics_list


def run_experiment_on_data(data_dir, data_string):
    X_train, X_test, y_train, y_test = load_data(data_dir, data_string, dtype='float32')
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                             batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    all_metrics_df_on_current_data = pd.DataFrame()
    model_types = ['cae', 'shallow_ae', 'ae', 'vae', 'pca', 'rp', 'plain']
    for model_type in model_types:
        print("Running experiment on model: ", model_type)
        experiment_start_time = time.time()
        metrics_list = run_sub_experiment_on_data(model_type, X_train, X_test, y_train, y_test, train_loader,
                                                  test_loader, device)
        experiment_end_time = time.time()
        print("Experiment time: ", experiment_end_time - experiment_start_time)
        if train_only:
            continue

        for metrics in metrics_list:
            metrics['experiment_time'] = experiment_end_time - experiment_start_time
            metrics['data'] = data_string
            metrics['model'] = model_type
            metrics['seed'] = seed
            metrics['max_epoch_num'] = max_epoch_num
            metrics['device'] = device
            print(metrics)
            # save metrics
        metrics_df = pd.DataFrame(metrics_list)
        all_metrics_df_on_current_data = pd.concat([all_metrics_df_on_current_data, metrics_df], ignore_index=True)
    return all_metrics_df_on_current_data


def run_experiment():
    data_dir = './data/marker/'
    # main_metrics_df = run_experiment_on_data(data_dir, "marker_Cirrhosis.txt")
    data_list = os.listdir(data_dir)
    for data_string in data_list:
        global global_data_identifier
        global_data_identifier = data_string.split('.')[0]
        if data_string == 'marker_Cirrhosis.txt' or data_string == 'marker_Colorectal.txt' or data_string == 'marker_IBD.txt':
            continue
        main_metrics_df = pd.DataFrame()
        print("Running experiment on data: ", data_string)
        all_metrics_df_on_current_data = run_experiment_on_data(data_dir, data_string)
        if train_only:
            continue
        main_metrics_df = pd.concat([main_metrics_df, all_metrics_df_on_current_data], ignore_index=True)
        csv_file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_metrics.csv'
        main_metrics_df.to_csv(csv_file_name, index=False)
        print("Metrics saved to file: ", csv_file_name)


max_epoch_num = 2000
train_only = False
cache_trained_model = True
global_data_identifier = ''

run_experiment()

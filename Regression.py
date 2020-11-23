"""
This script collects the results from all the algorithms on all the 100 regression datasets in a
10-fold cross validation with an inner 3-fold cross validation for parameter tuning.
"""

import os
import csv
import numpy as np
import pandas as pd
from time import time, perf_counter
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# candidate algorithms
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from pycobra.cobra import Cobra
from pycobra.ewa import Ewa
from boruta import BorutaPy


def get_time_string(time_in_seconds):
    """
    Coverts the input time in seconds into the same time expressed in days, hours, minutes and seconds.
    :param time_in_seconds: to print
    :return: string
    """
    time_string = '%.1f(sec)' % (time_in_seconds % 60)
    if time_in_seconds >= 60:
        time_in_seconds /= 60
        time_string = '%d(min) %s' % (time_in_seconds % 60, time_string)
        if time_in_seconds >= 60:
            time_in_seconds /= 60
            time_string = '%d(hour) %s' % (time_in_seconds % 24, time_string)
            if time_in_seconds >= 24:
                time_in_seconds /= 24
                time_string = '%d(day) %s' % (time_in_seconds, time_string)
    return time_string


analyze_model = True  # for extracting the illustration examples shown in the report

num_folds = 10
random_search_iters = 50
random_search_cv = 3  # num of inner cross validation folds
models = [  # rs_params indicates the parameter space to be explored in the Random Search
    {'name': 'AdaBoost', 'model': AdaBoostRegressor(DecisionTreeRegressor(random_state=1), random_state=1),
     'rs_params': {'n_estimators': randint(5, 100), 'base_estimator__ccp_alpha': uniform(0, 0.1)}},

    {'name': 'Cobra', 'model': Cobra},

    {'name': 'Ewa', 'model': Ewa},

    {'name': 'Boruta', 'model': RandomForestRegressor(random_state=1),
     'rs_params': {'n_estimators': randint(5, 50), 'ccp_alpha': uniform(0, 0.01)}},
]

# prepare results log
datasets_in_log = {}  # for not computing an iteration if it's already in log
if not os.path.exists('results/results.csv'):
    with open('results/results.csv', 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        header = ['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]', 'Hyper-Parameters Values',
                  'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2_score',
                  'explained_variance_score', 'Training Time', 'Inference Time']
        writer.writerow(header)
    with open('results/progress_log.txt', 'w') as file:
        file.write('Progress:\n')
else:  # load already logged results to avoid redundancy of computations to shorten runtime
    for dataset_name, log_dataset in pd.read_csv('results/results.csv').groupby('Dataset Name'):
        folds_in_dataset = {}
        datasets_in_log[dataset_name] = folds_in_dataset
        for fold, log_fold in log_dataset.groupby('Cross Validation [1-10]'):
            folds_in_dataset[fold] = set(log_fold['Algorithm Name'])

files = os.listdir(os.fsencode('datasets'))
avg_runtime, iteration, iterations = 0, 0, len(files) * len(models) * num_folds  # only ETA calculation
for dataset_idx, file in enumerate(files):

    # load and pre-process dataset
    dataset_name = os.fsdecode(file)[:-4]
    dataset = pd.read_csv('datasets/%s.csv' % dataset_name)
    dataset = dataset.fillna(dataset.mean())  # fill nan values
    target_col = dataset.columns[-1]
    X = pd.get_dummies(dataset.drop(columns=target_col)).to_numpy()  # one-hot encode categorical features
    y = dataset[target_col].to_numpy()
    X = MinMaxScaler().fit_transform(X)
    y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).ravel()

    # start k-fold cross validation
    folds = list(KFold(n_splits=num_folds, shuffle=True, random_state=1).split(dataset))
    for fold_idx, fold in enumerate(folds):
        # organize samples for this fold
        indexes_train, indexes_test = fold
        X_test, y_test = X[indexes_test], y[indexes_test]
        X_train, y_train = X[indexes_train], y[indexes_train]

        for model_idx, model in enumerate(models):
            iteration += 1
            try:  # check if log already contains this iteration
                if model['name'] not in datasets_in_log[dataset_name][fold_idx + 1]:
                    raise KeyError()
            except KeyError:  # if not, run iteration
                start_time = int(time() * 1000)

                if model['name'] in ['Cobra', 'Ewa']:
                    # fit model
                    best_model = model['model'](random_state=1)
                    start_time_train = perf_counter()
                    # the inner cross validation is performed internally in set_epsilon() or set_beta()
                    if model['name'] == 'Cobra':
                        best_model.set_epsilon(X_epsilon=X_train, y_epsilon=y_train, grid_points=random_search_iters)
                        best_parameters = {'epsilon': best_model.epsilon}
                    else:  # Ewa
                        best_model.set_beta(X_beta=X_train, y_beta=y_train)
                        best_parameters = {'beta': best_model.beta}
                    best_model.fit(X_train, y_train)
                    runtime_train = perf_counter() - start_time_train
                else:
                    if model['name'] == 'Boruta':
                        print('\ntraining boruta...\n')
                        feature_selection = BorutaPy(RandomForestRegressor(n_jobs=-1, max_depth=7, random_state=1),
                                                     random_state=1)
                        # the inner cross validation was performed on the base inducers
                        feature_selection.fit(X_train, y_train)
                        accept_idx = []
                        # get the indexes of the columns to be kept
                        for idx, value in enumerate(feature_selection.support_):
                            if value:
                                accept_idx.append(idx)
                        if len(accept_idx) > 0:  # if boruta decided to throw away all columns, ignore it
                            backup = [X_train, X_test]  # for running the other algorithms on this fold
                            # remove features not selected by boruta
                            X_test = X_test[:, accept_idx]
                            X_train = X_train[:, accept_idx]
                    # fit model
                    best_model = RandomizedSearchCV(model['model'], model['rs_params'], random_state=1,
                                                    n_iter=random_search_iters, cv=random_search_cv)
                    start_time_train = perf_counter()
                    best_model.fit(X_train, y_train)
                    runtime_train = perf_counter() - start_time_train
                    best_parameters = best_model.best_params_

                # measure inference time
                X_inference = X_test[np.random.randint(0, len(X_test), 1000)]  # normalize to 1000 samples
                start_time_test = perf_counter()
                best_model.predict(X_inference)
                runtime_test = perf_counter() - start_time_test

                # compute metrics and prepare log entry
                y_pred = best_model.predict(X_test)
                row = [dataset_name, model['name'], fold_idx + 1, best_parameters,
                       metrics.mean_squared_error(y_test, y_pred),
                       metrics.mean_absolute_error(y_test, y_pred),
                       metrics.median_absolute_error(y_test, y_pred),
                       metrics.r2_score(y_test, y_pred),
                       metrics.explained_variance_score(y_test, y_pred),
                       runtime_train, runtime_test]

                if model['name'] == 'Boruta':  # return removed features
                    X_train, X_test = backup

                # save entry to log
                with open('results/results.csv', 'a', newline='') as log_file:
                    writer = csv.writer(log_file)
                    writer.writerow(row)

                if analyze_model:  # only for extracting the illustration examples in the report
                    e = best_model.epsilon
                    val_len = len(best_model.X_l_)
                    test_len = len(X_test)
                    df_analysis = pd.DataFrame({
                        'set': ['val'] * val_len + ['test'] * test_len,
                        'i': list(range(1, val_len + 1)) + list(range(1, test_len + 1)),
                        'y': list(best_model.y_l_) + list(y_test),
                        'pred': [''] * val_len + list(y_pred),
                    })
                    for name, estimator in best_model.estimators_.items():
                        preds_val = list(best_model.machine_predictions_[name])
                        preds_test = list(estimator.predict(X_test))
                        df_analysis[name] = preds_val + preds_test
                        for idx, y_t in enumerate(preds_test):
                            accept_col = [1 if np.abs(y_v - y_t) <= e else 0 for y_v in preds_val]
                            accept_col += [''] * test_len
                            df_analysis['%s accept %d' % (name, idx + 1)] = accept_col
                    df_analysis.to_csv('results/analysis.csv', index=False)
                    exit()

                # print runtime and ETA
                runtime = (round(time() * 1000) - start_time) / 1000
                avg_runtime = (avg_runtime * (iteration - 1) + runtime) / iteration
                eta = get_time_string((iterations - iteration) * avg_runtime)
                progress_row = '%d/%d dataset %d/%d (%s) fold %d/%d model %d/%d (%s) time: %s ETA: %s' % (
                    iteration, iterations, dataset_idx + 1, len(files), dataset_name, fold_idx + 1, len(folds),
                    model_idx + 1, len(models), model['name'], get_time_string(runtime), eta)
                print(progress_row)
                with open('results/progress_log.txt', 'a') as prog_file:
                    prog_file.write('%s\n' % progress_row)

"""
This script trains and tests the meta-learning model (XGBoost) on the meta-features and results collected by
Regression.py, including statistical tests and extraction of feature importance.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import rankdata


def create_meta_dataset():
    """
    Utilize the meta-feature and Regression.py result files to build the training data of the meta-model
    """
    df_model_performance = pd.read_csv('results/average_results.csv')
    df_meta_features = pd.read_csv('meta_learning/meta_features.csv')
    df_meta_features = df_meta_features.fillna(df_meta_features.mean())
    model_names = list(df_model_performance.columns[1:])
    rows = []
    for row_idx in range(len(df_model_performance)):
        row_model_performance = df_model_performance.loc[row_idx]
        row_meta_features = df_meta_features.loc[row_idx]
        if row_model_performance['dataset'] != row_meta_features['dataset']:  # sanity check
            raise ValueError('dataset order is different on average_results.csv and meta_features.csv!')
        best_model_performance = row_model_performance[1:].min()  # set to min() because we measure MSE
        found_best = False  # for sanity check
        for model_idx, model_name in enumerate(model_names):
            row_model_features = [0] * (len(model_names) + 1)
            row_model_features[model_idx] = 1  # set the one hot vector for model name
            if row_model_performance[model_name] == best_model_performance:  # if this model is best
                row_model_features[-1] = 1  # set target class ("is best model") to 1
                found_best = True
            rows.append(list(row_meta_features) + row_model_features)
        if not found_best:  # sanity check
            raise ValueError('there was no best model in a dataset!')
    meta_dataset = pd.DataFrame(rows, columns=list(df_meta_features.columns) + model_names + ['is best'])
    meta_dataset.to_csv('meta_learning/meta_dataset.csv', index=False)


def get_meta_learning_results():
    """
    Train and test the XGBoost meta-learner in a leave-one-out style cross-validation.
    """
    meta_dataset = pd.read_csv('meta_learning/meta_dataset.csv')
    datasets = pd.unique(meta_dataset['dataset'])
    average_results = pd.read_csv('results/average_results.csv', index_col='dataset')
    models = average_results.columns  # the candidate algorithms

    # start the leave-one-out cross validation
    correct_preds = 0  # to measure meta-model's performance
    meta_model_results = []
    for dataset_idx, dataset in enumerate(datasets):
        train_set = meta_dataset.loc[meta_dataset['dataset'] != dataset].drop(columns='dataset').to_numpy()
        test_set = meta_dataset.loc[meta_dataset['dataset'] == dataset].drop(columns='dataset').to_numpy()
        X_train, y_train = train_set[:, :-1], train_set[:, -1]
        X_test, y_test = test_set[:, :-1], test_set[:, -1]

        meta_model = xgb.XGBClassifier()
        meta_model.fit(X_train, y_train)
        y_pred = meta_model.predict_proba(X_test)
        selected_model_idx = np.argmax(y_pred[:, 1])  # for finding if the prediction was correct
        meta_model_results.append(average_results.loc[dataset][models[selected_model_idx]])
        is_correct = False
        if y_test[selected_model_idx] == 1:  # if the selected model was the correct one
            is_correct = True
            correct_preds += 1
        print('%d/%d select correct model: %s' % (dataset_idx + 1, len(datasets), is_correct))

    print('\naccuracy = %.4f' % (correct_preds / len(datasets)))
    average_results['MetaModel'] = meta_model_results
    average_results.to_csv('meta_learning/average_results.csv')


def do_statistical_test():
    """
    do the friedman and post-hoc tests that include the meta-learning results
    """
    df_results = pd.read_csv('meta_learning/average_results.csv')
    model_names = df_results.columns[1:]
    t_stat, p_val = friedmanchisquare(*[df_results[i] for i in model_names])
    print('\nfriedman test p-val = %s' % p_val)
    post_hoc_p_vals = posthoc_nemenyi_friedman(df_results.drop(columns='dataset').to_numpy())
    post_hoc_p_vals.columns = model_names
    print('\npost hoc p-vals:\n%s' % post_hoc_p_vals)
    post_hoc_p_vals.to_csv('meta_learning/post_hoc.csv', index=False)


def get_feature_importances():
    """
    extract importance and SHAP values of the meta-features
    """
    meta_dataset = pd.read_csv('meta_learning/meta_dataset.csv').drop(columns='dataset')
    features = list(meta_dataset.columns[:-1])
    array = meta_dataset.to_numpy()
    X, y = array[:, :-1], array[:, -1]
    meta_model = xgb.XGBClassifier()
    meta_model.fit(X, y)

    weights = meta_model.get_booster().get_score(importance_type='weight')
    gains = meta_model.get_booster().get_score(importance_type='gain')
    covers = meta_model.get_booster().get_score(importance_type='cover')
    importances = meta_model.feature_importances_

    y_pred = meta_model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    shap_values = y_pred.mean(axis=0)
    rows = []
    for i in range(len(features)):
        feat_id = 'f%d' % i
        if feat_id in weights:  # check because some features are sometimes completely ignored by meta-model
            weight, gain, cover = weights[feat_id], gains[feat_id], covers[feat_id]
        else:
            weight, gain, cover = 0, 0, 0
        rows.append([features[i], weight, gain, cover, importances[i], shap_values[i]])
    columns = ['feature', 'weight', 'gain', 'cover', 'importance', 'shap']
    pd.DataFrame(rows, columns=columns).to_csv('meta_learning/feature_importances.csv', index=False)


def save_ranks():
    """
    Save the ranks of the algorithms including the meta-agent to compare their performance
    """
    df_results = pd.read_csv('meta_learning/average_results.csv')
    df = df_results.drop(columns='dataset')
    ranks = rankdata(df.to_numpy(), method='dense', axis=1)
    df = pd.DataFrame(ranks, columns=df.columns)
    df['dataset'] = df_results['dataset']
    df.to_csv('meta_learning/ranks.csv')


# run this in the indicated order
create_meta_dataset()
get_meta_learning_results()
do_statistical_test()
get_feature_importances()
save_ranks()

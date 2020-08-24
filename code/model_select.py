from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
import numpy as np
import imblearn.over_sampling
from sklearn.model_selection import GridSearchCV


def cross_val_default(X, y, fold_type='kfold', nr_folds=5):
    """
    Cross validation for multiple classifiers with default settings
    """

    if fold_type == 'stratified':
        kf = StratifiedKFold(n_splits=nr_folds, shuffle=True)
    else:
        kf = KFold(n_splits=nr_folds, shuffle=True)

    lr_train, lr_vals = [], []
    sv_train, sv_vals = [], []
    rf_train, rf_vals = [], []
    gb_train, gb_vals = [], []
    kn_train, kn_vals = [], []
    ens_train, ens_vals = [], []

    for train_ind, val_ind in kf.split(X, y):
        x_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
        x_val, y_val = X.iloc[val_ind], y.iloc[val_ind]

        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        lr = LogisticRegression()
        lr.fit(x_train_scaled, y_train)
        y_score_val = lr.predict_proba(x_val_scaled)[:, 1]
        y_score_train = lr.predict_proba(x_train_scaled)[:, 1]
        lr_vals.append(roc_auc_score(y_val, y_score_val))
        lr_train.append(roc_auc_score(y_train, y_score_train))

        sv = SVC(probability=True)
        sv.fit(x_train_scaled, y_train)
        y_score_val = sv.predict_proba(x_val_scaled)[:, 1]
        y_score_train = sv.predict_proba(x_train_scaled)[:, 1]
        sv_vals.append(roc_auc_score(y_val, y_score_val))
        sv_train.append(roc_auc_score(y_train, y_score_train))

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        y_score_val = rf.predict_proba(x_val)[:, 1]
        y_score_train = rf.predict_proba(x_train)[:, 1]
        rf_vals.append(roc_auc_score(y_val, y_score_val))
        rf_train.append(roc_auc_score(y_train, y_score_train))

        gb = GradientBoostingClassifier()
        gb.fit(x_train, y_train)
        y_score_val = gb.predict_proba(x_val)[:, 1]
        y_score_train = gb.predict_proba(x_train)[:, 1]
        gb_vals.append(roc_auc_score(y_val, y_score_val))
        gb_train.append(roc_auc_score(y_train, y_score_train))

        kn = KNeighborsClassifier()
        kn.fit(x_train_scaled, y_train)
        y_score_val = kn.predict_proba(x_val_scaled)[:, 1]
        y_score_train = kn.predict_proba(x_train_scaled)[:, 1]
        kn_vals.append(roc_auc_score(y_val, y_score_val))
        kn_train.append(roc_auc_score(y_train, y_score_train))

        voting_classifer = VotingClassifier(
            estimators=[('kn', kn), ('gb', gb), ('rf', rf), ('lr', lr)],
            voting='soft',
            n_jobs=-1)
        ens = voting_classifer.fit(x_train, y_train)
        y_score_val = ens.predict_proba(x_val)[:, 1]
        y_score_train = ens.predict_proba(x_train)[:, 1]
        ens_vals.append(roc_auc_score(y_val, y_score_val))
        ens_train.append(roc_auc_score(y_train, y_score_train))

    print('ROC_AUC scores: \n')
    print(f'log reg:        val {np.mean(lr_vals):.3f} +- {np.std(lr_vals):.3f} | '
          f'train {np.mean(lr_train):.3f} +- {np.std(lr_train):.3f}')
    print(f'random forest:  val {np.mean(rf_vals):.3f} +- {np.std(rf_vals):.3f} | '
          f'train {np.mean(rf_train):.3f} +- {np.std(rf_train):.3f}')
    print(f'gradient boost: val {np.mean(gb_vals):.3f} +- {np.std(gb_vals):.3f} | '
          f'train {np.mean(gb_train):.3f} +- {np.std(gb_train):.3f}')
    print(f'knn:            val {np.mean(kn_vals):.3f} +- {np.std(kn_vals):.3f} | '
          f'train {np.mean(kn_train):.3f} +- {np.std(kn_train):.3f}')
    print(f'SVC:            val {np.mean(sv_vals):.3f} +- {np.std(sv_vals):.3f} | '
          f'train {np.mean(sv_train):.3f} +- {np.std(sv_train):.3f}')
    print(f'ensemble:       val {np.mean(ens_vals):.3f} +- {np.std(ens_vals):.3f} | '
          f'train {np.mean(ens_train):.3f} +- {np.std(ens_train):.3f}')

    return None


def cross_val_model(X, y, classifier, scale=True, nr_splits=5, fold_type='stratified', threshold=0.5, over_sample=1):
    """
    Cross-validation for a single model, allows for stratified of standard Kfold (default)
    """
    if fold_type == 'stratified':
        kf = StratifiedKFold(n_splits=nr_splits, shuffle=True)
    else:
        kf = KFold(n_splits=nr_splits, shuffle=True)

    vals_auc, vals_f1 = [], []
    train_auc, train_f1 = [], []

    for train_ind, val_ind in kf.split(X, y):
        x_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
        x_val, y_val = X.iloc[val_ind], y.iloc[val_ind]

        if scale:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

        if over_sample != 1:
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            ratio = {1: n_pos * over_sample, 0: n_neg}
            smote = imblearn.over_sampling.SMOTE(sampling_strategy=ratio)
            x_train, y_train = smote.fit_sample(x_train, y_train)

        model = classifier
        model.fit(x_train, y_train)
        y_score_val = model.predict_proba(x_val)[:, 1]
        y_score_train = model.predict_proba(x_train)[:, 1]
        y_pred_train = (y_score_train > threshold)
        y_pred_val = (y_score_val > threshold)
        vals_auc.append(roc_auc_score(y_val, y_score_val))
        train_auc.append(roc_auc_score(y_train, y_score_train))
        vals_f1.append(f1_score(y_val, y_pred_val))
        train_f1.append(f1_score(y_train, y_pred_train))

    print(classifier)
    print(f'auc: val {np.mean(vals_auc):.3f} +- {np.std(vals_auc):.3f} | '
          f'train {np.mean(train_auc):.3f} +- {np.std(train_auc):.3f}')
    print(f'f1 : val {np.mean(vals_f1):.3f} +- {np.std(vals_f1):.3f} | '
          f'train {np.mean(train_f1):.3f} +- {np.std(train_f1):.3f}')


def grid_search(X, y):
    """
    Grid Search with pre-defined parameters
    """
    knn_param = {'weights': ['uniform', 'distance'],
                 'n_neighbors': [7, 11, 13, 15, 19]}
    search_kn = GridSearchCV(KNeighborsClassifier(), knn_param, scoring='roc_auc', n_jobs=-1, verbose=1)
    search_kn.fit(X, y)
    print(search_kn.best_estimator_, search_kn.best_score_)

    rf_param = {'criterion': ['gini', 'entropy'],
                'n_estimators': [50, 100, 150, 200, 300],
                'max_features': ['auto', 'log2'],
                'max_depth': [4, 8, 12, 16, 24, 30, None]}
    search_rf = GridSearchCV(RandomForestClassifier(), rf_param, scoring='roc_auc', n_jobs=-1, verbose=1)
    search_rf.fit(X, y)
    print(search_rf.best_estimator_, search_rf.best_score_)

    lr_param = {'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1000]}
    search_lr = GridSearchCV(LogisticRegression(), lr_param, scoring='roc_auc', n_jobs=-1, verbose=1)
    search_lr.fit(X, y)
    print(search_lr.best_estimator_, search_lr.best_score_)

    return None

import numpy as np
import pandas as pd
import optuna
from scipy.stats import loguniform
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, root_mean_squared_error


train = pd.read_csv('Data/train.csv', index_col='id')
test = pd.read_csv('Data/test.csv', index_col='id')
orig = pd.read_csv('Data/orig.csv')
sample_submission = pd.read_csv('Data/sample_submission.csv')
sub_id = test.index

y = train['BeatsPerMinute']
X = train.drop('BeatsPerMinute', axis=1)

X['TrackDurationMs'] = X['TrackDurationMs']/60
test['TrackDurationMs'] = test['TrackDurationMs']/60


max_index = train.index.max()
orig.index = range(max_index+1, max_index+1+len(orig))
df = pd.concat([train, orig])

y_orig = df['BeatsPerMinute']
X_orig = df.drop('BeatsPerMinute', axis=1)


# %% Optuna


def objective(trial):
    params = {
        'verbose': False,
        'task_type': 'GPU',
        'devices': '0',
        'loss_function': 'RMSE',
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        # 'rsm': trial.suggest_float('rsm', 0.8, 1.0),
        # 'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'iterations': trial.suggest_int('iterations', 1000, 2500),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_seed': 9,  # Изменено с random_state на random_seed
    }

    model = CatBoostRegressor(**params)
    scorer = make_scorer(root_mean_squared_error)
    scores = cross_val_score(model, X, y, cv=cv_strat, scoring=scorer)

    return scores.mean()


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=150, timeout=40000, show_progress_bar=True)

print(f"Best score: {study.best_value}")

# Создаем финальные параметры с правильными именами
best_params = study.best_params.copy()
fixed_params = {
    'verbose': False,
    'task_type': 'GPU',
    'devices': '0',
    'loss_function': 'RMSE',
    'random_seed': 9,
}

# Объединяем параметры
final_params = {**fixed_params, **best_params}

# Явно преобразуем целочисленные параметры
final_params['iterations'] = int(final_params['iterations'])
final_params['depth'] = int(final_params['depth'])


f_model = CatBoostRegressor(**final_params)
f_model.fit(X, y)
y_pred = f_model.predict(test)

submission = pd.DataFrame({
    'id': sub_id,
    'BeatsPerMinute': y_pred
})
submission.to_csv('submission_cat_4.csv', index=False)
print("Submission file saved as 'submission_cat_4.csv'")


# Best score: 26.459135975314005
# LB Score: 26.38768
# %%
n_splits = 15
cv_strat = KFold(n_splits=n_splits, shuffle=True, random_state=9)
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)


class Feature_engineering(BaseEstimator, TransformerMixin):
    def __init__(self, method=None):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.method == 'log':
            return self.log_transform(X)
        elif self.method == 'binning':
            return self.binning_transform(X)
        elif self.method == 'interactions':
            return self.interaction_transform(X)
        elif self.method == 'statistical':
            return self.statictical_transform(X)
        else:
            return X

    def log_transform(self, X):
        X = X.copy()
        cols = X.columns
        shift_value = abs(np.min(X))
        for col in cols:
            if (X[col] < 0).any():
                X[col+'_log'] = np.log1p(X[col]+shift_value+1e-3)
            else:
                X[col+'_log'] = np.log1p(X[col])
        return X

    def binning_transform(self, X):
        X = X.copy()
        bins = 3
        cols = X.columns
        for col in cols:
            X[col+'_bin'] = pd.cut(X[col], bins=bins, labels=False)
        return X

    def interaction_transform(self, X):
        X = X.copy()
        cols = X.columns
        for i, col in enumerate(cols):
            for col2 in cols[i+1:]:
                X[f"{col}_mul_{col2}"] = X[col]*X[col2]
        return X

    def statictical_transform(self, X):
        X = X.copy()
        cols = X.columns
        X['sum'] = X[cols].sum(axis=1)
        X['mean'] = X[cols].mean(axis=1)
        X['std'] = X[cols].std(axis=1)
        for col in cols:
            X[col+'_std'] = X[col]-X[col].std()

        return X


def evaluate_features(X, y, params, cv):
    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv_strat, scoring=rmse_scorer)
    return -scores.mean()


base_score = 26.458866297489173
print(f"Base score: {base_score}")
print(f"N splits: {n_splits:3}")

best_params = {
    'min_data_in_leaf': 10,
    'iterations': 1792,
    'depth': 5,
    'learning_rate': 0.003797591296601685,
    'l2_leaf_reg': 9.52700758382667,
    'random_strength': 0.015805756536487414,
    'bagging_temperature': 0.9538697010145216
}

fixed_params = {
    'verbose': False,
    'task_type': 'GPU',
    'devices': '0',
    'loss_function': 'RMSE',
    'random_seed': 9,
}

final_params = {**fixed_params, **best_params}

transformations = [
    'original',
    'log',
    'binning',
    'interactions',
    'statistical'
]

result = []

for transform in transformations:
    engineer = Feature_engineering(method=transform if transform != 'original'
                                   else None)
    X_transformed = engineer.fit_transform(X)

    score = evaluate_features(X_transformed, y, final_params, cv_strat)

    improve = base_score - score

    result.append({
        'transform': transform,
        'rmse': score,
        'improve': improve

    })
    print(
        f"Transform: {transform:12} | RMSE: {score:.5f} | improve: {improve:+.5f}")

best_result = min(result, key=lambda x: x['rmse'])
print(
    f"\nBest trasformation: {best_result['transform']} with RMSE: {best_result['rmse']:.4f}")
# N splits:  5
# Transform: original     | RMSE: 26.45872 | improve: +0.00014
# Transform: log          | RMSE: 26.45868 | improve: +0.00019
# Transform: binning      | RMSE: 26.45873 | improve: +0.00013
# Transform: interactions | RMSE: 26.45981 | improve: -0.00094
# Transform: statistical  | RMSE: 26.45876 | improve: +0.00011

# Base score: 26.458866297489173
# N splits:  15
# Transform: original     | RMSE: 26.45874 | improve: +0.00013
# Transform: log          | RMSE: 26.45876 | improve: +0.00011
# Transform: binning      | RMSE: 26.45872 | improve: +0.00015
# Transform: interactions | RMSE: 26.45931 | improve: -0.00044
# Transform: statistical  | RMSE: 26.45868 | improve: +0.00019
# # %%
#
parameter = {'min_data_in_leaf': 10,
             'iterations': 1792,
             'depth': 5,
             'learning_rate': 0.003797591296601685,
             'l2_leaf_reg': 9.52700758382667,
             'random_strength': 0.015805756536487414,
             'bagging_temperature': 0.9538697010145216}
# %%

best_params = study.best_params.copy()
fixed_params = {
    'verbose': False,
    'task_type': 'GPU',
    'devices': '0',
    'loss_function': 'RMSE',
    'random_seed': 9,
}

# Объединяем параметры
final_params = {**fixed_params, **best_params}

# Явно преобразуем целочисленные параметры
final_params['iterations'] = int(final_params['iterations'])
final_params['depth'] = int(final_params['depth'])

f_model = CatBoostRegressor(**final_params)


def binning(X, bins):
    # Создаем словарь для новых признаков
    new_features = {}

    data_col = ['AudioLoudness', 'VocalContent', 'AcousticQuality',
                'InstrumentalScore', 'LivePerformanceLikelihood']  # list(X.columns)
    for col in data_col:
        new_features[f'{col}_bin'] = pd.cut(
            X[col], bins=bins, labels=False)

    # Создаем DataFrame из новых признаков
    new_features_df = pd.DataFrame(new_features, index=X.index)

    # Объединяем с исходным DataFrame за один раз
    result = pd.concat([X, new_features_df], axis=1)

    return result


def static(X):
    # Создаем словарь для новых признаков
    new_features = {
        'row_sum': X.sum(axis=1),
        'row_mean': X.mean(axis=1),
        'row_std': X.std(axis=1)
    }

    # Создаем DataFrame из новых признаков
    new_features_df = pd.DataFrame(new_features, index=X.index)

    # Объединяем с исходным DataFrame за один раз
    result = pd.concat([X, new_features_df], axis=1)

    return result


X = static(X)
test = static(test)

f_model.fit(X, y)
y_pred = f_model.predict(test)

submission = pd.DataFrame({
    'id': sub_id,
    'BeatsPerMinute': y_pred
})
submission.to_csv('submission_cat_5.csv', index=False)
print("Submission file saved as 'submission_cat_5.csv'")

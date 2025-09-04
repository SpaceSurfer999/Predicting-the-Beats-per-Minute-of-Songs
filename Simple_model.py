import numpy as np
import pandas as pd
import optuna
from scipy.stats import loguniform
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
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


# %%
cv_strat = KFold(n_splits=7, shuffle=True, random_state=9)
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)


def objective(trial):
    params = {
        'verbose': False,
        'task_type': 'GPU',
        'devices': '0',
        'loss_function': 'RMSE',
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
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
study.optimize(objective, n_trials=50, show_progress_bar=True)

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
submission.to_csv('submission_cat_3.csv', index=False)
print("Submission file saved as 'submission_cat_3.csv'")


# Best score: 26.459135975314005
# LB Score: 26.38768

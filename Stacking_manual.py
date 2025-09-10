# LOAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import (StandardScaler, PowerTransformer,
                                   QuantileTransformer)
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


train = pd.read_csv('Data/train.csv', index_col='id')
test = pd.read_csv('Data/test.csv', index_col='id')

X = train.drop('BeatsPerMinute', axis=1)
y = train['BeatsPerMinute']

columns = X.columns

# %% Create Features


def create_feature(df):
    df = df.copy()

    skewed_features = ['AudioLoudness', 'VocalContent',
                       'LivePerformanceLikelihood', 'InstrumentalScore']
    for feature in skewed_features:

        df[f'log_{feature}'] = np.log1p(df[feature] - df[feature].min())
        df[f'sqrt_{feature}'] = np.sqrt(df[feature] - df[feature].min())

    bimodal_feaatures = ['AcousticQuality', 'InstrumentalScore',
                         'LivePerformanceLikelihood']
    for feature in bimodal_feaatures:

        df[f'binary_{feature}'] = (
            df[feature] > df[feature].median()).astype(int)
        df[f'{feature}_cat'] = pd.qcut(df[feature], q=3, labels=False)

    # Interactions
    df['rhytm_energy_interaction'] = df['RhythmScore']*df['Energy']
    df['loudness_duration_interection'] = df['AudioLoudness'] * \
        np.log1p(df['TrackDurationMs'])
    df['vocal_acoustic_interaction'] = df['VocalContent']*df['AcousticQuality']

    # Transform feature
    transformer = PowerTransformer(method='yeo-johnson')
    feature_to_transform = ['AudioLoudness', 'VocalContent', 'TrackDurationMs']
    transformed = transformer.fit_transform(df[feature_to_transform])
    for i, feature in enumerate(feature_to_transform):
        df[f'{feature}_transformed'] = transformed[:, i]

    # Quantile
    quantile_transformer = QuantileTransformer(output_distribution='normal')
    quantile_features = ['VocalContent', 'Energy']
    quantile_transformed = quantile_transformer.fit_transform(
        df[quantile_features])
    for i, feature in enumerate(quantile_features):
        df[f'{feature}_quantile'] = quantile_transformed[:, i]

    # Polynomial features
    df['Energy_squared'] = df['Energy'] ** 2
    df['Rhythm_cube'] = df['RhythmScore'] ** 3
    df['AudioLoudness_sqrt'] = np.sqrt(np.abs(df['AudioLoudness']))

    # Clustering
    cluster_features = ['RhythmScore', 'Energy', 'AudioLoudness']
    if len(df) > 100:
        kmeans = KMeans(n_clusters=3, random_state=3)
        df['cluster'] = kmeans.fit_predict(df[cluster_features])

    # Statistic
    df['Zscore_energy'] = df['Energy'] - df['Energy'].mean()/df['Energy'].std()
    df['mean_rhythm_energy'] = df['RhythmScore'] + df['Energy']/2

    return df


# %% Create new X
X_engin = create_feature(X)
X_test_engin = create_feature(test)
# %% test with raw data
X_engin = X.copy()
X_test_engin = test.copy()

# %% Model params

cat_params = {
    'min_data_in_leaf': 51,
    'iterations': 2012,
    'depth': 4,
    'learning_rate': 0.005963995834838986,
    'l2_leaf_reg': 1.2078386951231046,
    'random_strength': 0.24192087720842353,
    'bagging_temperature': 1.0436582568501205
}

lgbm_params = {
    'n_estimators': 1717,
    'learning_rate': 0.0011725347593800623,
    'num_leaves': 30,
    'max_depth': 10,
    'subsample': 0.7120673254283085,
    'min_child_sample': 52,
    'colsample_bytree': 0.9712970828836915,
    'reg_alpha': 0.8923871491205672,
    'reg_lambda': 0.6785342488759971,

}

# %% Stacking
cat_scores = []
lgbm_scores = []

X_train, X_val, y_train, y_val = train_test_split(X_engin, y, test_size=0.2,
                                                  random_state=3)

n_folds = 12
kf = KFold(n_splits=n_folds, shuffle=True, random_state=3)

lgbm_oof = np.zeros(len(X_engin))
cat_oof = np.zeros(len(X_engin))

lgbm_test_preds = np.zeros(len(X_test_engin))
cat_test_preds = np.zeros(len(X_test_engin))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_engin, y)):
    print(f"\nFold: {fold+1} / {n_folds}")
    X_train, X_val = X_engin.iloc[train_idx], X_engin.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgbm = LGBMRegressor(**lgbm_params,
                         verbose=-1,
                         random_state=3,
                         early_stopping_round=100,
                         device='gpu'
                         )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    lgbm_oof[val_idx] = lgbm.predict(X_val)
    lgbm_test_preds += lgbm.predict(X_test_engin) / n_folds

    cat = CatBoostRegressor(**cat_params,
                            random_state=3,
                            verbose=False,
                            task_type='GPU'
                            )
    cat.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        # verbose=False
    )
    cat_oof[val_idx] = cat.predict(X_val)
    cat_test_preds += cat.predict(X_test_engin) / n_folds

    lgbm_score = np.sqrt(mean_squared_error(y_val, lgbm_oof[val_idx]))
    cat_score = np.sqrt(mean_squared_error(y_val, cat_oof[val_idx]))

    lgbm_scores.append(lgbm_score)
    cat_scores.append(cat_score)

    print(f"LGBM Score: {lgbm_score:.5f},\nCat Score:  {cat_score:.5f}")

meta_features = np.column_stack([lgbm_oof, cat_oof])
meta_features_test = np.column_stack([lgbm_test_preds, cat_test_preds])

meta_model = Ridge(alpha=1.0)

print("Meta model Fiting...")
meta_model.fit(meta_features, y)

final_predictions = meta_model.predict(meta_features_test)

final_oof_score = np.sqrt(mean_squared_error(
    y, meta_model.predict(meta_features)))

lgbm_mean_score = np.mean(lgbm_scores)
cat_mean_score = np.mean(cat_scores)
print("\nAverage models score:")
print(f"LGBM mean: {lgbm_mean_score:.5f}\nCat mean: {cat_mean_score:.5f}")
print(f"Final OOF RMSE: {final_oof_score:.5f}")
# %%
sub_name = 'submission_stacking_03'
submission = pd.DataFrame({
    'id': test.index,
    'BeatsPerMinute': final_predictions
})
submission.to_csv(f'{sub_name}.csv', index=False)
print(f"Submission file saved as {sub_name}.csv")

# %%
#               TEST

# %%


Final OOF RMSE: 26.45856
Public:         26.38767

Final OOF RMSE: 26.45765
Public:         26.38726


Raw data:
Average models score 5 fold:
LGBM mean: 26.45854
Cat mean: 26.45922
Final OOF RMSE: 26.45807

Average models score 12 fold:
LGBM mean: 26.45800
Cat mean: 26.45839
Final OOF RMSE: 26.45717
Public:         26.38624

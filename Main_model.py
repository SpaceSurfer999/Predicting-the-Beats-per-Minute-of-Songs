import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import (StandardScaler, PowerTransformer,
                                   QuantileTransformer, RobustScaler)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor


def load_data():

    train = pd.read_csv('Data/train.csv', index_col='id')
    test = pd.read_csv('Data/test.csv', index_col='id')

    train['TrackDurationMs'] = train['TrackDurationMs']/500000
    test['TrackDurationMs'] = test['TrackDurationMs']/500000

    train['AudioLoudness'] = train['AudioLoudness']/-30
    test['AudioLoudness'] = test['AudioLoudness']/-30

    train['VocalContent'] = train['VocalContent']*3.5
    test['VocalContent'] = test['VocalContent']*3.5

    y = train['BeatsPerMinute']
    X = train.drop('BeatsPerMinute', axis=1)

    print(f"\nTrain size: {train.shape}\nTest size: {test.shape}")

    return X, y, test


def create_features_1(df):
    df = df.copy()

    bimodal_features = ['AcousticQuality', 'InstrumentalScore',
                        'LivePerformanceLikelihood']
    for feature in bimodal_features:

        df[f'binary_{feature}'] = (
            df[feature] > df[feature].median()).astype(int)
        df[f'{feature}_cat'] = pd.qcut(df[feature], q=[0, .2, .4, .6, .8, 1],
                                       labels=False,  duplicates='drop')

    quantile_transformer = QuantileTransformer(
        output_distribution='normal')
    quantile_features = ['RhythmScore', 'Energy', 'AudioLoudness']
    quantile_transformed = quantile_transformer.fit_transform(
        df[quantile_features])
    for i, feature in enumerate(quantile_features):
        df[f'{feature}_quantile'] = quantile_transformed[:, i]

    df['Energy_squared'] = df['Energy'] ** 2
    df['Rhythm_squared'] = df['RhythmScore'] ** 2
    df['Energy_cube'] = df['Energy'] ** 3
    df['Rhythm_cube'] = df['RhythmScore'] ** 3
    df['AudioLoudness_sqrt'] = np.sqrt(np.abs(df['AudioLoudness']))
    df['RhythmScore_sqrt'] = np.sqrt(np.abs(df['RhythmScore']))
    df['Energy_sqrt'] = np.sqrt(np.abs(df['Energy']))

    cluster_features = ['RhythmScore', 'Energy', 'AudioLoudness']
    if len(df) > 100:
        kmeans = KMeans(n_clusters=5, random_state=3)
        df['cluster'] = kmeans.fit_predict(df[cluster_features])

    return df


def create_features_2(df):

    df = df.copy()
    columns = df.columns.to_list()

    # log
    for col in columns:

        df[f'log_{col}'] = np.log1p(df[col]-df[col].min())

    # quantile
    quantile_transformer = QuantileTransformer(
        output_distribution='normal')
    quantile_features = columns  # ['RhythmScore', 'Energy', 'AudioLoudness']
    quantile_transformed = quantile_transformer.fit_transform(
        df[quantile_features])
    for i, feature in enumerate(quantile_features):
        df[f'{feature}_quantile'] = quantile_transformed[:, i]

    # polynom
    for col in columns:
        df[f'squared_{col}'] = df[col]**2
        df[f'cube_{col}'] = df[col]**3
        df[f'sqrt_{col}'] = np.sqrt(np.abs(df[col]))

    # statistic
    for col in columns:
        df[f'zscore_{col}'] = (df[col] - df[col].mean())/df[col].std()

    df['mean_rhythm_energy'] = (df['RhythmScore'] + df['Energy'])/2

    return df


def create_features_3(df):
    """Create additional features that might help predict BPM"""
    df = df.copy()

    # Rhythm and Energy interactions
    df['RhythmEnergyProduct'] = df['RhythmScore'] * df['Energy']
    df['RhythmEnergyRatio'] = df['RhythmScore'] / (df['Energy'] + 1e-8)

    # Audio characteristics
    df['LoudnessEnergyProduct'] = df['AudioLoudness'] * df['Energy']
    df['VocalInstrumentalRatio'] = df['VocalContent'] / \
        (df['InstrumentalScore'] + 1e-8)

    # Track duration features
    df['DurationMoodProduct'] = df['TrackDurationMs'] * df['MoodScore']

    # Performance and quality features
    df['QualityPerformanceProduct'] = df['AcousticQuality'] * \
        df['LivePerformanceLikelihood']

    # Polynomial features for top correlated features
    top_3_features = ['RhythmScore', 'MoodScore', 'TrackDurationMs']
    for feature in top_3_features:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))

    # Binned features
    df['EnergyBin'] = pd.qcut(df['Energy'],  q=[0, .2, .4, .6, .8, 1],
                              labels=False)
    df['RhythmBin'] = pd.qcut(df['RhythmScore'],  q=[0, .2, .4, .6, .8, 1],
                              labels=False)

    # Interaction between rhythm and tempo-related features
    df['RhythmDurationInteraction'] = df['RhythmScore'] * df['TrackDurationMs']

    return df


def choose_feature_engin(X, test, create=1):
    X = X.copy()
    test = test.copy()
    if create == 1:
        X = create_features_1(X)
        test = create_features_1(test)
    elif create == 2:
        X = create_features_2(X)
        test = create_features_2(test)
    else:
        X = create_features_3(X)
        test = create_features_3(test)
    print(f"\nApply create Features #{create} ")
    return X, test


def scaled_data(X, test):
    columns_name = X.columns
    scaler = RobustScaler()

    X = scaler.fit_transform(X)
    test = scaler.transform(test)

    X = pd.DataFrame(X, columns=columns_name)
    test = pd.DataFrame(test, columns=columns_name)
    print("\nApply Scaler")
    print(f"\nX_scaled: {X.shape}\nTest scaled: {test.shape}")

    return X, test


def evaluate_model(model, X, y,
                   model_name, n_folds=12):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    fold_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1} / {n_folds}")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = root_mean_squared_error(y_val, y_pred)
        fold_scores.append(score)

    avg_score = np.mean(fold_scores)
    result = {
        'model_name': model_name,
        'RMSE_score': avg_score,
        # 'model': model
    }

    return result


def create_model(X, y, n_folds):
    results = []
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            random_state=1,
            n_jobs=5,
            max_depth=10
        ),
        'XGBoost': XGBRegressor(
            n_estimators=1000,
            random_state=1,
            n_jobs=5,
            device='gpu'
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=1000,
            random_state=1,
            n_jobs=5,
            verbose=-1,
            device='gpu'
        ),
        'CatBoost': CatBoostRegressor(
            iterations=1000,
            random_state=1,
            verbose=False,
            task_type='GPU'
        ),
        'Ridge': Ridge(alpha=1.0),
        'Elastic Net': ElasticNet(alpha=1.0, random_state=1)
    }
    for name, model in models.items():
        print(f"\nTraining model: {name}")
        result = evaluate_model(model, X,  y, name, n_folds=n_folds)
        print(f"RMSE: {result['RMSE_score']} {name}")
        results.append(result)

    return results


def main(create, n_folds):
    X, y, test = load_data()

    X, test = choose_feature_engin(X, test, create)

    scaled_X, scaled_test = scaled_data(X, test)

    results = create_model(scaled_X, y, n_folds)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE_score')

    # Печать для отладки
    print(f"Results DataFrame shape: {results_df.shape}")

    return scaled_X, scaled_test, results_df


if __name__ == "__main__":
    start = datetime.now()
    print(f"Start time:\n{start}")

    scaled_X, scaled_test, results_df = main(create=3, n_folds=5)
    # Добавьте вывод results_df для проверки
    print(f"\nResults DataFrame:\n{results_df}")

    print(f"\nTime spent\n{datetime.now()-start}")

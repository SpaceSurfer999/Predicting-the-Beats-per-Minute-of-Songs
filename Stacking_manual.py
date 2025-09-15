# LOAD
import time
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (KFold, train_test_split,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import (StandardScaler, PowerTransformer,
                                   QuantileTransformer, RobustScaler)
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import randint, uniform
# %%
start = datetime.now()
print(f"Start time:\n{start}")
train = pd.read_csv('Data/train.csv', index_col='id')
test = pd.read_csv('Data/test.csv', index_col='id')


# train = train[train['LivePerformanceLikelihood'] > 0.03]
# train = train[train['MoodScore'] > 0.1]

y = train['BeatsPerMinute']
X = train.drop('BeatsPerMinute', axis=1)

X['TrackDurationMs'] = X['TrackDurationMs']/500000
test['TrackDurationMs'] = test['TrackDurationMs']/500000

X['AudioLoudness'] = X['AudioLoudness']/-30
test['AudioLoudness'] = test['AudioLoudness']/-30

X['VocalContent'] = X['VocalContent']*3.5
test['VocalContent'] = test['VocalContent']*3.5
# %%

r_scaler = RobustScaler()
x_scale = pd.DataFrame(r_scaler.fit_transform(X))
x_scale.hist(bins=100, figsize=(25, 15))
# X_test_engin_scale = r_scaler.transform(X_test_engin)v
# %%


def val_counts(X):
    columns = X.columns
    for col in columns:
        va_count = X[col].value_counts().head(1).index
        median_col = X[col].median()
        X[f'is_{col}'] = (X[col].isin(va_count)).astype(int)
        X[f"replace_{col}"] = X[col].replace(va_count, median_col)

# %%

# %%

# %%


# %%
X.hist(figsize=(25, 15), bins=100)
# %%

# %%
                                #  EDA

# %%

ax = train.hist(bins=70, figsize=(25, 15))
plt.suptitle('Train')
# %%
val_count_audioloud = train['AudioLoudness'].value_counts().head(1).index
df_audio = train[train['AudioLoudness'].isin(val_count_audioloud)]
# ax = df_audio.hist(column = ['AudioLoudness', 'BeatsPerMinute'], bins=100)
df_audio.hist(figsize=(25, 15), bins=100)
# %%
# train['AudioLoudness'].hist(bins=100)

# %%

# %%
# %%
df_audio.hist(bins=70, figsize=(25, 15)

# %%

df_no_audio=train[train['AudioLoudness'] != -1.357]
df_temp['TrackDurationMs']=df_temp['TrackDurationMs']/500000
df_no_audio.hist(bins=100, figsize=(25, 15))

corr_matrix=df_no_audio.corr()

plt.figure(figsize=(20, 15))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
    )
# %%
df_temp=df_no_audio[df_no_audio['AudioLoudness'] > -8]
df_temp['TrackDurationMs']=df_temp['TrackDurationMs']/500000
df_temp.hist(figsize=(25, 15), bins=100)
# %%
corr_matrix=df_no_audio.corr()
plt.figure(figsize=(20, 15))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
    )
# %%

# %%


# %%
va_count=X['AudioLoudness'].value_counts()
# va_count.hist(x='AudioLoudness',bins=1)
rare_value=va_count[va_count == 57595].index
# %%
X['AudioLoudness']=X['AudioLoudness'].replace(rare_value, 1)
X['AudioLoudness'].hist(bins=100)
# %%
df=X.copy()
# %%
va_count=X['LivePerformanceLikelihood'].value_counts()
X['LivePerformanceLikelihood'].hist(bins=200)
rare_value=va_count[va_count == 84241].index
X['LivePerformanceLikelihood']=X['LivePerformanceLikelihood'].replace(
    rare_value, X['LivePerformanceLikelihood'].median())
X['LivePerformanceLikelihood'].hist(bins=100)

# %%

# %%

# %%


# %%
print(X.shape)
print(y.shape)
# %%
print("До преобразования:")
print(X.dtypes)
print(f"Память: {X.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
# %%
num_cols=X.select_dtypes(include=['float64', 'int64']).columns
X_16=X.copy()
X_16[num_cols]=X_16[num_cols].astype('float32')
# %%

print("После преобразования:")
print(X_16.dtypes)
print(f"Память: {X_16.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
# %%
# df_all = X[X['LivePerformanceLikelihood'] > 0.03]
# %%

# print(len(X) - len(df_all))
# %% Create Features


def create_feature(df):
    df=df.copy()

    bimodal_feaatures=['AcousticQuality', 'InstrumentalScore',
                         'LivePerformanceLikelihood']
    for feature in bimodal_feaatures:

        df[f'binary_{feature}']=(
            df[feature] > df[feature].median()).astype(int)
        df[f'{feature}_cat']=pd.qcut(df[feature], q=5,
                                       labels=False,  duplicates='drop')

    quantile_transformer=QuantileTransformer(
        output_distribution='normal')
    quantile_features=['RhythmScore', 'Energy', 'AudioLoudness']
    quantile_transformed=quantile_transformer.fit_transform(
        df[quantile_features])
    for i, feature in enumerate(quantile_features):
        df[f'{feature}_quantile']=quantile_transformed[:, i]

    df['Energy_squared']=df['Energy'] ** 2
    df['Rhythm_squared']=df['RhythmScore'] ** 2
    df['Energy_cube']=df['Energy'] ** 3
    df['Rhythm_cube']=df['RhythmScore'] ** 3
    df['AudioLoudness_sqrt']=np.sqrt(np.abs(df['AudioLoudness']))
    df['RhythmScore_sqrt']=np.sqrt(np.abs(df['RhythmScore']))
    df['Energy_sqrt']=np.sqrt(np.abs(df['Energy']))

    cluster_features=['RhythmScore', 'Energy', 'AudioLoudness']
    if len(df) > 100:
        kmeans=KMeans(n_clusters=5, random_state=3)
        df['cluster']=kmeans.fit_predict(df[cluster_features])

    return df

# %% test with raw data
X_engin=X.copy()
X_test_engin=test.copy()

# %% scaler raw data
r_scaler=RobustScaler()
X_engin_scale=r_scaler.fit_transform(X_engin)
X_test_engin_scale=r_scaler.transform(X_test_engin)
print(
    f'X_scaled raw :{ X_engin_scale.shape}\nX_test_scale raw: {X_test_engin_scale.shape}')

# %% test with raw data
X_engin=X_16.copy()
X_test_engin=test.copy()

# %% scaler raw data
r_scaler=RobustScaler()
X_engin_scale=r_scaler.fit_transform(X_engin[columns])
X_test_engin_scale=r_scaler.transform(X_test_engin)
print(
    f'X_scaled raw :{ X_engin_scale.shape}\nX_test_scale raw: {X_test_engin_scale.shape}')

# # %% Create new X
# X_engin = create_feature(X)
# X_test_engin = create_feature(test)
# # %% Scaler

# r_scaler = RobustScaler()
# X_engin_scale = r_scaler.fit_transform(X_engin)
# X_test_engin_scale = r_scaler.transform(X_test_engin)
# print(
#     f'X_scaled :{ X_engin_scale.shape}\nX_test_scale: {X_test_engin_scale.shape}')
# %% Model params

cat_params={
    'min_data_in_leaf': 94,
    'iterations': 1655,
    'depth': 5,
    'learning_rate': 0.004218008633973081,
    'l2_leaf_reg': 9.707409868449949,
    'random_strength': 0.04287333829445321,
    'bagging_temperature': 1.480556448408197
}

lgbm_params={
    'n_estimators': 1376,
    'learning_rate': 0.0015647126709542488,
    'num_leaves': 35,
    'max_depth': 10,
    'subsample': 0.8932518516979349,
    'min_child_sample': 92,
    'colsample_bytree': 0.9944970695439489,
    'reg_alpha': 0.8214512149531262,
    'reg_lambda': 0.21103767689899985

}

# %% Stacking

cat_scores=[]
lgbm_scores=[]

X_train, X_val, y_train, y_val=train_test_split(X_engin, y, test_size=0.2,
                                                  random_state=3)

n_folds=12
kf=KFold(n_splits=n_folds, shuffle=True, random_state=3)

lgbm_oof=np.zeros(len(X_engin))
cat_oof=np.zeros(len(X_engin))

lgbm_test_preds=np.zeros(len(X_test_engin))
cat_test_preds=np.zeros(len(X_test_engin))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_engin, y)):
    # if fold % 2 == 0:
    print(f"Fold: {fold+1} / {n_folds}")
    X_train, X_val=X_engin.iloc[train_idx], X_engin.iloc[val_idx]
    y_train, y_val=y.iloc[train_idx], y.iloc[val_idx]

    lgbm=LGBMRegressor(**lgbm_params,
                         verbose=-1,
                         random_state=3,
                         early_stopping_round=100,
                         device='gpu',
                         n_jobs=6
                         )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    lgbm_oof[val_idx]=lgbm.predict(X_val)
    lgbm_test_preds += lgbm.predict(X_test_engin) / n_folds

    cat=CatBoostRegressor(**cat_params,
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
    cat_oof[val_idx]=cat.predict(X_val)
    cat_test_preds += cat.predict(X_test_engin) / n_folds

    lgbm_score=np.sqrt(mean_squared_error(y_val, lgbm_oof[val_idx]))
    cat_score=np.sqrt(mean_squared_error(y_val, cat_oof[val_idx]))

    lgbm_scores.append(lgbm_score)
    cat_scores.append(cat_score)

    # if fold % 2 == 0:
    #     print(f"LGBM Score: {lgbm_score:.5f},\nCat Score:  {cat_score:.5f}")
# %% Meta model train
opti_time=datetime.now()
print(opti_time)

meta_Ridge=Ridge(alpha=1.0, random_state=3)
meta_Elastic=ElasticNet(alpha=0.001,
                          max_iter=100,
                          tol=1e-06,
                          random_state=3)

meta_forest=RandomForestRegressor(n_estimators=100, max_depth=5,
                                    # max_features='auto',
                                    min_samples_split=5,
                                    min_samples_leaf=5,
                                    random_state=3,
                                    n_jobs=-1)

meta_features=np.column_stack([lgbm_oof, cat_oof])
meta_features_test=np.column_stack([lgbm_test_preds, cat_test_preds])

print("Optimized meta model..")

# param_dist={
#     'n_estimators': randint(50, 150),
#     'max_depth': [5, 8, 10],
#     'min_samples_split': randint(2, 8),
#     'min_samples_leaf': randint(1, 5),
#     # 'bootstrap': [True, False],
#     'max_features': uniform(0.1, 0.9)
# }

# # Создание и настройка RandomizedSearchCV
# random_search=RandomizedSearchCV(
#     estimator=RandomForestRegressor(random_state=42),
#     param_distributions=param_dist,
#     n_iter=5,
#     cv=12,
#     scoring='neg_mean_squared_error',
#     n_jobs=-1,
#     verbose=1,
#     random_state=3
# )

# # Обучаем модель один раз
# random_search.fit(meta_features, y)

# # Получаем предсказания для тестовых данных
# final_predictions_f=random_search.predict(meta_features_test)

# # Вычисляем OOF score (осторожно: это оценка на тренировочных данных!)
# final_oof_score_f=np.sqrt(mean_squared_error(
#     y, random_search.predict(meta_features)))

# # Правильный вывод лучших параметров
# print(f"Best parameters for meta model: {random_search.best_params_}")
# best_model=random_search.best_estimator_


# Ridge
print('Ridge fit')
meta_Ridge.fit(meta_features, y)

final_predictions_r=meta_Ridge.predict(meta_features_test)

final_oof_score_r=np.sqrt(mean_squared_error(
    y, meta_Ridge.predict(meta_features)))


# Elastic
print('Elastic fit')
meta_Elastic.fit(meta_features, y)

final_predictions_l=meta_Elastic.predict(meta_features_test)

final_oof_score_l=np.sqrt(mean_squared_error(
    y, meta_Elastic.predict(meta_features)))

# Forest
print('Forestd fit')
meta_forest.fit(meta_features, y)

final_predictions_f=meta_forest.predict(meta_features_test)

final_oof_score_f=np.sqrt(mean_squared_error(
    y, meta_forest.predict(meta_features)))


lgbm_mean_score=np.mean(lgbm_scores)
cat_mean_score=np.mean(cat_scores)

# importances=meta_forest.feature_importances_
# print(importances)


print("\nAverage models score:")
print(f"LGBM mean: {lgbm_mean_score:.5f}\nCat mean: {cat_mean_score:.5f}")
print(f"Final OOF RMSE Ridge:   {final_oof_score_r:.5f}")
print(f"Final OOF RMSE Elastic: {final_oof_score_l:.5f}")
print(f"Final OOF RMSE Forest: {final_oof_score_f:.5f}")
end=datetime.now()
print(f"Model optimized  time: {end-opti_time}")
# print(f"All optimized time:    {end-start}")

# %%

# После завершения поиска
results=random_search.cv_results_
scores=results['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(scores)+1), scores, 'b-', label='Test score')
plt.plot(range(1, len(scores)+1), np.maximum.accumulate(scores),
         'r-', label='Best so far')
plt.xlabel('Iteration')
plt.ylabel('Score (negative MSE)')
plt.title('RandomizedSearchCV Progress')
plt.legend()
plt.grid(True)
plt.show()

# %%


# %%
# if final_oof_score_r < final_oof_score_l:
#     final_predictions = final_predictions_r
# elif final_oof_score_l < final_oof_score_f:
#     final_predictions = final_predictions_l
# else:
# final_predictions = final_predictions_f

sub_name='submission_stacking_14'
submission=pd.DataFrame({
    'id': test.index,
    'BeatsPerMinute': final_predictions_f
})
submission.to_csv(f'{sub_name}.csv', index=False)
print(f"Submission file saved as {sub_name}.csv")

# %%
#               TEST

# %%


# Final OOF RMSE: 26.45856
# Public:         26.38767

# Final OOF RMSE: 26.45765
# Public:         26.38726


# Raw data:
# Average models score 5 fold:
# LGBM mean: 26.45854
# Cat mean: 26.45922
# Final OOF RMSE: 26.45807

# Average models score 12 fold:
# LGBM mean: 26.45800
# Cat mean: 26.45839
# Final OOF RMSE: 26.45717
# Public:         26.38624


Best score cat: 26.459068091002415
best_params cat: {'min_data_in_leaf': 42,
                  'iterations': 2392,
                  'depth': 5,
                  'learning_rate': 0.0027274212349796332,
                  'l2_leaf_reg': 6.123720639826032,
                  'random_strength': 0.6577883246794647,
                  'bagging_temperature': 0.922906797377011}

Best score lgbm: 26.45961788476395
Best params lgbm: {'n_estimators': 1737,
                   'learning_rate': 0.0010570252271237916,
                   'num_leaves': 52,
                   'max_depth': 8,
                   'subsample': 0.8322191124994692,
                   'min_child_sample': 31,
                   'colsample_bytree': 0.7134499314247663,
                   'reg_alpha': 0.9824302646384262,
                   'reg_lambda': 0.149745000467102}


15 fold
Best score cat: 26.45884492078436
best_params cat: {'min_data_in_leaf': 65,
                  'iterations': 2373,
                  'depth': 5,
                  'learning_rate': 0.0033139319869651936,
                  'l2_leaf_reg': 9.145329900729005,
                  'random_strength': 0.2657833313587459,
                  'bagging_temperature': 0.9157599186561314}

Trial 85 finished with value: 26.458606758336987
parameters: {'n_estimators': 2316,
             'learning_rate': 0.0008743861905514981,
             'num_leaves': 32,
             'max_depth': 10,
             'subsample': 0.8365509294561634,
             'min_child_sample': 36,
             'colsample_bytree': 0.9781216760868153,
             'reg_alpha': 0.9143568371828773,
             'reg_lambda': 0.8436412454821802}.


********************************************************
Original DATA

Best score cat: 26.458800579014152
best_params cat: {'min_data_in_leaf': 94,
                  'iterations': 1655,
                  'depth': 5,
                  'learning_rate': 0.004218008633973081,
                  'l2_leaf_reg': 9.707409868449949,
                  'random_strength': 0.04287333829445321,
                  'bagging_temperature': 1.480556448408197}

Best score lgbm: 26.45852103933812
Best params lgbm: {'n_estimators': 1376,
                   'learning_rate': 0.0015647126709542488,
                   'num_leaves': 35,
                   'max_depth': 10,
                   'subsample': 0.8932518516979349,
                   'min_child_sample': 92,
                   'colsample_bytree': 0.9944970695439489,
                   'reg_alpha': 0.8214512149531262,
                   'reg_lambda': 0.21103767689899985}

Best:
Average models score: 12 fold  RobustScaler
LGBM mean: 26.45787
Cat mean: 26.45857
Final OOF RMSE Ridge:   26.45726
Final OOF RMSE Elastic: 26.45726
Final OOF RMSE Forest: 26.43701
Public: 26.38608

Average models score: 12 fold No RobustScaler
LGBM mean: 26.45789
Cat mean: 26.45857
Final OOF RMSE Ridge:   26.45729
Final OOF RMSE Elastic: 26.45729
Final OOF RMSE Forest: 26.43779





** ******************************************************
Average models score:
LGBM mean: 26.45787
Cat mean: 26.45857
Final OOF RMSE Ridge:   26.45728
Final OOF RMSE Elastic: 26.45728
Final OOF RMSE Forest: 26.43739

** ******************************************************
Best score cat: 26.417905018293112
best_params cat: {'min_data_in_leaf': 31,
                  'iterations': 2236,
                  'depth': 7,
                  'learning_rate': 0.0012088914565158168,
                  'l2_leaf_reg': 6.180021444877318,
                  'random_strength': 0.35975411628501924,
                  'bagging_temperature': 0.9799254085864368}

Best score lgbm: 26.415879086329824
Best params lgbm: {'n_estimators': 1871,
                   'learning_rate': 0.0014548681197464768,
                   'num_leaves': 33,
                   'max_depth': 8,
                   'subsample': 0.8317435661163157,
                   'min_child_sample': 61,
                   'colsample_bytree': 0.8434530130269543,
                   'reg_alpha': 0.34999404589204935,
                   'reg_lambda': 0.6095706112392547}

RAW DATA cut data

Average models score:
LGBM mean: 26.41633
Cat mean: 26.41753
Final OOF RMSE Ridge:   26.41602
Final OOF RMSE Elastic: 26.41602
Final OOF RMSE Forest: 26.39331
Public 26.38694
All optimized time: 0: 09: 20.141346

Scaled data:
Average models score:
LGBM mean: 26.41613
Cat mean: 26.41754
Final OOF RMSE Ridge:   26.41578
Final OOF RMSE Elastic: 26.41578
Final OOF RMSE Forest: 26.39370
Public 26.38661
All optimized time: 0: 09: 25.086521

** ******************************************************
Best score cat: 26.458800579014152
best_params cat: {'min_data_in_leaf': 94,
                  'iterations': 1655,
                  'depth': 5,
                  'learning_rate': 0.004218008633973081,
                  'l2_leaf_reg': 9.707409868449949,
                  'random_strength': 0.04287333829445321,
                  'bagging_temperature': 1.480556448408197}

Best score lgbm: 26.45852103933812
Best params lgbm: {'n_estimators': 1376,
                   'learning_rate': 0.0015647126709542488,
                   'num_leaves': 35,
                   'max_depth': 10,
                   'subsample': 0.8932518516979349,
                   'min_child_sample': 92,
                   'colsample_bytree': 0.9944970695439489,
                   'reg_alpha': 0.8214512149531262,
                   'reg_lambda': 0.21103767689899985}

RAW DATA Scaled RobustScaler

Average models score:
LGBM mean: 26.41601
Cat mean: 26.41756
Final OOF RMSE Ridge:   26.41550
Final OOF RMSE Elastic: 26.41550
Final OOF RMSE Forest: 26.40618
Public 26.38587

All optimized time: 0: 06: 10.616893

Scaled data cut data:

Average models score:
LGBM mean: 26.41617
Cat mean: 26.41761
Final OOF RMSE Ridge:   26.41567
Final OOF RMSE Elastic: 26.41567
Final OOF RMSE Forest: 26.39495
Public 26.38661

All optimized time: 0: 06: 00.232415

** ******************************************************

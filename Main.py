from Data_load import Dataloader
from Preprocessing import Preprocessor
from EDA import EDA
from Model import MLModel
from Tuning import XGBOptimizer
from Multioptimizer import Multioptimizer
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import warnings

# Игнорируем предупреждения, связанные с параметрами XGBoost
warnings.filterwarnings("ignore", message=".*are not used.*")


def main(train_model=False, plot=False, sub=False):
    # Load class
    loader = Dataloader()
    eda_data = EDA()
    preprocessor_data = Preprocessor()
    model = MLModel()

    # load data
    train, test, orig, sub_id = loader.load_data()
    train_col = list(train.columns)
    test_col = list(test.columns)
    print(train_col)

    max_index = train.index.max()
    orig.index = range(max_index+1, max_index+1+len(orig))
    df = pd.concat([train, orig])

    # # eda analyze
    describe_data = eda_data.summary(train)
    df_corr = eda_data.correlation(train)

    # plot
    if plot:
        violin_plot = eda_data.plot_violinplot(train)
        box_plot = eda_data.plot_boxplot(train)
        violin_plot = eda_data.plot_violinplot(df)

    # Prepare data
    y = train['BeatsPerMinute']
    X = train.drop('BeatsPerMinute', axis=1)

    X['TrackDurationMs'] = X['TrackDurationMs']/60
    test['TrackDurationMs'] = test['TrackDurationMs']/60

    # X = preprocessor_data.genetic_feature_engineering(X, y)
    # test = preprocessor_data.genetic_feature_engineering(test)

    # # feature engineering

    X = preprocessor_data.logar(X)
    test = preprocessor_data.logar(test)

    # # plot_distributions = eda_data.plot_distribution(X[:, 10])

    X = preprocessor_data.binning(X, bins=20)
    test = preprocessor_data.binning(test, bins=20)

    X = preprocessor_data.sqrt_method(X)
    test = preprocessor_data.sqrt_method(test)

    X = preprocessor_data.static(X)
    test = preprocessor_data.static(test)

    X = preprocessor_data.pair_feature(X)
    test = preprocessor_data.pair_feature(test)
    # if plot:
    #     plot_distributions = eda_data.plot_distribution(X)

    if train_model:
        X_scaled = preprocessor_data.fit_transform(X)
        test_scaled = preprocessor_data.transform(test)
        print(f"X_scaled: {X_scaled.shape}")

        # Преобразуем обратно в DataFrame с сохранением имен признаков
        if hasattr(X, 'columns'):
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            test_scaled = pd.DataFrame(test_scaled, columns=test.columns)

        # Оптимизация гиперпараметров для нескольких моделей
        print("Starting hyperparameter optimization for multiple models...")
        optimizer = Multioptimizer(X_scaled, y, cv=8, n_trials=10)
        best_params, best_scores = optimizer.optimize_all(
            models=['catboost',
                    'BayesianRidge'])  # 'xgb', 'lgbm', 'catboost', 'LinearRegression',

        # Находим лучшую модель
        best_model_name = min(best_scores, key=best_scores.get)
        print(
            f"\nBest model is {best_model_name} with score: {best_scores[best_model_name]:.4f}")
        print(
            f"Best parameters for {best_model_name}: {best_params[best_model_name]}")

        # Тренировка финальной модели с лучшими параметрами
        print(
            f"Training final model ({best_model_name}) with best parameters...")
        # final_model = optimizer.get_best_model(best_model_name)
        # final_model.fit(X, y)

        # Визуализация процесса оптимизации для лучшей модели
        # try:
        #     import optuna.visualization as vis
        #     fig = vis.plot_optimization_history(
        #         optimizer.studies[best_model_name])
        #     fig.show()
        # except ImportError:
        #     print("Install plotly to visualize optimization history")

    #     y_pred = final_model.predict(test_scaled)
    # else:
    #     y_pred = None  # Добавляем переменную y_pred для случая, когда train_model=False

    # if sub and y_pred is not None:
    #     # Создание submission файла
    #     submission = pd.DataFrame({
    #         'id': sub_id,
    #         'BeatsPerMinute': y_pred
    #     })
    #     submission.to_csv('submission_xgb_optimized.csv', index=False)
    #     print("Submission file saved as 'submission_xgb_optimized.csv'")

    # Возвращаем все необходимые данные
    return X, train, test, orig, df


if __name__ == '__main__':
    # Получаем все возвращаемые значения
    x, train, test, orig, df = main(
        train_model=True,
        plot=False,
        sub=True
    )

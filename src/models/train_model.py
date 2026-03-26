import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def train_demand_model(data_path='data/processed/preprocessed_data.csv'):
    """
    Загружает обработанные данные, обучает модель LightGBM и оценивает результат.
    """
    if not os.path.exists(data_path):
        print(f"Ошибка: Файл {data_path} не найден. Сначала запустите preprocessing.py.")
        return

    print(f"Загрузка данных из {data_path}...")
    df = pd.read_csv(data_path)
    df['Дата'] = pd.to_datetime(df['Дата'])

    # 1. Подготовка признаков
    # Определяем категориальные признаки
    cat_features = ['Пекарня', 'Номенклатура', 'Категория', 'Город', 'ДеньНедели']
    for col in cat_features:
        df[col] = df[col].astype('category')

    # Признаки для обучения (исключаем таргет и потенциальные утечки данных)
    # Выпуск и Остаток известны только по факту, поэтому их нельзя использовать для предсказания будущего спроса
    features = [
        'Пекарня', 'Номенклатура', 'Категория', 'Город', 'ДеньНедели', 'День',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_roll_mean_3'
    ]
    target = 'Продано'

    # 2. Хронологическое разделение на Train и Test
    # У нас данные за март 2026. Возьмем последние 3 доступных дня для теста.
    max_date = df['Дата'].max()
    test_start_date = max_date - pd.Timedelta(days=2)

    train = df[df['Дата'] < test_start_date].copy()
    test = df[df['Дата'] >= test_start_date].copy()

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    print(f"Период обучения: {train['Дата'].min().date()} - {train['Дата'].max().date()}")
    print(f"Период теста:    {test['Дата'].min().date()} - {test['Дата'].max().date()}")
    print(f"Размер Train: {X_train.shape[0]} строк, Test: {X_test.shape[0]} строк")

    # 3. Инициализация и обучение LightGBM
    # Модель хорошо работает с категориальными признаками "из коробки"
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        importance_type='gain',
        verbose=-1
    )

    print("\nОбучение модели LightGBM...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='mae'
    )

    # 4. Предсказание и оценка
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0) # Спрос не может быть отрицательным

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*30)
    print("ИТОГОВЫЕ МЕТРИКИ НА ТЕСТЕ:")
    print(f"MAE:  {mae:.2f} шт. (средняя ошибка в штуках на позицию)")
    print(f"RMSE: {rmse:.2f} шт.")
    print(f"R2:   {r2:.4f}")
    print("="*30)

    # 5. Анализ важности признаков
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("\nВАЖНОСТЬ ПРИЗНАКОВ:")
    print(importance.to_string(index=False))

    # 6. Сохранение предсказаний для детального изучения
    test['Predicted_Demand'] = y_pred.round(2)
    test['Error'] = (test['Predicted_Demand'] - test['Продано']).abs()

    output_results = 'reports/model_predictions_analysis.csv'
    test.to_csv(output_results, index=False, encoding='utf-8-sig')
    print(f"\nДетальные результаты теста сохранены в {output_results}")

    return model

if __name__ == "__main__":
    trained_model = train_demand_model()

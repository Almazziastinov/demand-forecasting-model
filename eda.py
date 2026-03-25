import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

def run_eda(file_path):
    """
    Performs exploratory data analysis on the given Excel file.

    Args:
        file_path (str): The path to the Excel file.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data from: {file_path}")
        print("-" * 50)

        # 1. Общая информация о данных
        print("\n--- General Information ---")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_output = buffer.getvalue()
        print(info_output)
        print("-" * 50)

        # 2. Описательная статистика (включая все типы данных)
        print("\n--- Descriptive Statistics ---")
        print(df.describe(include='all').to_markdown(numalign="left", stralign="left"))
        print("-" * 50)

        # 3. Проверка на пропущенные значения
        print("\n--- Missing Values ---")
        missing_values = df.isnull().sum()
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_values, 'Missing Percentage (%)': missing_percentage})
        print(missing_df[missing_df['Missing Count'] > 0].to_markdown(numalign="left", stralign="left"))
        print("-" * 50)

        # 4. Анализ уникальных значений для категориальных признаков (топ 10)
        print("\n--- Unique Values for Categorical Features (top 10) ---")
        for column in df.select_dtypes(include='object').columns:
            print(f"\nColumn: {column}")
            print(df[column].value_counts().head(10).to_markdown(numalign="left", stralign="left"))
        print("-" * 50)

        # 5. Гистограммы для числовых признаков
        print("\n--- Histograms for Numerical Features ---")
        numerical_cols = df.select_dtypes(include=['number']).columns
        if not numerical_cols.empty:
            # Графики будут сгенерированы и сохранены, но не показаны интерактивно
            # df[numerical_cols].hist(bins=20, figsize=(15, 10))
            # plt.suptitle('Histograms of Numerical Features')
            # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.savefig('demand-forecasting-model/histograms.png') # Сохранение в файл
            # plt.close()
            print("Histograms would be generated and saved as 'histograms.png'.")
        else:
            print("No numerical features found for histograms.")
        print("-" * 50)

        # 6. Если есть колонка с датой, строим график временного ряда
        # Попытаемся определить колонку с датой (может быть 'Date', 'Day', 'OrderDate' и т.п.)
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
        if date_columns:
            date_col = date_columns[0] # Берем первую найденную
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                print(f"\n--- Time Series Plot (using {date_col} as index) ---")
                # Предполагаем, что целевая переменная - это количество или объем продаж
                # Попробуем найти колонку, которая может быть таргетом, например 'Sales', 'Quantity', 'Demand'
                target_cols = [col for col in df.columns if 'sales' in col.lower() or 'quantity' in col.lower() or 'demand' in col.lower()]
                if target_cols:
                    target_col = target_cols[0]
                    # df[target_col].plot(figsize=(15, 6))
                    # plt.title(f'Time Series of {target_col}')
                    # plt.xlabel('Date')
                    # plt.ylabel(target_col)
                    # plt.grid(True)
                    # plt.savefig('demand-forecasting-model/time_series.png') # Сохранение в файл
                    # plt.close()
                    print(f"Time series plot for '{target_col}' would be generated and saved as 'time_series.png'.")
                else:
                    print("Could not identify a clear target column for time series plotting (e.g., 'Sales', 'Quantity', 'Demand').")
            except Exception as e:
                print(f"Could not convert '{date_col}' to datetime or plot time series: {e}")
        else:
            print("\nNo obvious date column found for time series analysis.")
        print("-" * 50)

        # 7. Тепловая карта корреляций для числовых признаков
        print("\n--- Correlation Heatmap ---")
        if not numerical_cols.empty and len(numerical_cols) > 1:
            # plt.figure(figsize=(12, 10))
            # sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
            # plt.title('Correlation Heatmap of Numerical Features')
            # plt.savefig('demand-forecasting-model/correlation_heatmap.png') # Сохранение в файл
            # plt.close()
            print("Correlation heatmap would be generated and saved as 'correlation_heatmap.png'.")
        elif len(numerical_cols) <= 1:
            print("Not enough numerical features for a correlation heatmap.")
        else:
            print("No numerical features found for correlation heatmap.")
        print("-" * 50)


    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    excel_file_path = 'beigl_data.xlsx'
    run_eda(excel_file_path)

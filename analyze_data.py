import pandas as pd
import io

file_path = 'beigl_data.xlsx'

try:
    df = pd.read_excel(file_path)
    print(f"Successfully loaded data from: {file_path}")

    print()
    print("--- First 5 rows ---")
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

    print()
    print("--- Data Info ---")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_output = buffer.getvalue()
    print(info_output)

    print()
    print("--- Missing Values ---")
    missing_values = df.isnull().sum()
    print(missing_values.to_markdown(numalign="left", stralign="left"))

except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

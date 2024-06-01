
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(file_path)

def process_data(df):
    """Procesa los datos: limpieza, escalado, etc."""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

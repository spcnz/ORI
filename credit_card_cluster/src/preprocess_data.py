import pandas as pd

def prepare_data(path):
    data = pd.read_csv(path)
    column_names = data.columns[data.isna().any()].tolist()
    for name in column_names:
        data[name].fillna(data[name].median(), inplace=True)

    return data
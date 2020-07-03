import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import numpy as np

def scale_values(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    data = pd.DataFrame(scaled, index=data.index, columns=data.columns)

    return data

def treat_outliers(data):
    data = data.applymap(lambda x: np.log(x + 1))

    return data

def remove_null_values(data):
    column_names = data.columns[data.isna().any()].tolist()
    for name in column_names:
        data[name].fillna(data[name].median(), inplace=True)

    return data

def prepare_data(path):
    data = pd.read_csv(path)
    # Removes non utile columns
    data = data.drop(['CUST_ID'], axis=1)
    data = remove_null_values(data)

    # data["AVG_MONTH_PURCHASES"] = data["PURCHASES"] / data["TENURE"]
    # data["AVG_MONTH_CASH_ADVANCE"] = data["CASH_ADVANCE"] / data["TENURE"]

    data = treat_outliers(data)
    data = scale_values(data)

    return data



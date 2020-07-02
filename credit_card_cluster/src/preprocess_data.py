import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

def scale_values(data):
    # data = data.apply(lambda x: (x - x.mean())/ x.std())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    data = pd.DataFrame(scaled, columns=data.columns)
    return data


def treat_outliers(data):
    data.plot(kind='box')

    for column in data.columns:
        upper_bound = data[column].quantile(0.95)
        lower_bound = data[column].quantile(0.05)
        data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
        data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])

    #Q1 = data.quantile(0.25)
    # Q3 = data.quantile(0.75)
    # IQR = Q3 - Q1
    sns.catplot(x="BALANCE", kind="box", data=data)
    data.plot(kind='box')
    #
    # print(data.shape)
    # print(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
    # data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    # print(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
    # print(data.shape)

    # perform quantile clipping in order to treat outliers
    # data = data.applymap(lambda x: np.log(x + 1))
    # sns.catplot(x="BALANCE", kind="box", data=data)
    # plt.show()
    plt.show()
    return data

def remove_null_values(data):
    column_names = data.columns[data.isna().any()].tolist()
    for name in column_names:
        data[name].fillna(data[name].median(), inplace=True)

    return data


def prepare_data(path):
    data = pd.read_csv(path)
    data = remove_null_values(data)
    # Removes non utile columns (CUSTID, TENURE)
    data = data.drop(['CUST_ID'], axis=1)
    data = treat_outliers(data)
    data = scale_values(data)

    print(data)
    #ODREDITI KORELACIJU I IZBACIITI NEPOTREBNE KOLONE

    return data



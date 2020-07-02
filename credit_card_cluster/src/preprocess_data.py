import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def scale_values(data):
    
    return data

def treat_outliers(data):
    print (data['BALANCE'])
    data.plot(kind='box')

    for column in data.columns:
        upper_bound = data[column].quantile(0.95)
        lower_bound = data[column].quantile(0.05)
        data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
        data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])

    # sns.catplot(x="BALANCE", kind="box", data=data)
    Q1 = data.quantile(0.25)
    print('ispod ovogaaa ehehhe')

    # Q3 = data.quantile(0.75)
    # IQR = Q3 - Q1
    # sns.catplot(x="BALANCE", kind="box", data=data)
    # plt.show()

    data.plot(kind='box')
    plt.show()
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


    #ODREDITI KORELACIJU I IZBACIITI NEPOTREBNE KOLONE

    return data



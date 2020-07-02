import numpy as np
import seaborn as sns
from preprocess_data import prepare_data

if __name__ == '__main__':
    file_path = "../data/credit_card_data.csv"
    data = prepare_data(file_path)
    print (data.isnull().sum())




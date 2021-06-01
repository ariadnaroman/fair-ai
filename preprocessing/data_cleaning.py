import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class DataCleaning:
    def __init__(self, dataset):
        self.name = 'DataCleaning'
        if dataset == 'german':
            self.raw_data = pd.read_csv(
                'D:\\Facultate\Disertatie\\experiments\\fair-ai\\datasets\\raw\\german\\german.csv')
            self.numerical_columns = [1,4,7,10,12,15,17]
            self.categorical_columns = [0,2,3,5,6,8,9,11,13,14,16,18,19]
            self.input_columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
            self.target_column = [20]
            self.target_column_name = 'good/bad'

    def get_preprocessed_data(self):
        x_processed_data, y_processed_data = self.split_input_target()
        x = self.preprocess_input_data(x_processed_data)
        y = self.preprocess_target_data(y_processed_data)
        return x, y

    def preprocess_input_data(self, input_data):
        # self.handle_missing_values(input_data)
        # self.encode_labels(input_data, self.categorical_columns)
        input_data = self.apply_one_hot_encoder(input_data, self.categorical_columns)
        input_data = self.normalize_values(input_data)
        return input_data

    def preprocess_target_data(self, target_data):
        self.encode_labels(target_data, [0])
        # target_data = self.apply_one_hot_encoder(target_data, [0])
        target_data = self.normalize_values(target_data)
        return target_data

    def split_input_target(self):
        processed = self.raw_data.copy(deep=True)
        df = pd.DataFrame(processed)
        y_data = df.iloc[:, self.target_column]
        x_data = df.iloc[:, self.input_columns]
        return x_data, y_data

    def normalize_values(self, dataset):
        scaler = MinMaxScaler()
        dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
        return dataset

    def apply_one_hot_encoder(self, dataset, columns):
        for i in columns:
            encoder = OneHotEncoder(sparse=False)
            encoded = pd.DataFrame(encoder.fit_transform(dataset.iloc[:, i].values.reshape(-1, 1)))
            encoded.columns = encoder.get_feature_names(dataset.columns[[i]])
            dataset = pd.concat([dataset, encoded], axis=1)
        for i in reversed(columns): # drop columns in reverse so that the index do not change
            dataset.drop(dataset.columns[[i]], axis=1, inplace=True)
        return dataset

    def encode_labels(self, merged_data, columns):
        le = LabelEncoder()
        for i in columns:
            merged_data.iloc[:, i] = le.fit_transform(merged_data.iloc[:, i].values.ravel())

    def handle_missing_values(self, dataset):
        dataset.replace([' NaN', 'NaN', 'NaN '], np.nan, inplace=True)

        num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        num_imputer.fit(dataset.iloc[:, self.numerical_columns])
        dataset.iloc[:, self.numerical_columns] = num_imputer.transform(dataset.iloc[:, self.numerical_columns])

        categ_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Others')
        categ_imputer.fit(dataset.iloc[:, self.categorical_columns])
        dataset.iloc[:, self.categorical_columns] = categ_imputer.transform(dataset.iloc[:, self.categorical_columns])


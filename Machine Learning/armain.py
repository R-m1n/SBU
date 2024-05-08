from math import sqrt
from typing import List

import numpy as np
import pandas as pd

class Scaler():
    def standardize(self, data: pd.DataFrame,
                    exclude: List[str] = None):
        
        for label in data.columns:
            if exclude == None or label not in exclude:
                mean = data[label].mean()
                std = data[label].std()
                
                data[label] = (data[label] - mean) / std
                
    def min_max(self, data: pd.DataFrame,
                exclude: List[str]):
        
        for label in data.columns:
            if exclude == None or label not in exclude:
                min = data[label].min()
                max = data[label].max()

                data[label] = (data[label] - min) / (max - min)


class DataLoader():
    def __init__(self, data: pd.DataFrame, target_label: str,
                 split: float = 0.8, basis: str = "linear",
                 gaussian_parameter: float = 1, multiquadratic_parameter: float = 1):
        
        self.data = data.copy(deep=True)

        self.target_label = target_label

        match basis.lower():
            case "gaussian":
                self._apply_gaussian_basis(gaussian_parameter)

            case "multiquadratic":
                self._apply_multiquadratic_basis(multiquadratic_parameter)

        self.data_size = len(self.data)

        self.train_size = int(self.data_size * split)
        self.val_size = self.data_size - self.train_size

        train_data = self.data[:self.train_size]
        val_data = self.data[self.train_size:]

        self.input_train = np.array(train_data.drop(target_label, axis=1))
        self.target_train = np.array(train_data[target_label])
        
        self.input_val = np.array(val_data.drop(target_label, axis=1))
        self.target_val = np.array(val_data[target_label])

        self.features = len(self.data.columns) - 1

    def head(self, n: int):
        return self.data.head(n)

    def get_training(self):
        return self.input_train, self.target_train
    
    def get_validation(self):
        return self.input_val, self.target_val
    
    def get_feature_mean(self, label: str):
        return self.data[label].mean()

    def get_feature_std(self, label: str):
        return self.data[label].std()
    
    def _apply_gaussian_basis(self, parameter):
        for label in self.data.columns:
            if label != self.target_label:
                mean = self.data[label].mean()

                num = -1 * np.power(self.data[label] - mean, 2)

                denom = (2 * pow(parameter, 2))

                self.data[label] = np.exp(num / denom)

    def _apply_multiquadratic_basis(self, parameter):
        for label in self.data.columns:
            if label != self.target_label:
                mean = self.data[label].mean()

                term = np.power(self.data[label] - mean, 2)

                self.data[label] = np.sqrt(term + pow(parameter, 2))


class Regression():
    def __init__(self):
        self.w = None
        self.b = 0

        self._model = lambda x: self.w.dot(x) + self.b

    def train(self, data_loader: DataLoader, epoch = 10,
              learning_rate = 0.01, lasso = 0, ridge = 0):
        
        input_train, target_train = data_loader.get_training() 
        input_val, target_val = data_loader.get_validation()

        self.w = np.zeros(data_loader.features)

        train_size = data_loader.train_size
        val_size = data_loader.val_size

        train_rmsd = []
        val_rmsd = []

        sign = np.vectorize(lambda x : 1 if x >= 0 else -1)

        for _ in range(epoch):
            train_cost = 0
            val_cost = 0

            for idx in range(train_size):
                input, target = input_train[idx], target_train[idx]

                prediction = self._model(input)

                error = prediction - target

                train_loss = np.power(error, 2) / 2

                train_cost += train_loss

                l1_update = lasso * sign(self.w)

                l2_update = (2 * ridge) * self.w

                regularization_term = l1_update + l2_update

                update_term = (error * input) + regularization_term

                self.w -= learning_rate * update_term

                self.b -= learning_rate * error


            for idx in range(val_size):
                input, target = input_val[idx], target_val[idx]

                prediction = self._model(input)

                error = prediction - target

                val_loss = np.power(error, 2) / 2

                val_cost += val_loss


            train_rmsd.append(sqrt(train_cost / train_size))
            val_rmsd.append(sqrt(val_cost / val_size))

        return train_rmsd, val_rmsd
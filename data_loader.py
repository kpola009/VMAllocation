import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader

from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = StandardScaler()

def clean_data(file_name):
    # Read data
    df = pd.read_csv(file_name, sep=';')

    df['CPU usage [%]'] = df['CPU usage [%]'].str.replace(',', '.')
    df['Memory usage [%]'] = df['Memory usage [%]'].str.replace(',', '.')

    df = df.astype({'CPU cores': int, 'CPU capacity provisioned [MHZ]': int,
                    'CPU usage [MHZ]': int, 'CPU usage [%]': float, 'Memory capacity provisioned [KB]': int,
                    'Memory usage [KB]': int, 'Memory usage [%]': float, 'Disk read throughput [KB/s]': int,
                    'Disk write throughput [KB/s]': int, 'Disk size [GB]': int,
                    'Network received throughput [KB/s]': int,
                    'Network transmitted throughput [KB/s]': int})

    df = df.drop(columns=['Timestamp', 'CPU cores', 'CPU capacity provisioned [MHZ]',
                          'CPU usage [%]', 'Memory capacity provisioned [KB]', 'Memory usage [%]', 'Disk size [GB]'])

    return df

#TODO comments for multivariate timeseries target-columns and drop_targets
def generate_sequence(df: pd.DataFrame, tw: int, pw: int, target_columns, drop_targets=False):

    '''

    :param df: Pandas DataFrame of the  time-series
    :param tw: Training Window - Integer defining how many steps to look back
    :param pw: Prediction Window - Integer defining how many steps forward to predict
    :param target_columns:
    :param drop_targets:
    :return: Dictionary of sequences and target for all sequences
    '''

    data = dict()
    L = len(df)

    for i in range(L-tw):
        if drop_targets:
            df.drop(target_columns, axis=1, inplace=True)

        sequence = df[i:i+tw].values

        target = df[i+tw:i+tw+pw][target_columns].values
        data[i] = {'sequence': sequence, 'target': target}

    return data


class SequenceDataset(Dataset):

    def __init__(self, df):
        self.data = df

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])

    def __len__(self):
        return len(self.data)


def get_data_univariate(df: pd.DataFrame, batch_size, split,  tw: int, pw: int, target_columns):

    clean_df = clean_data(df)
    forecast_column = clean_df['CPU usage [MHZ]']

    forecast_data = scaler.fit_transform(forecast_column.values.reshape(-1,1))

    sequences = generate_sequence(pd.DataFrame(forecast_data), tw, pw, target_columns)
    dataset = SequenceDataset(sequences)

    train_len = int(len(dataset) * split)
    lens = [train_len, len(dataset) - train_len]
    train_ds, test_ds = random_split(dataset, lens)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return trainloader, testloader





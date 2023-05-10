import argparse
import torch
from torch import nn
import os
import pandas as pd

from data_loader import get_data_univariate
from models import LSTMForecaster

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    trainloader, testloader = get_data_univariate(args.csv_file, args.batch_size, args.split, args.tw, args.pw, args.tc)

    model = LSTMForecaster(n_features= args.input_dim, n_hidden=args.hidden_size, n_outputs=args.output_dim, sequence_len=args.tw, n_deep_layers=args.hidden_layers, n_lstm_layers=args.lstm_layers, dropout=args.dropout).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

    # Lists to store training and validation losses
    t_losses, v_losses = [], []
    # Loop over epochs
    for epoch in range(args.epochs):
        train_loss, valid_loss = 0.0, 0.0

        model.train()
        # Loop over train dataset
        for x, y in trainloader:
            optimizer.zero_grad()
            # move inputs to device
            x = x.to(device)
            y = y.squeeze().to(device)
            # Forward Pass
            preds = model(x).squeeze()
            loss = criterion(preds, y)  # compute batch loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = train_loss / len(trainloader)
        t_losses.append(epoch_loss)

        # validation step
        model.eval()
        # Loop over validation dataset
        for x, y in testloader:
            with torch.no_grad():
                x, y = x.to(device), y.squeeze().to(device)
                preds = model(x).squeeze()
                error = criterion(preds, y)
            valid_loss += error.item()
        valid_loss = valid_loss / len(testloader)
        v_losses.append(valid_loss)


        print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')

    torch.save(model.state_dict(), os.path.join(
        args.model_path, 'model.ckpt'
    ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/models', help='path for saving trained models')
    parser.add_argument('--csv_file', type=str, help='Input data as csv')


    # Model Parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--tw', type=int, default=48)
    parser.add_argument('--pw', type=int, default=1)
    parser.add_argument('--tc', type=int, default=0)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--hidden_layers', type=int, default=5)
    parser.add_argument('--lstm_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()
    print(args)
    main(args)







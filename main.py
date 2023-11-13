#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:21:36 2023

@author: shavinkalu
"""

import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
# Define the ticker symbol for S&P 500
ticker_symbol = "^GSPC"

# Define the date range: current date and 3 years ago
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=20*365)

# Use yfinance to download the stock sp500_data
sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Display the sp500_data
print(sp500_data.head())

sp500_data.describe()
sp500_data.info()

# Technical Indicators
sp500_data['RSI'] = sp500_data.ta.rsi()
sp500_data['SMA'] = sp500_data.ta.sma(close='Close', length=20)
sp500_data['EMA'] = sp500_data.ta.ema(close='Close', length=20)
sp500_data['MACD'] = sp500_data.ta.macd(close='Close')['MACD_12_26_9']
sp500_data['ATR'] = sp500_data.ta.atr()
sp500_data['Stoch'] = sp500_data.ta.stoch()[['STOCHk_14_3_3']]
# Calculate the daily price change ratio (in percentage)
sp500_data['PCR'] = sp500_data['Close'].pct_change()*100
# Drop NaN values
sp500_data.dropna(inplace=True)

# Selecting features to normalize
features = ['RSI', 'SMA', 'EMA', 'MACD', 'ATR', 'Stoch', 'PCR' ]

target_column = 'PCR'  # The column you are predicting

# Initialize a scaler
# scaler = MinMaxScaler()
scaler = StandardScaler()

sp500_data_scaled = sp500_data.copy()
# Fit and transform the features
sp500_data_scaled[features] = scaler.fit_transform(sp500_data[features])

# Calculating the correlation matrix
corr_matrix = sp500_data_scaled[features].corr()

# Generating a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


#The closing price is the last price at which the stock is traded during the regular trading day.
# A stock’s closing price is the standard benchmark used by investors to track its performance over time.
# Plotting
plt.figure(figsize=(10, 6))
sp500_data['Close'].plot(title='S&P 500 Closing Prices Over The Last 3 Years')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()



# Plotting the daily change
plt.figure(figsize=(10, 6))
sp500_data['PCR'].plot(title='Daily Change in S&P 500 Closing Prices Over The Last 3 Years')
plt.xlabel('Date')
plt.ylabel('Daily Change in Close Price')
plt.grid(True)
plt.show()


# Splitting the data - assume 'target_column' is the column you want to predict
train_data, test_data = train_test_split(sp500_data[features], test_size=0.2, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.25, shuffle=False)  # 0.25 * 0.8 = 0.2

# prevent data leakage
scaler.fit(train_data)
scaled_train = pd.DataFrame(scaler.transform(train_data), columns=features)
scaled_val = pd.DataFrame(scaler.transform(val_data), columns=features)
scaled_test = pd.DataFrame(scaler.transform(test_data), columns=features)

# Define a function to create sequences
def create_sequences(input_data, target_column, sequence_length):
    sequences = []
    labels = []
    for i in range(len(input_data) - sequence_length):
        sequence = input_data.iloc[i:i + sequence_length]
        label = input_data.iloc[i + sequence_length][target_column]
        sequences.append(sequence)
        labels.append(label)
    return sequences, labels

# Prepare the sequences
sequence_length = 60  # sequence length
train_seq, train_labels = create_sequences(scaled_train, target_column, sequence_length)
val_seq, val_labels = create_sequences(scaled_val, target_column, sequence_length)
test_seq, test_labels = create_sequences(scaled_test, target_column, sequence_length)


class StockDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.tensor(sequence.values).float(), torch.tensor(label).float()

# Create datasets
train_dataset = StockDataset(train_seq, train_labels)
val_dataset = StockDataset(val_seq, val_labels)
test_dataset = StockDataset(test_seq, test_labels)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Model parameters
input_dim = len(features)
hidden_dim = 64
num_layers = 2
output_dim = 1

# Initialize the model
model = StockPredictor(input_dim, hidden_dim, num_layers, output_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for sequences, labels in train_loader:
        # Forward pass
        outputs = model(sequences).squeeze()
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    val_losses = []
    for sequences, labels in val_loader:
        outputs = model(sequences).squeeze()
        loss = criterion(outputs, labels)
        val_losses.append(loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f'Average Validation Loss: {avg_val_loss:.4f}')
    
    
# Assuming 'test_loader' is your DataLoader for the test set
model.eval()  # Set the model to evaluation mode

actuals = []
predictions = []

with torch.no_grad():
    for sequences, labels in test_loader:
        outputs = model(sequences)
        outputs = outputs.squeeze()  # Adjust the shape if necessary
        # scaler.inverse_transform(outputs)
        actuals.extend(labels.tolist())
        predictions.extend(outputs.tolist())

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted PCR Values')
plt.xlabel('Samples')
plt.ylabel('PCR Value')
plt.legend()
plt.show()

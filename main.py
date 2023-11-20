#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:21:36 2023

@author: shavinkalu
"""
#%%
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK,STATUS_FAIL, hp, fmin, tpe, Trials
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
# set seed for reproducability

seed = 1234

np.random.seed(seed) 
torch.manual_seed(seed)
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


# Seasonal decomposition of closing prices
closing_prices = sp500_data['Close']
decomposition = seasonal_decompose(closing_prices, model='multiplicative', period=252)  # Assuming 252 trading days in a year


# Plot the seasonal decomposition
plt.rcParams.update({'figure.figsize': (10,10)})
decomposition.plot()
plt.show()

# Technical Indicators
sp500_data['RSI'] = sp500_data.ta.rsi()
sp500_data['SMA'] = sp500_data.ta.sma(close='Close', length=20)
sp500_data['EMA'] = sp500_data.ta.ema(close='Close', length=20)
sp500_data['MACD'] = sp500_data.ta.macd(close='Close')['MACD_12_26_9']
sp500_data['ATR'] = sp500_data.ta.atr()
sp500_data['Stoch'] = sp500_data.ta.stoch()[['STOCHk_14_3_3']]
# Calculate the daily price change ratio (in percentage)

sp500_data['PCR'] = np.log(sp500_data['Close']).pct_change()*100
# Drop NaN values
sp500_data.dropna(inplace=True)

# Selecting features to normalize
features = ['RSI', 'SMA', 'EMA', 'MACD', 'ATR', 'Stoch', 'PCR' , 'Close']
features = ['PCR' ]
target_column = 'PCR'  # The column you are predicting

# Initialize a scaler
scaler = MinMaxScaler()
# scaler = StandardScaler()

sp500_data_scaled = sp500_data.copy()
# Fit and transform the features
sp500_data_scaled[features] = scaler.fit_transform(sp500_data[features])

# Calculating the correlation matrix
corr_matrix = sp500_data_scaled[features].corr()

# Generating a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Selecting features to normalize
features = ['RSI', 'MACD', 'ATR', 'Stoch', 'PCR' , 'Close']
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

#%%
# Splitting the data - assume 'target_column' is the column you want to predict
train_data, test_data = train_test_split(sp500_data[features], test_size=0.2, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.25, shuffle=False)  # 0.25 * 0.8 = 0.2

# prevent data leakage
scaler.fit(train_data)
scaled_train = pd.DataFrame(scaler.transform(train_data), columns=features)
scaled_val = pd.DataFrame(scaler.transform(val_data), columns=features)
scaled_test = pd.DataFrame(scaler.transform(test_data), columns=features)

target_scaler = MinMaxScaler()
 # target_scaler = StandardScaler()
target_scaler.fit(train_data[[target_column]])


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
# Prepare the sequences
sequence_length = 60  # sequence length
train_seq, train_labels = create_sequences(scaled_train, target_column, sequence_length)
val_seq, val_labels = create_sequences(scaled_val, target_column, sequence_length)
test_seq, test_labels = create_sequences(scaled_test, target_column, sequence_length)

# Create datasets
train_dataset = StockDataset(train_seq, train_labels)
val_dataset = StockDataset(val_seq, val_labels)
test_dataset = StockDataset(test_seq, test_labels)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%%
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, lr_scheduler = None):
    """
    Trains a PyTorch model and evaluates it on the validation set.

    Parameters:
    model: The PyTorch model to train.
    train_loader: DataLoader for the training set.
    val_loader: DataLoader for the validation set.
    criterion: Loss function.
    optimizer: Optimization algorithm.
    num_epochs: Number of epochs to train for.

    Returns:
    A tuple containing the training and validation loss history.
    """
    # Track loss history
    train_loss_history = []
    val_loss_history = []
    mse_history = []
    rmse_history = []
    

    for epoch in range(num_epochs):
        print( optimizer.param_groups[0]["lr"])
        # Training phase
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            outputs = outputs.squeeze(1) if outputs.size(1) == 1 else outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Calculate average loss over the epoch
        train_avg_loss = train_loss / len(train_loader)
        train_loss_history.append(train_avg_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences)
                outputs = outputs.squeeze(1) if outputs.size(1) == 1 else outputs.squeeze().squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_labels.extend(labels.tolist())
                all_predictions.extend(outputs.flatten().tolist())
                
        # reduce learning rate based on schedular
        if lr_scheduler:
          lr_scheduler.step()
          
        # Calculate average validation loss
        val_avg_loss = val_loss / len(val_loader)
        val_loss_history.append(val_avg_loss)
        # calculate mse and and rmse
        mse = mean_squared_error(all_labels, all_predictions)
        rmse = mse ** 0.5
        mse_history.append(mse)
        rmse_history.append(rmse)
        # Print epoch stats
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_avg_loss:.4f}, Validation Loss: {val_avg_loss:.4f}')
    
    return train_loss_history, val_loss_history, mse_history, rmse_history




#add drop out to improve model





# class LSTMModel(nn.Module):
#     def __init__(self, input_dim):
#         super(LSTMModel, self).__init__()

#         # First LSTM layer with 128 units, returns sequences
#         self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        
#         # Second LSTM layer with 64 units, does not return sequences
#         self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        
#         # Dense layer with 25 units
#         self.fc1 = nn.Linear(64, 25)
        
#         # Final Dense layer with 1 unit
#         self.fc2 = nn.Linear(25, 1)

#     def forward(self, x):
#         # Pass data through LSTM layers
#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)

#         # We only use the output of the last time step
#         x = x[:, -1, :]

#         # Pass through the fully connected layers
#         x = self.fc1(x)
#         x = self.fc2(x)

#         return x


# Model parameters
input_dim = len(features)
hidden_dim = 8
num_layers = 2
output_dim = 1

# Initialize the model
# model = StockPredictor(input_dim, hidden_dim, num_layers, output_dim)

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
   
    
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out
    
    
# Model parameters
input_dim = len(features)  # Number of input features
hidden_dim = 8  # Number of hidden units
output_dim = 1  # One output
lr=0.0001
epochs = 5
criterion=nn.MSELoss()
lstm_model = LSTMModel(input_dim, hidden_dim, nu5_layers, output_dim)

# Initialize the RNN model
rnn_model = RNNModel(input_dim, hidden_dim, num_layers, output_dim)

# Initialize the GRU model
gru_model = GRUModel(input_dim, hidden_dim, num_layers, output_dim)


# Train RNN model
rnn_train_loss, rnn_val_loss, rnn_mse, rnn_rmse = train_model(
    model=rnn_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=torch.optim.Adam(rnn_model.parameters(), lr=lr),
    num_epochs=epochs
)


# Train LSTM model
lstm_train_loss, lstm_val_loss, lstm_mse, lstm_rmse = train_model(
    model=lstm_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=torch.optim.Adam(lstm_model.parameters(), lr=lr),
    num_epochs=epochs
)

# Train GRU model
gru_train_loss, gru_val_loss, gru_mse, gru_rmse = train_model(
    model=gru_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=torch.optim.Adam(gru_model.parameters(), lr=lr),
    num_epochs=epochs  # Or any number of epochs that suits your dataset
)


lstm_rmse = lstm_rmse[-1]
rnn_rmse = rnn_rmse[-1]
gru_rmse = gru_rmse[-1]
lstm_mse = lstm_mse[-1]
rnn_mse = rnn_mse[-1]
gru_mse = gru_mse[-1]

# Print the performance
print(f'LSTM MSE: {lstm_mse:.4f}, RMSE: {lstm_rmse:.4f}')
print(f'RNN MSE: {rnn_mse:.4f}, RMSE: {rnn_rmse:.4f}')
print(f'GRU MSE: {gru_mse:.4f}, RMSE: {gru_rmse:.4f}')

# We choose LSTM architecture since it had the lowest MSE


num_layers = 3  # 5 layers

ml_lstm_model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim,)

# Train ml lstm model
ml_lstm_train_loss, ml_lstm_val_loss, ml_lstm_mse, ml_lstm_rmse = train_model(
    model=ml_lstm_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=torch.optim.Adam(ml_lstm_model.parameters(), lr=lr),
    num_epochs=epochs  # Or any number of epochs that suits your dataset
)



ml_lstm_mse = ml_lstm_mse[-1]
ml_lstm_rmse = ml_lstm_rmse[-1]
print(f'ML LSTM MSE: {ml_lstm_mse:.4f}, RMSE: {ml_lstm_rmse:.4f}')

class DropoutLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(DropoutLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

num_layers = 3  # 5 layers
dropout_rate = 0.3
output_dim = 1
do_ml_lstm_model = DropoutLSTMModel(input_dim, hidden_dim, num_layers, output_dim ,dropout_rate)

# Train ml lstm model
do_ml_lstm_train_loss, do_ml_lstm_val_loss, do_ml_lstm_mse, do_ml_lstm_rmse = train_model(
    model=do_ml_lstm_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=torch.optim.Adam(do_ml_lstm_model.parameters(), lr=lr),
    num_epochs=epochs  # Or any number of epochs that suits your dataset
)
do_ml_lstm_mse = do_ml_lstm_mse[-1]
do_ml_lstm_rmse = do_ml_lstm_rmse[-1]
print(f'Drop Out ML LSTM MSE: {do_ml_lstm_mse:.4f}, RMSE: {do_ml_lstm_rmse:.4f}')


#%%
# Hyperparameter optimisation
def objective(params,scaled_train,scaled_val, features, target_column):
    # Unpack your parameters
    num_layers = int(params['num_layers'])
    hidden_dim = int(params['hidden_dim'])
    dropout_rate = params['dropout_rate']
    lr = params['learning_rate']
    batch_size = int(params['batch_size'])
    sequence_length = int(params['sequence_length'])
    # Model parameters
    input_dim = len(features)  # Number of input features
    output_dim = 1  # One output
    epochs = 3
    
    # Prepare the sequences
    train_seq, train_labels = create_sequences(scaled_train, target_column, sequence_length)
    val_seq, val_labels = create_sequences(scaled_val, target_column, sequence_length)
 
    # Create datasets
    train_dataset = StockDataset(train_seq, train_labels)
    val_dataset = StockDataset(val_seq, val_labels)
 
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define model
    model =  DropoutLSTMModel(input_dim, hidden_dim, num_layers, output_dim,dropout_rate)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize variables to track the best validation loss and improvement threshold
    best_val_loss = np.inf
    loss_improvement_threshold = 0.1  # Change as needed, represents 1% improvement

    for epoch in range(epochs):
        for sequences, labels in train_loader:
           optimizer.zero_grad()
           outputs = model(sequences)
           outputs = outputs.squeeze(1) if outputs.size(1) == 1 else outputs.squeeze()
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences)
                outputs = outputs.squeeze(1) if outputs.size(1) == 1 else outputs.squeeze()
                val_loss += criterion(outputs, labels).item()

        # Calculate average validation loss
        val_avg_loss = val_loss / len(val_loader)

        # Early stopping condition for first 3 epochs
        if epoch < 3:
            improvement = (best_val_loss - val_avg_loss) / best_val_loss
            if improvement < loss_improvement_threshold:
                return {'loss': val_avg_loss, 'status': STATUS_FAIL}  # Skip this set of parameters

        # Update best_val_loss
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss

    # Return the final average validation loss
    return {'loss': best_val_loss, 'status': STATUS_OK}


space = {
    'num_layers': hp.choice('num_layers', [2, 3, 4]),
    'hidden_dim': hp.choice('hidden_dim', [4, 8, 16,32]),
    'learning_rate': hp.loguniform('learning_rate', -5, -1),  # e^-5 to e^1
    'dropout_rate': hp.uniform('dropout_rate', 0, 0.5),
    'batch_size': hp.choice('batch_size', [16, 32, 64]),
    'sequence_length': hp.choice('sequence_length', [10, 30, 60, 90])  # Assuming fixed for now
}                      
                                   
# Object to hold the history
trials = Trials()

# Run the optimization
best = fmin(
    fn=lambda params: objective(params, scaled_train=scaled_train, scaled_val=scaled_val, features=features ,target_column=target_column),
    space=space,
    algo=tpe.suggest,
    max_evals=50,  # Number of iterations
    trials=trials
)

print("Best: ", best)             
                                   
#%%
# Extract the actual parameter values
best_params = {
    'num_layers': [2, 3, 4][best['num_layers']],
    'hidden_dim': [4, 8, 16, 32][best['hidden_dim']],
    'learning_rate': best['learning_rate'],  # No change needed, already a direct value
    'dropout_rate': best['dropout_rate'],    # No change needed, already a direct value
    'batch_size': [16, 32, 64][best['batch_size']],
    'sequence_length': [10, 30, 60, 90][best['sequence_length']]
}

print("Best parameters: ", best_params)

num_layers = int(best_params['num_layers'])
hidden_dim = int(best_params['hidden_dim'])
dropout_rate = best_params['dropout_rate']
lr = best_params['learning_rate']
batch_size = int(best_params['batch_size'])
sequence_length =int(best_params['sequence_length'])
epochs = 5 #train final_model for a longer ephochs

train_seq, train_labels = create_sequences(scaled_train, target_column, sequence_length)
val_seq, val_labels = create_sequences(scaled_val, target_column, sequence_length)
test_seq, test_labels = create_sequences(scaled_test, target_column, sequence_length)

# Create datasets
train_dataset = StockDataset(train_seq, train_labels)
val_dataset = StockDataset(val_seq, val_labels)
test_dataset = StockDataset(test_seq, test_labels)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Train with best parameters
final_model = DropoutLSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout_rate)
optimizer=torch.optim.Adam(final_model.parameters(), lr=lr)
# Train ml lstm model
final_loss, final_loss, final_mse, final_rmse = train_model(
    model=final_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=epochs,  
    lr_scheduler= optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.5,total_iters=epochs)
)
final_mse = final_mse[-1]
final_rmse = final_rmse[-1]
print(f'Final model MSE: {final_mse:.4f}, RMSE: {final_rmse:.4f}')

#%%%
actuals = []
predictions = []

#plot loss and accuracy curve of final model
final_model.eval()
with torch.no_grad():
    for sequences, labels in test_loader:
        outputs = final_model(sequences)
        outputs = outputs.squeeze()
        # Adjust the shape if necessary
        # scaler.inverse_transform(outputs)
        #actuals.extend(labels)
        #predictions.extend(outputs)
        
        actuals.extend(target_scaler.inverse_transform(labels.reshape(1, -1)).flatten().tolist())
        predictions.extend(target_scaler.inverse_transform(outputs.reshape(1, -1)).flatten().tolist())

mean_squared_error(actuals, predictions)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted PCR Values')
plt.xlabel('Samples')
plt.ylabel('PCR Value')
plt.legend()
plt.show()



plot_df = pd.DataFrame(test_data['Close'][sequence_length:], index = test_data[sequence_length:].index)
plot_df['pred_PCR'] = predictions
plot_df['Predicted_Close'] = np.log(plot_df[ 'Close']).shift(1) * (1 + np.exp(plot_df['pred_PCR']) / 100)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(plot_df['Predicted_Close'], label='Actual')
plt.plot(plot_df['Predicted_Close'], label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Closing Values')
plt.xlabel('Samples')
plt.ylabel('PCR Value')
plt.legend()
plt.show()
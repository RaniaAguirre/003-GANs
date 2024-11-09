# Functions
import tensorflow as tf
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import scipy.optimize as opt
from scipy.optimize import minimize
from datetime import date
import yfinance as yf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU, BatchNormalization
from collections import namedtuple
import ta

def prepare_stock_data(ticker, start_date, end_date, train_fraction=0.8):
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock = pd.DataFrame(stock['Adj Close'])
    log_ret = pd.DataFrame(np.log(stock / stock.shift(1))).dropna()
    log_ret_norm = (log_ret - log_ret.mean()) / log_ret.std()

    x_train = pd.DataFrame(log_ret_norm.iloc[:int(len(stock) * train_fraction)].to_numpy())
    x_test = pd.DataFrame(log_ret_norm.iloc[int(len(stock) * train_fraction):].to_numpy())  
    return x_train, x_test

def stock(ticker, start_date, end_date, train_fraction=0.8):
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock = pd.DataFrame(stock['Adj Close'])
    log_ret = pd.DataFrame(np.log(stock / stock.shift(1))).dropna()
    log_ret_norm = (log_ret - log_ret.mean()) / log_ret.std()

    return stock, log_ret

def discriminator():
    model = Sequential([
        Dense(128, activation="relu", input_shape=(252, 1)),  # Adjust input_shape for sequence length
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")  # Use sigmoid for binary classification (real/fake)
    ])
    return model

def generator():
    model = Sequential([
        Dense(128, activation = "relu", input_shape=(500,)),
        Dense(256, activation="relu"),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(252, activation="linear")  # Output size matching sequence length (252 time steps)
    ])
    return model

@tf.function
def train_step(data, batch_size = 100):
    noise = tf.random.normal([batch_size, 500])
    #for:
    #    ...
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = gen_model(noise, training = True)
        y_real = disc_model(data, training = True)
        y_fake = disc_model(generated_data, training = True)
    
        gen_loss = -tf.math.reduce_mean(y_fake)
        disc_loss = tf.reduce_mean(y_fake) - tf.reduce_mean(y_real)
        
        
    gradients_gen = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_gen, gen_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_disc, disc_model.trainable_variables))
    
    return gen_loss, disc_loss
    
def simulate_price_scenarios(generated_series, log_ret, stock):
    scenarios = ((generated_series + log_ret.mean().values[0]) * log_ret.std().values[0]).tolist()
    S0 = stock['Adj Close'].iloc[0]
    
    data_n = []
    
    for scenario in scenarios:
        prices = [S0]
        for log_return in scenario:
            next_price = prices[-1] * np.exp(log_return)
            #next_price.iloc[i] = prices.iloc[i-1] * (np.exp(log_return.iloc[i]))
            prices.append(next_price)
        data_n.append(prices)
    data = pd.DataFrame(data_n)
    return data

def backtest_strategy(data, stop_loss_levels, take_profit_levels):
    results = {}

    # Iterar sobre niveles de stop-loss y take-profit
    for sl, tp in zip(stop_loss_levels, take_profit_levels):
        ratios = []
        for j in range(len(data)):
            data_gen_df = pd.DataFrame()
            data_gen_df["Close"] = data.iloc[[j], :].T
            bb = ta.volatility.BollingerBands(close=data_gen_df.Close, window=25, window_dev=2)
            data_gen_df["SELL_SIGNAL"] = bb.bollinger_hband_indicator()
            data_gen_df["BUY_SIGNAL"] = bb.bollinger_lband_indicator()
            data_gen_df = data_gen_df.dropna()

            # Inicializar variables de backtesting
            capital = 1_000_000
            active_positions = []
            COM = .25 / 100
            portfolio_value = [capital]
            n_shares = 100
            Position = namedtuple("Position", ["ticker", "price", "n_shares", "timestamp"])

            for i, row in data_gen_df.iterrows():
                trading_signal = row['BUY_SIGNAL']
                trading_signal_short = row['SELL_SIGNAL']

                # Cerrar posiciones largas
                for position in active_positions.copy():
                    if row.Close > position.price * (1 + tp):
                        capital += row.Close * n_shares * (1 - COM)
                        active_positions.remove(position)
                    elif row.Close < position.price * (1 - sl):
                        capital += row.Close * n_shares * (1 - COM)
                        active_positions.remove(position)

                # Señal de compra
                if trading_signal == True:
                    cost = row.Close * n_shares * (1 + COM)
                    if capital > cost and len(active_positions) < 100:
                        capital -= cost
                        active_positions.append(Position(ticker="AAPL", price=row.Close, n_shares=n_shares, timestamp=row.name))

                # Calcular valor del portafolio
                long = sum([position.n_shares * row.Close for position in active_positions])
                equity = capital + long 
                portfolio_value.append(equity)

            # Liquidar posiciones al final del backtest
            for position in active_positions.copy():
                capital += row.Close * position.n_shares * (1 - COM)
                active_positions.remove(position)

            # Calcular el Sharpe Ratio
            returns = pd.Series(portfolio_value).pct_change().dropna()
            mean_return = returns.mean()
            std_dev = returns.std()
            sharpe_ratio = mean_return / std_dev if std_dev != 0 else np.nan
            ratios.append(sharpe_ratio)

        # Guardar Sharpe Ratio promedio para el par (sl, tp)
        results[f"{sl}-{tp}"] = np.mean(ratios)

    return results
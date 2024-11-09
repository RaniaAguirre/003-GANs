import numpy as np
import pandas as pd
import ta
from collections import namedtuple
from FunctionsGANs import generator, prepare_stock_data, stock, simulate_price_scenarios, backtest_strategy

stock, log_ret = stock('AAPL', '2014-11-01', '2024-11-17')

noise = np.load('noise.npy')
generated_series = np.load('generated_series.npy')

data = simulate_price_scenarios(generated_series, log_ret, stock)

stop_loss_levels = [.01,.02,.03,.04,.05,.06,.07,.08,.09,.1]
take_profit_levels = [.01,.02,.03,.04,.05,.06,.07,.08,.09,.1]

results = backtest_strategy(data, stop_loss_levels, take_profit_levels)
results

best_sl_tp = max(results, key=results.get)
best_sharpe_ratio = results[best_sl_tp]

print("Mejor Stop-Loss y Take-Profit:")
print(f"Stop-Loss y Take-Profit: {best_sl_tp}")
print(f"Sharpe Ratio promedio: {best_sharpe_ratio}")
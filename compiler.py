import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import stats
import os
import csv

np.set_printoptions(legacy='1.25')

def main():
    compile()

def compile():
    #for every csv in backtest_output, we need to calc the mean
    filename = 'compiled_results.csv'
    if os.path.exists(filename):
        os.remove(filename)
    for file in os.listdir('backtest_output'):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join('backtest_output', file))
            model_mean = df['Model Result'].mean()
            model_std = df['Model Result'].std()
            model_sharpe = df['Model Sharpe'].mean()
            model_sharpe_std = df['Model Sharpe'].std()
            buy_hold_mean = df['Buy/Hold Result'].mean()
            buy_hold_std = df['Buy/Hold Result'].std()
            buy_hold_sharpe = df['Buy/Hold Sharpe'].mean()
            buy_hold_sharpe_std = df['Buy/Hold Sharpe'].std()
            if "SPY Buy/Hold Result" in df.columns:
                spy_mean = df['SPY Buy/Hold Result'].mean()
                spy_std = df['SPY Buy/Hold Result'].std()
                spy_sharpe = df['SPY Sharpe'].mean()
                spy_sharpe_std = df['SPY Sharpe'].std()
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if csvfile.tell() == 0:  # Check if file is empty
                    if "SPY Buy/Hold Result" in df.columns:
                        writer.writerow(['CSV file','Model Mean', 'Model Std', 'Model Sharpe', 'Model Sharpe Std',
                                         'Buy/Hold Mean', 'Buy/Hold Std', 'Buy/Hold Sharpe', 'Buy/Hold Sharpe Std',
                                         'SPY Buy/Hold Mean', 'SPY Buy/Hold Std', 'SPY Sharpe', 'SPY Sharpe Std'])
                    else:
                        writer.writerow(['CSV file','Model Mean', 'Model Std', 'Model Sharpe', 'Model Sharpe Std',
                                         'Buy/Hold Mean', 'Buy/Hold Std', 'Buy/Hold Sharpe', 'Buy/Hold Sharpe Std'])
                elif "SPY Buy/Hold Result" in df.columns:
                    writer.writerow([file, round(model_mean,2), round(model_std,2), round(model_sharpe,2), round(model_sharpe_std,2),
                                     round(buy_hold_mean,2), round(buy_hold_std,2), round(buy_hold_sharpe,2), round(buy_hold_sharpe_std,2),
                                     round(spy_mean,2), round(spy_std,2), round(spy_sharpe,2), round(spy_sharpe_std,2)])
                else:
                    writer.writerow([file, round(model_mean,2), round(model_std,2), round(model_sharpe,2), round(model_sharpe_std,2),
                                     round(buy_hold_mean,2), round(buy_hold_std,2), round(buy_hold_sharpe,2), round(buy_hold_sharpe_std,2)])
    print(f"Compiled results saved to {filename}")

if __name__ == "__main__":
    main()
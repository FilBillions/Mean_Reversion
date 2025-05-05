import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta

class Mean_Rev_BackTest():
        def __init__(self, ma1, ticker, start = str(date.today() - timedelta(59)), end = str(date.today() - timedelta(1)), interval = "1d"):
                df = yf.download(ticker, start, end, interval = interval, multi_level_index=False)
                self.ticker = ticker
                self.df = df
                self.ma1 = ma1

        def run_algo(self, p1 = 5, p2 = 10, mean = 50, p3 = 90, p4 = 95, print_table=True):
                #adding day count
                day_count = np.arange(1, len(self.df) + 1)
                self.df['Day Count'] = day_count
                #dropping unnecessary columns
                if 'Volume' in self.df.columns:
                        self.df.drop(columns=['Volume'], inplace = True)
                if 'Capital Gains' in self.df.columns:
                        self.df.drop(columns=['Capital Gains'], inplace = True)
                if 'Dividends' in self.df.columns:
                        self.df.drop(columns=['Dividends'], inplace = True)
                if 'Stock Splits' in self.df.columns:
                        self.df.drop(columns=['Stock Splits'], inplace = True)

                # --- INITIALISE THE DATAFRAME ---
                # ---
                self.df['Return %'] = (np.log(self.df['Close']).diff()) * 100
                self.df['Cumulative Return %'] = (np.exp(self.df['Return %'] / 100).cumprod() - 1) * 100
                self.df[f'{self.ma1}-day SMA'] = self.df['Close'].rolling(int(self.ma1)).mean().shift()
                self.df['Ratio'] = self.df['Close'] / self.df[f'{self.ma1}-day SMA']
                self.df['Ratio'] = self.df['Ratio'].fillna(0)
                # ---

                # --- intialising percentiles and areas where we'll long and short ---
                percentiles = [p1, p2, mean, p3, p4]

                # percentile calc
                filtered_ratio = self.df['Ratio'][(self.df['Ratio'] > 0) & (self.df['Ratio'].notna())]
                p = np.percentile(filtered_ratio, percentiles)
                print(f'p1: {round(p[0], 4)} p2: {round(p[1], 4)} mean: {round(p[2], 4)} p3: {round(p[3], 4)} p4: {round(p[4], 4)}')

                short = p[4]
                short2 = p[3]
                long = p[0]
                long2 = p[1]
                exit = p[2]

                # --- CODE FOR ADDITIVE POSITION TRACKING --
                # ---
                # ---
                # ---
                # Initialize the 'Position' column with 0 if it doesn't exist
                if 'Position' not in self.df.columns:
                        self.df['Position'] = 0

                # Iterate through the DataFrame row by row to update Signal cumulatively
                for i in range(1, len(self.df)):
                # Carry forward the previous position
                        self.df.loc[self.df.index[i], 'Position'] = self.df.loc[self.df.index[i - 1], 'Position']

                        # Increment position when crossing the long lines
                        if (self.df.loc[self.df.index[i - 1], 'Ratio'] >= long and self.df.loc[self.df.index[i], 'Ratio'] < long) or \
                        (self.df.loc[self.df.index[i - 1], 'Ratio'] >= long2 and self.df.loc[self.df.index[i], 'Ratio'] < long2):
                                self.df.loc[self.df.index[i], 'Position'] += 1

                        # Decrement position when crossing the short lines
                        if (self.df.loc[self.df.index[i - 1], 'Ratio'] <= short and self.df.loc[self.df.index[i], 'Ratio'] > short) or \
                        (self.df.loc[self.df.index[i - 1], 'Ratio'] <= short2 and self.df.loc[self.df.index[i], 'Ratio'] > short2):
                                self.df.loc[self.df.index[i], 'Position'] -= 1

                        # Exit logic: Close position if the ratio hits the exit point
                        if (self.df.loc[self.df.index[i - 1], 'Ratio'] < exit and self.df.loc[self.df.index[i], 'Ratio'] >= exit) and \
                        (self.df.loc[self.df.index[i], 'Position'] > 0):
                                self.df.loc[self.df.index[i], 'Position'] = 0

                        if (self.df.loc[self.df.index[i - 1], 'Ratio'] > exit and self.df.loc[self.df.index[i], 'Ratio'] <= exit) and \
                        (self.df.loc[self.df.index[i], 'Position'] < 0):
                                self.df.loc[self.df.index[i], 'Position'] = 0
                # End of additive position tracking
                # ---
                # ---
                # ---

                # Decrement and increment position when crossing the short line
                self.df['Signal'] = np.where(self.df['Ratio'] > short2, -1, np.nan)
                self.df['Signal'] = np.where(self.df['Ratio'] < long2, 1, self.df['Signal'])
                self.df['Entry'] = self.df.Position.diff()
                self.df['Signal'] = self.df['Signal'].ffill()
                # drop must be placed here to avoid NaN values in the Entry column
                self.df.dropna(inplace=True)

                # Buy and Sell Columns with the closing price on the day
                self.df['Buy'] = np.where(self.df['Entry'] >= 1, self.df['Close'], np.nan)
                self.df['Sell'] = np.where(self.df['Entry'] <= -1, self.df['Close'], np.nan)

                # Model Returns
                self.df['Mean Reversion Model Return'] = self.df['Return %'] * self.df['Signal']
                self.df['Mean Reversion Model Return'] = np.where(self.df['Position'].shift() == 0, 0, self.df['Mean Reversion Model Return'])
                self.df['Cumulative Mean Reversion Model Return'] = (np.exp(self.df['Mean Reversion Model Return'] / 100).cumprod() - 1) * 100

                # Buy and Hold Returns
                self.df['Return %'] = (np.log(self.df['Close']).diff()) * 100
                self.df['Cumulative Return %'] = (np.exp(self.df['Return %'] / 100).cumprod() - 1) * 100

                # Formatting the table
                self.df = round(self.df, 4)

                #format date as YYYY-MM-DD
                self.df.index = pd.to_datetime(self.df.index).strftime('%Y-%m-%d-%H:%M')
                if print_table:
                        #options to show all rows and columns
                        pd.set_option('display.max_rows', None)
                        pd.set_option('display.max_columns', None)
                        pd.set_option('display.width', None)
                        pd.set_option('display.max_colwidth', None)
                        self.df['Ratio'].plot(legend = True)
                        plt.axhline(p[0], c= (.5,.5,.5), ls ='--')
                        plt.axhline(p[1], c= (.5,.5,.5), ls ='--')
                        plt.axhline(p[2], c= (.5,.5,.5), ls ='--')
                        plt.axhline(p[3], c= (.5,.5,.5), ls ='--')
                        plt.axhline(p[4], c= (.5,.5,.5), ls ='--')
                        return self.df
                pass

        def gen_signal(self):
                model_days = self.df['Day Count'].iloc[-1] - self.df['Day Count'].iloc[0] + 1
        # Closing price figure
                self.df.index = pd.to_datetime(self.df.index).strftime('%Y-%m-%d-%H:%M')
                fig1 = plt.figure(figsize=(12, 6))

        # Use the actual index for x-values
                x_values = range(len(self.df.iloc[-model_days:]))

        # Plot the closing prices
                plt.plot(x_values, self.df.iloc[-model_days:]['Close'], label='Close')

        #Plot 12 and 26 day EMAs
                plt.plot(x_values, self.df.iloc[-model_days:][f'{self.ma1}-day SMA'], label=f'{self.ma1}-day SMA') 

        # Plot buy signals (Entry == 2)

                plt.scatter(
                [x_values[i] for i in range(len(self.df.iloc[-model_days:])) if self.df.iloc[-model_days:].iloc[i]['Buy'] > 0],
                self.df.iloc[-model_days:]['Close'][self.df.iloc[-model_days:]['Buy'] > 0],
                marker='^', color='g', s=100, label='Buy Signal'
                )

        # Plot sell signals (Entry == -2)
                plt.scatter(
                [x_values[i] for i in range(len(self.df.iloc[-model_days:])) if self.df.iloc[-model_days:].iloc[i]['Sell'] > 0],
                self.df.iloc[-model_days:]['Close'][self.df.iloc[-model_days:]['Sell'] > 0],
                marker='v', color='r', s=100, label='Sell Signal'
                )

        # Set x-axis to date values and make it so they dont spawn too many labels
                plt.xticks(ticks=x_values, labels=self.df.iloc[-model_days:].index, rotation=45)
                plt.locator_params(axis='x', nbins=10)

        # grid and legend
                plt.grid(True, alpha=0.5)
                plt.legend(loc=2)

        #print statements        
                print(f'from {self.df.index[-model_days]} to {self.df.index[-1]}')
                print(f'count of buy signals: {len(self.df[self.df["Buy"] > 0])}')
                print(f'count of sell signals: {len(self.df[self.df["Sell"] > 0])}')

        def gen_comp(self):
                labels = pd.to_datetime(self.df.index).strftime('%Y-%m-%d')
                fig1= plt.figure(figsize=(12, 6))
                x_values = range(len(self.df))

        # add buy/hold to legend if it doesn't exist
                if f'{self.ticker} Buy/Hold' not in [line.get_label() for line in plt.gca().get_lines()]:
                        plt.plot(x_values, self.df['Cumulative Return %'], label=f'{self.ticker} Buy/Hold')
                # model plot
                        plt.plot(x_values, self.df['Cumulative Mean Reversion Model Return'], label=f'{self.ticker} Mean Reversion Model')

                # Set x-axis to date values and make it so they dont spawn too many labels
                        plt.xticks(ticks=x_values, labels=labels, rotation=45)
                        plt.locator_params(axis='x', nbins=10)

                # grid and legend
                        plt.legend(loc=2)
                        plt.grid(True, alpha=.5)
                # print cumulative return if not already printed
                        print(f"{self.ticker} Cumulative Mean Reversion Model Return:", round(self.df['Cumulative Mean Reversion Model Return'].iloc[-1], 2))
                        print(f"{self.ticker} Cumulative Return:", round(self.df['Cumulative Return %'].iloc[-1], 2))
                        print(f" from {self.df.index[0]} to {self.df.index[-1]}")

        def sharpe(self):
                buyhold_avg_r = float((np.mean(self.df['Return %'])))
                buyhold_std = float((np.std(self.df['Return %'])))
                buyhold_sharpe = (buyhold_avg_r / buyhold_std) * 252 ** 0.5
                model_avg_r = float((np.mean(self.df['Mean Reversion Model Return'])))
                model_std = float((np.std(self.df['Mean Reversion Model Return'])))
                model_sharpe = (model_avg_r / model_std) * 252 ** 0.5
                print(f" Buy/Hold Sharpe {round(buyhold_sharpe, 3)}")
                print(f" Model Sharpe {round(model_sharpe, 3)}")
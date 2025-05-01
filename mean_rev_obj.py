import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fin_table_obj import Table

class Mean_Rev_Table(Table):
    def __init__(self, df, ma1):
        super().__init__(df)
        self.ma1 = ma1

    def gen_table(self, optional_bool=True):
        super().gen_table()
    
# initalising SMA1 and ratio
        self.df[f'{self.ma1}-day SMA'] = self.df['Close'].rolling(int(self.ma1)).mean().shift()
        self.df['Ratio'] = self.df['Close'] / self.df[f'{self.ma1}-day SMA']

# intialising percentiles and areas where we'll long and short
        percentiles = [5, 10, 50, 90, 95]
        p = np.percentile(self.df['Ratio'].dropna(), percentiles)
        print("distance between percentile and sma: ", p[0], p[1], p[2], p[3], p[4])
        
        self.df['Ratio'].dropna().plot(legend = True)
        plt.axhline(p[0], c= (.5,.5,.5), ls ='--')
        plt.axhline(p[1], c= (.5,.5,.5), ls ='--')
        plt.axhline(p[2], c= (.5,.5,.5), ls ='--')
        plt.axhline(p[3], c= (.5,.5,.5), ls ='--')
        plt.axhline(p[4], c= (.5,.5,.5), ls ='--')

# long and short logic ---- we want to close position if we are in the middle of the range
        short = p[4]
        short2 = p[3]
        long = p[0]
        long2 = p[1]
        exit = p[2]

        # Long and short logic ---- we want to adjust the position when crossing the long or short lines
        short = p[4]
        long = p[0]
        exit = p[2]

        # Increment position when crossing the long line
        self.df['Position'] = np.where(self.df['Ratio'] < long, 1, np.nan)

        # Decrement position when crossing the short line
        self.df['Position'] = np.where(self.df['Ratio'] > short, -1, self.df['Position'])

        # Forward-fill the position to maintain the current state
        self.df['Position'] = self.df['Position'].ffill()


# Model return
        self.df['Mean Reversion Model Return'] = self.df['Return'] * self.df['Position']
# Entry column for visualization
        self.df['Entry'] = self.df.Position.diff()
# drop rows
        self.df.dropna(inplace=True)
# Cumulative Returns
        self.df['Cumulative Mean Reversion Model Return'] = (np.exp(self.df['Mean Reversion Model Return'] / 100).cumprod() - 1) * 100
# Recalculate return and cumulative return to include model returns
        self.df['Return'] = (np.log(self.df['Close']).diff()) * 100
        self.df['Cumulative Return'] = (np.exp(self.df['Return'] / 100).cumprod() - 1) * 100
# Formatting the table
        self.df = round((self.df[['Day Count', 'Open', 'High', 'Low', 'Close', f'{self.ma1}-day SMA', 'Return', 'Cumulative Return', 'Mean Reversion Model Return', 'Cumulative Mean Reversion Model Return', 'Ratio', 'Position', 'Entry']]), 3)
#format date as YYYY-MM-DD
        self.df.index = pd.to_datetime(self.df.index).strftime('%Y-%m-%d-%H:%M')
        if optional_bool:
            #options to show all rows and columns
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            return self.df
        pass

    def gen_mean_rev_visual(self):
        pass

    def gen_buyhold_comp(self, ticker):
        labels = pd.to_datetime(self.df.index).strftime('%Y-%m-%d')
        fig1= plt.figure(figsize=(12, 6))
        x_values = range(len(self.df))

# add buy/hold to legend if it doesn't exist
        if f'{ticker} Buy/Hold' not in [line.get_label() for line in plt.gca().get_lines()]:
            plt.plot(x_values, self.df['Cumulative Return'], label=f'{ticker} Buy/Hold')
# model plot
        plt.plot(x_values, self.df['Cumulative Mean Reversion Model Return'], label=f'{ticker} Mean Reversion Model')

# Set x-axis to date values and make it so they dont spawn too many labels
        plt.xticks(ticks=x_values, labels=labels, rotation=45)
        plt.locator_params(axis='x', nbins=10)

# grid and legend
        plt.legend(loc=2)
        plt.grid(True, alpha=.5)
# print cumulative return if not already printed
        print(f"{ticker} Cumulative Mean Reversion Model Return:", round(self.df['Cumulative Mean Reversion Model Return'].iloc[-1], 2))
        print(f" from {self.df.index[0]} to {self.df.index[-1]}")
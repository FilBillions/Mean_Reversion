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
    
# --- initalising SMA1 and ratio ---
        self.df[f'{self.ma1}-day SMA'] = self.df['Close'].rolling(int(self.ma1)).mean().shift()
        self.df['Ratio'] = self.df['Close'] / self.df[f'{self.ma1}-day SMA']

# --- intialising percentiles and areas where we'll long and short ---
        percentiles = [5, 10, 50, 90, 95]
        p = np.percentile(self.df['Ratio'].dropna(), percentiles)
        print("distance between percentile and sma: ", p[0], p[1], p[2], p[3], p[4])
        
        self.df['Ratio'].dropna().plot(legend = True)
        plt.axhline(p[0], c= (.5,.5,.5), ls ='--')
        plt.axhline(p[1], c= (.5,.5,.5), ls ='--')
        plt.axhline(p[2], c= (.5,.5,.5), ls ='--')
        plt.axhline(p[3], c= (.5,.5,.5), ls ='--')
        plt.axhline(p[4], c= (.5,.5,.5), ls ='--')

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

        # Decrement and increment position when crossing the short line
        self.df['Signal'] = np.where(self.df['Ratio'] > short, -1, np.nan)
        self.df['Signal'] = np.where(self.df['Ratio'] < long, 1, self.df['Signal'])
# End of additive position tracking
# ---
# ---
# ---

# Initialize Entry and Signal columns
        self.df['Entry'] = self.df.Position.diff()
        self.df['Signal'] = self.df['Signal'].ffill()

# drop must be placed here to avoid NaN values in the Entry column
        self.df.dropna(inplace=True)

# Buy and Sell Columns with the closing price on the day
        self.df['Buy'] = np.where(self.df['Entry'] >= 1, self.df['Close'], np.nan)
        self.df['Sell'] = np.where(self.df['Entry'] <= -1, self.df['Close'], np.nan)

# Model Returns
        self.df['Mean Reversion Model Return'] = self.df['Return'] * self.df['Signal']
        self.df['Cumulative Mean Reversion Model Return'] = (np.exp(self.df['Mean Reversion Model Return'] / 100).cumprod() - 1) * 100
# Recalculate return and cumulative return to include model returns

        self.df['Return'] = (np.log(self.df['Close']).diff()) * 100
        self.df['Cumulative Return'] = (np.exp(self.df['Return'] / 100).cumprod() - 1) * 100

# Formatting the table
        self.df = round((self.df[['Day Count', 'Open', 'High', 'Low', 'Close', f'{self.ma1}-day SMA', 'Return', 'Cumulative Return', 'Mean Reversion Model Return', 'Cumulative Mean Reversion Model Return', 'Ratio', 'Position', 'Signal', 'Buy', 'Sell', 'Entry']]), 3)

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

    def gen_visual(self, model_days):
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

    def gen_comp(self, ticker):
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
import sys
sys.path.append(".")
import csv
import random
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

# function should always be named conditional_probability
from contents.mean_rev_strategy import Mean_Rev_BackTest
from contents.simple_return import Simple_Return
# Goal of this function
# 1.) Take in a conditonal probability object
# 2.) Set fixed paramters such as target probability, ticker, interval
# 3.) Test the model an N number of times based on the fixed parameters
# 4.) Export the results to a CSV file, to be used in futher analysis

class Backtest():
    def __init__(self):
        with open('inputs.csv', 'r') as file:
            reader = csv.DictReader(file)  # <-- remove delimiter='\t'
            rows = list(reader)
            self.ticker_list = [row['Ticker'] for row in rows if row['Ticker']]
            self.moving_average_list = [row['Moving Average'] for row in rows if row['Moving Average']]
            self.interval_list = [row['Interval'] for row in rows if row['Interval']]
        
        #Argument Checks
        if len(sys.argv) < 3:
            raise ValueError("Not enough arguments provided. Please follow the format: python backtest.py <number_of_iterations> <cash or equity>")
        if len(sys.argv) == 3:
            # Check if first argument is an integer, it should be
            number_check = None
            try:
                number_check = int(sys.argv[1])
            except ValueError:
                raise ValueError("Invalid argument. Please provide number of iterations as an integer.")
            if isinstance(number_check, int):
                self.arg1 = int(sys.argv[1])
            else:
                raise ValueError("Invalid argument. Please provide number of iterations as an integer.")
            if sys.argv[2] == 'cash':
                self.cash = True
                self.equity = False
            elif sys.argv[2] == 'equity':
                self.cash = False
                self.equity = True
            else:
                raise ValueError("Invalid argument. Please provide either 'cash' or 'equity' as the second argument.")
        if len(sys.argv) > 3:
            raise ValueError("Too many arguments. Please provide only the number of iterations as an integer.")

# Inputs
        self.extra_short_days_list=['2m', '5m']
        self.short_days_list =['15m', '30m']
        self.medium_days_list = ['60m', '90m', '1h']
        self.long_days_list = ['1d', '5d', '1wk', '1mo', '3mo']
        self.today = datetime.now()

    def backtest(self):
        for ticker in self.ticker_list:
            for interval in self.interval_list:
                print("-" * 50)
                print(f"Downloading data for {ticker} with interval {interval}...")
                if interval in self.long_days_list:
                    self.universe = datetime.strptime("2000-01-01", "%Y-%m-%d")
                    self.tie_in = 365
                    self.end_date_range = self.today - timedelta(days=self.tie_in)  
                    self.step_input = 5
                if interval in self.medium_days_list:
                    self.universe = datetime.combine(self.today, datetime.min.time()) - timedelta(days=729, hours=5, minutes=30)
                    self.tie_in = 180
                    self.end_date_range = self.today - timedelta(days=self.tie_in)
                    self.step_input = 5
                if interval in self.short_days_list:
                    self.universe = datetime.combine(self.today, datetime.min.time()) - timedelta(days=59, hours=5, minutes=30)
                    self.tie_in = 30
                    self.end_date_range = self.today - timedelta(days=self.tie_in)
                    self.step_input = 5
                if interval in self.extra_short_days_list:
                    self.universe = datetime.combine(self.today, datetime.min.time()) - timedelta(days=59, hours=5, minutes=30)
                    self.tie_in = 7
                    self.end_date_range = self.today - timedelta(days=self.tie_in)
                    self.step_input = 5
                print("-" * 50)
                print(f'Simulation Start Date Range : {self.universe}')
                print(f'Simulation End Date Range: {self.end_date_range}')
                try:
                    print("-" * 50)
                    print(f"Downloading {ticker} data...")
                    self.df = yf.download(ticker, start=self.universe, end=str(date.today() - timedelta(1)), interval=interval, multi_level_index=False, ignore_tz=True)
                    self.index1 = self.df.index[0]
                    #Check if input ticker is a valid ticker
                    if self.df.empty:
                        print("-" * 50)
                        print("-" * 50)
                        print(f'Job halted: {ticker} is an invalid Ticker')
                        print("-" * 50)
                        print("-" * 50)
                        sys.exit(1)
                    if ticker != 'SPY':
                        print(f"Downloading SPY...")
                        self.spydf = yf.download('SPY', start = self.universe, end = str(date.today() - timedelta(1)), interval = interval, multi_level_index=False, ignore_tz=True)
                except Exception as e:
                    print(f"Failed to download {ticker} ({interval}): {e} - - - Network Issue")
                for moving_average in self.moving_average_list:
                    print("-" * 50)
                    print(f"Running backtest for {ticker} with interval {interval} and moving average {moving_average}...")
                    for i in range(self.arg1):
                        print("-" * 50)
                        print(f"Backtest {i + 1} of {self.arg1}...")
                        # Generate a random start date within the range
                        # To stop errors from generating due to bad dates, we need to subtract end_date_range by the duration of our algorithm
                        if interval in self.long_days_list:
                            random_input = random.randint(0, (self.end_date_range - self.index1).days)
                            input_start_date = pd.to_datetime(self.index1 + timedelta(days=random_input))
                        if interval in self.medium_days_list or interval in self.short_days_list or interval in self.extra_short_days_list:
                            minutes = (self.end_date_range - self.index1).total_seconds() // 60
                            random_input = random.randint(0, int(minutes))
                            input_start_date = pd.to_datetime(self.index1 + timedelta(minutes=random_input))
                        input_end_date = pd.to_datetime(input_start_date + timedelta(days=self.tie_in))
                        # Check if input_end_date is valid
                        if input_end_date < self.today:
                            # Testing Modules for a specific date
                            #input_start_date = "2000-07-29"
                            #input_end_date = "2001-07-29"
                            try:
                                model = Mean_Rev_BackTest(ticker=ticker, ma1=moving_average, interval=interval, start=self.universe, optional_df=self.df)
                                model.run_algo(start_date=input_start_date, end_date=input_end_date, return_table=False)
                            except Exception as e:
                                print(f"Iteration {i+1}: {e}")
                                print("Skipping this iteration due to insufficient data.")
                                continue  # Skip to the next iteration
                            real_start_date = model.df.index[0]  # Get the first date in the DataFrame
                            real_end_date = model.df.index[-1]  # Get the last date in the DataFrame
                            if ticker != 'SPY':
                                spy_model = Simple_Return(ticker=ticker, interval=interval, start=input_start_date, end=real_end_date, optional_df=self.spydf)
                            if self.equity == True and self.cash == False:
                                backtest_result = model.backtest_percent_of_equity(return_table=False, print_statement=False, model_return=True)
                                buy_hold_result = model.backtest_percent_of_equity(return_table=False, buy_hold=True)
                            elif self.cash == True and self.equity == False:
                                backtest_result = model.backtest_cash(return_table=False, print_statement=False, model_return=True)
                                buy_hold_result = model.backtest_cash(return_table=False, buy_hold=True)

                            #Sharpe Ratios
                            backtest_sharpe = model.sharpe_ratio(return_model=True)
                            buy_hold_sharpe = model.sharpe_ratio(return_buy_hold=True)
                            if ticker != 'SPY':
                                spy_result = spy_model.get_return()
                                spy_sharpe = spy_model.get_sharpe()
                                spy_delta = backtest_result - spy_result
                                print(f"SPY Buy/Hold Result: {spy_result}")
                            delta = backtest_result - buy_hold_result

                            if np.isnan(backtest_sharpe):
                                print(f"Error: Errors found in backtest due to overload. Backtest #{i + 1} scrapped.")
                                return
                            else:
                                if self.cash == True and self.equity == False:
                                    filename = f"backtest_output/{ticker}_{interval}_{moving_average}-MA_cash_backtest_results.csv"
                                elif self.equity == True and self.cash == False:
                                    filename = f"backtest_output/{ticker}_{interval}_{moving_average}-MA_percent_of_equity_backtest_results.csv"
                                with open(filename, 'a', newline='') as csvfile:
                                    writer = csv.writer(csvfile)
                                    if csvfile.tell() == 0:  # Check if file is empty
                                        # Write header only if the file is empty
                                        if ticker != 'SPY':
                                            writer.writerow(['Input Start Date', 'Input End Date', 'Start Date', 'End Date', 'Model Result', 'Buy/Hold Result', 'Delta', 'Model Sharpe', 'Buy/Hold Sharpe', 'SPY Buy/Hold Result', 'SPY Sharpe', 'SPY Delta'])
                                        else:
                                            writer.writerow(['Input Start Date', 'Input End Date', 'Start Date', 'End Date', 'Model Result', 'Buy/Hold Result', 'Delta', 'Model Sharpe', 'Buy/Hold Sharpe'])    # header
                                    if ticker != 'SPY':
                                        writer.writerow([input_start_date, input_end_date, real_start_date, real_end_date, backtest_result, buy_hold_result, round(delta,2), backtest_sharpe, buy_hold_sharpe, spy_result, spy_sharpe, round(spy_delta,2)]) # data
                                    else:
                                        writer.writerow([input_start_date, input_end_date, real_start_date, real_end_date, backtest_result, buy_hold_result, round(delta,2), backtest_sharpe, buy_hold_sharpe]) # data
                        elif input_end_date >= self.today:
                            print(f"End Date is not valid, no entry recorded")
                            print(f"{input_end_date}")
        print("-" * 50)
        print("-" * 50)
        print("Backtest completed")
        print("-" * 50)
        print("-" * 50)

if __name__ == "__main__":
    test = Backtest()
    test.backtest()
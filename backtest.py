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
        #Argument Checks
        if len(sys.argv) > 1:
            # Check if first argument is an integer, it should not be
            number_check = None
            try:
                number_check = int(sys.argv[1])
            except ValueError:
                pass
            if isinstance(number_check, int):
                print("-" * 50)
                print(f"Argument: {sys.argv[1]} -> provided integer, not ticker")
                print("Usage: python backtest.py <ticker symbol> <number_of_iterations> <interval>")
                print("-" * 50)
                sys.exit(1)
        if len(sys.argv) >= 5:
            # checking amount of inputs
            print("-" * 50)
            print("Invalid argument. Too many inputs")
            print("Usage: python backtest.py <ticker symbol> <number_of_iterations> <interval>")
            print("-" * 50)
            sys.exit(1)
        if len(sys.argv) == 2:
            self.arg = str(sys.argv[1])
            self.arg2 = 1
            self.arg3 = '1d'
        elif len(sys.argv) == 3:
            try:
                self.arg = str(sys.argv[1])
                self.arg2 = int(sys.argv[2])
                self.arg3 = '1d'
            except ValueError:
                print("-" * 50)
                print("Invalid argument. Please provide a valid ticker and integer for the number of iterations. This should be a positive integer")
                print("Usage: python backtest.py <ticker symbol> <number_of_iterations> <interval>")
                print("-" * 50)
                sys.exit(1)
        elif len(sys.argv) >3:
        # Check if third argument is an integer, it should not be
            number_check = None
            try:
                number_check = int(sys.argv[3])
            except ValueError:
                pass
            if isinstance(number_check, int):
                print("-" * 50)
                print(f"Argument: {sys.argv[3]} -> provided integer, not interval")
                print("Usage: python backtest.py <ticker symbol> <number_of_iterations> <interval>")
                print("-" * 50)
                sys.exit(1)
            try:
                self.arg = str(sys.argv[1])
                self.arg2 = int(sys.argv[2])
                self.arg3 = str(sys.argv[3])
            except ValueError:
                print("-" * 50)
                print("Invalid argument. Please provide a valid ticker, integer, and interval for the number of iterations. This should be a positive integer")
                print("Usage: python backtest.py <ticker symbol> <number_of_iterations> <interval>")
                print("-" * 50)
        else:
            print("-" * 50)
            print("No argument provided. Please provide a valid ticker and integer for the number of iterations. This should be a positive integer")
            print("Usage: python backtest.py <ticker symbol> <number_of_iterations> <interval>")
            print("-" * 50)
            sys.exit(1)

# Inputs
        self.extra_short_days_list=['2m', '5m']
        self.short_days_list =['15m', '30m']
        self.medium_days_list = ['60m', '90m', '1h']
        self.long_days_list = ['1d', '5d', '1wk', '1mo', '3mo']
        self.interval = self.arg3
        self.today = datetime.now()
        if self.interval in self.long_days_list:
            self.universe = datetime.strptime("2000-01-01", "%Y-%m-%d")
            self.tie_in = 365
            self.end_date_range = self.today - timedelta(days=self.tie_in)  
            self.step_input = 5
        if self.interval in self.medium_days_list:
            self.universe = datetime.combine(self.today, datetime.min.time()) - timedelta(days=729, hours=5, minutes=30)
            self.tie_in = 180
            self.end_date_range = self.today - timedelta(days=self.tie_in)
            self.step_input = 5
        if self.interval in self.short_days_list:
            self.universe = datetime.combine(self.today, datetime.min.time()) - timedelta(days=59, hours=5, minutes=30)
            self.tie_in = 30
            self.end_date_range = self.today - timedelta(days=self.tie_in)
            self.step_input = 5
        if self.interval in self.extra_short_days_list:
            self.universe = datetime.combine(self.today, datetime.min.time()) - timedelta(days=59, hours=5, minutes=30)
            self.tie_in = 7
            self.end_date_range = self.today - timedelta(days=self.tie_in)
            self.step_input = 5
        self.ticker = self.arg
        self.target_probability = 0.55
        print(f'Simulation Start Date Range : {self.universe}')
        print(f'Simulation End Date Range: {self.end_date_range}')
# Declare df outside of the loop to avoid re-downloading data each iteration
        print(f"Downloading {self.ticker}...")
        self.df = yf.download(self.ticker, start = self.universe, end = str(date.today() - timedelta(1)), interval = self.interval, multi_level_index=False, ignore_tz=True)
        self.index1 = self.df.index[0]
        #Check if input ticker is a valid ticker
        if self.df.empty:
            print("-" * 50)
            print("-" * 50)
            print(f'Job halted: {self.ticker} is an invalid Ticker')
            print("-" * 50)
            print("-" * 50)
            sys.exit(1)
        if self.ticker != 'SPY':
            print(f"Downloading SPY...")
            self.spydf = yf.download('SPY', start = self.universe, end = str(date.today() - timedelta(1)), interval = self.interval, multi_level_index=False, ignore_tz=True)

    def backtest(self):
        for i in range(self.arg2):
            print(f"Backtest {i + 1} of {self.arg2}...")
            # Generate a random start date within the range
            # To stop errors from generating due to bad dates, we need to subtract end_date_range by the duration of our algorithm
            if self.interval in self.long_days_list:
                random_input = random.randint(0, (self.end_date_range - self.index1).days)
                input_start_date = pd.to_datetime(self.index1 + timedelta(days=random_input))
            if self.interval in self.medium_days_list or self.interval in self.short_days_list or self.interval in self.extra_short_days_list:
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
                    model = Mean_Rev_BackTest(ticker=self.ticker, interval=self.interval, start=self.universe, optional_df=self.df)
                    model.run_algo(start_date=input_start_date, end_date=input_end_date, return_table=False)
                except Exception as e:
                    print(f"Iteration {i+1}: {e}")
                    print("Skipping this iteration due to insufficient data.")
                    continue  # Skip to the next iteration
                real_start_date = model.df.index[0]  # Get the first date in the DataFrame
                real_end_date = model.df.index[-1]  # Get the last date in the DataFrame
                if self.ticker != 'SPY':
                    spy_model = Simple_Return(ticker=self.ticker, interval=self.interval, start=input_start_date, end=real_end_date, optional_df=self.spydf)
                backtest_result = model.backtest_percent_of_equity(return_table=False, print_statement=False, model_return=True)
                buy_hold_result = model.backtest_percent_of_equity(return_table=False, buy_hold=True)

                #Sharpe Ratios
                backtest_sharpe = model.sharpe_ratio(return_model=True)
                buy_hold_sharpe = model.sharpe_ratio(return_buy_hold=True)
                if self.ticker != 'SPY':
                    spy_result = spy_model.get_return()
                    spy_sharpe = spy_model.get_sharpe()
                    spy_delta = backtest_result - spy_result
                    print(f"SPY Buy/Hold Result: {spy_result}")
                delta = backtest_result - buy_hold_result

                # Export to CSV
                def export_to_csv(backtest_result, buy_hold_result, filename=f"{self.ticker}_{self.interval}_9-MA_backtest_results.csv"):
                    #Check for Overload Error
                    if np.isnan(backtest_sharpe):
                        print(f"Error: Errors found in backtest due to overload. Backtest #{i + 1} scrapped.")
                        return
                    else:
                        with open(filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            if csvfile.tell() == 0:  # Check if file is empty
                                # Write header only if the file is empty
                                if self.ticker != 'SPY':
                                    writer.writerow(['Input Start Date', 'Input End Date', 'Start Date', 'End Date', 'Model Result', 'Buy/Hold Result', 'Delta', 'Model Sharpe', 'Buy/Hold Sharpe', 'SPY Buy/Hold Result', 'SPY Sharpe', 'SPY Delta'])
                                else:
                                    writer.writerow(['Input Start Date', 'Input End Date', 'Start Date', 'End Date', 'Model Result', 'Buy/Hold Result', 'Delta', 'Model Sharpe', 'Buy/Hold Sharpe'])    # header
                            if self.ticker != 'SPY':
                                writer.writerow([input_start_date, input_end_date, real_start_date, real_end_date, backtest_result, buy_hold_result, round(delta,2), backtest_sharpe, buy_hold_sharpe, spy_result, spy_sharpe, round(spy_delta,2)]) # data
                            else:
                                writer.writerow([input_start_date, input_end_date, real_start_date, real_end_date, backtest_result, buy_hold_result, round(delta,2), backtest_sharpe, buy_hold_sharpe]) # data
                print("Done")
                export_to_csv(backtest_result, buy_hold_result)
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
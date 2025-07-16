import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
sb.set_theme()

# make charts generic

def comparison(object):
    labels = pd.to_datetime(object.df.index).strftime('%Y-%m-%d')
    fig1= plt.figure(figsize=(12, 6))
    x_values = range(len(object.df))

    # add buy/hold to legend if it doesn't exist
    if f'{object.ticker} Buy/Hold' not in [line.get_label() for line in plt.gca().get_lines()]:
        plt.plot(x_values, object.df['Buy/Hold Value'], label=f'{object.ticker} Buy/Hold')
    # model plot
    plt.plot(x_values, object.df['Model Value'], label=f'{object.ticker} Model')

    # Set x-axis to date values and make it so they dont spawn too many labels
    plt.xticks(ticks=x_values, labels=labels, rotation=45)
    plt.locator_params(axis='x', nbins=10)

    # grid and legend
    plt.legend(loc=2)
    plt.grid(True, alpha=.5)
    
def linear_regression(object):
            # --- Graph setup ---
    x = object.df[["Previous Period Return"]]
    y = object.df["Return"]
    model = LinearRegression().fit(x,y)
    x_range = np.linspace(x.min(),x.max(),100)
    y_pred_line = model.predict(x_range)
    fig1 = plt.figure(figsize=(12, 6))
    sb.scatterplot(x="Previous Period Return", y="Return", data=object.df, color='Blue', label="Returns")
    plt.plot(x_range, y_pred_line, color='red', label="Regression Line")
    plt.xlabel("Previous Period Return (%)")
    plt.ylabel("Current Return (%)")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()

def visual(object):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        shared_yaxes=False,
        row_heights=[2, 3],
        vertical_spacing=0.01,
        subplot_titles=("Candlesticks with Buy/Sell Signals", "Ratio Histogram")
    )

    # make fig size larger
    fig.update_layout(height=800, width=1200, title_text=f"{object.ticker} Mean Reversion Strategy Visualization", template='plotly_dark')

    # Candlestick (row 1)
    fig.add_trace(
        go.Candlestick(
            x=object.df.index,
            open=object.df['Open'],
            high=object.df['High'],
            low=object.df['Low'],
            close=object.df['Close'],
            name='Candlestick'
        ),
        row=1, col=1
    )

    # Buy Signal (row 1)
    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=np.where(object.df['Buy Signal'].notna(), object.df['Close'], np.nan),
            mode='markers',
            marker=dict(symbol='triangle-up', size=20, color='green'),
            name='Buy Signal'
        ),
        row=1, col=1
    )
    # Sell Signal (row 1)
    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=np.where(object.df['Sell Signal'].notna(), object.df['Close'], np.nan),
            mode='markers',
            marker=dict(symbol='triangle-down', size=20, color='red'),
            name='Sell Signal'
        ),
        row=1, col=1
    )

    # Buy Signal (row 2)
    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=np.where(object.df['Buy Signal'].notna(), object.df[f"Ratio"], np.nan),
            mode='markers',
            marker=dict(symbol='triangle-up', size=20, color='green'),
            name='Buy Signal (On Ratio)'
        ),
        row=2, col=1
    )

    # Sell Signal (row 2)
    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=np.where(object.df['Sell Signal'].notna(), object.df[f"Ratio"], np.nan),
            mode='markers',
            marker=dict(symbol='triangle-down', size=20, color='red'),
            name='Sell Signal (On Ratio)'
        ),
        row=2, col=1
    )

    # Add MA (row 1)
    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=object.df[f"{object.ma1}-day SMA"],
            mode='lines',
            name=f"{object.ma1}-day SMA",
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Add Ratio (row 2)
    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=object.df['Ratio'],
            mode='lines',
            name='Ratio',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )

    # Add long and short thresholds (row 2)
    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=np.repeat(object.long, len(object.df)),
            mode='lines',
            name='Long Threshold',
            line=dict(color='green', width=2, dash='dash')
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=np.repeat(object.short, len(object.df)),
            mode='lines',
            name='Short Threshold',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=2, col=1
    )

    # Add long2 and short2 thresholds (row 2)
    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=np.repeat(object.long2, len(object.df)),
            mode='lines',
            name='Long2 Threshold',
            line=dict(color='lightgreen', width=2, dash='dash')
        ),
        row=2, col=1
    )

    # Add short2 threshold (row 2)
    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=np.repeat(object.short2, len(object.df)),
            mode='lines',
            name='Short2 Threshold',
            line=dict(color='salmon', width=2, dash='dash')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=object.df.index,
            y=np.repeat(object.exit, len(object.df)),
            mode='lines',
            name='Exit Threshold',
            line=dict(color='orange', width=2, dash='dash')
        ),
        row=2, col=1
    )

    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark')
    return fig

def normal(object):
    # work on this some more, i would like the objkect captured to be flexible, i.e return, model return, or etc
    # normal distribution plot
    fig1 = plt.figure(figsize=(12, 6))
    return_modified = object.df['Return'] / 100
    mean = return_modified.mean()
    std = return_modified.std()
    plt.hist(return_modified, bins = 50, density = True)
    plt.xlim(return_modified.min(), return_modified.max())
    plt.plot(object.overlay, object.p, 'k')
    plt.axvline(mean, color='r', linestyle='dashed')
    
    # Standard Deviation Plots
    plt.axvline(mean + std, color='g', linestyle='dashed')
    plt.axvline(mean + (2 * std), color='b', linestyle='dashed')
    plt.axvline(mean - (2 * std), color='b', linestyle='dashed')
    plt.axvline(mean - std, color='g', linestyle='dashed')
    
    # labels
    plt.text(mean, plt.ylim()[1] * .9, 'mean', color='r', ha='center')
    plt.text(mean + std, plt.ylim()[1] * .8, '+1std', color='g', ha='center')
    plt.text(mean + (2 * std), plt.ylim()[1] * .7, '+2std', color='b', ha='center')
    plt.text(mean - (2 * std), plt.ylim()[1] * .7, '-2std', color='b', ha='center')
    plt.text(mean - std, plt.ylim()[1] * .8, '-1std', color='g', ha='center')
    plt.title(f"Mean: {round(mean, 6)}, Std: {round(std, 6)}")
    plt.xlabel('Percent Change')
    plt.ylabel('Density')
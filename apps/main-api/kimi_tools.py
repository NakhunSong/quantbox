"""Kimi Tool definitions for Quant AI Agent (OpenAI format)"""

QUANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_market_data",
            "description": "Fetch historical price data (OHLCV) for given stock tickers using Yahoo Finance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock ticker symbols (e.g., ['AAPL', 'GOOGL'])",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                        "description": "Time period for historical data",
                    },
                },
                "required": ["tickers", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_correlation",
            "description": "Calculate correlation matrix between multiple assets' returns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock ticker symbols",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1mo", "3mo", "6mo", "1y", "2y"],
                        "description": "Time period for analysis",
                    },
                },
                "required": ["tickers", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_price_chart",
            "description": "Plot price chart with moving averages for a stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1mo", "3mo", "6mo", "1y", "2y"],
                        "description": "Time period for chart",
                    },
                    "ma_windows": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Moving average windows (e.g., [20, 50])",
                    },
                },
                "required": ["ticker", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": "Run a simple moving average crossover backtest strategy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "short_window": {
                        "type": "integer",
                        "description": "Short moving average window (days)",
                    },
                    "long_window": {
                        "type": "integer",
                        "description": "Long moving average window (days)",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1y", "2y", "5y"],
                        "description": "Backtest period",
                    },
                },
                "required": ["ticker", "short_window", "long_window", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_prediction_model",
            "description": "Build a simple linear regression model to predict stock returns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol to predict",
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Feature tickers to use as predictors",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1y", "2y", "5y"],
                        "description": "Training period",
                    },
                },
                "required": ["ticker", "features", "period"],
            },
        },
    },
]


def generate_code_for_tool(tool_name: str, inputs: dict) -> str:
    """Generate Python code for the given tool and inputs"""

    if tool_name == "fetch_market_data":
        tickers = inputs["tickers"]
        period = inputs["period"]
        return f"""
import yfinance as yf

tickers = {tickers}
period = "{period}"

data = yf.download(tickers, period=period, progress=False)
if len(tickers) == 1:
    result = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_dict()
else:
    result = data['Close'].tail(10).to_dict()

_result_ = {{"tickers": tickers, "period": period, "data": str(result)}}
"""

    elif tool_name == "calculate_correlation":
        tickers = inputs["tickers"]
        period = inputs["period"]
        return f"""
import yfinance as yf
import seaborn as sns

tickers = {tickers}
period = "{period}"

data = yf.download(tickers, period=period, progress=False)['Close']
returns = data.pct_change().dropna()
corr_matrix = returns.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Asset Correlation Matrix')

_result_ = {{"correlation_matrix": corr_matrix.to_dict()}}
"""

    elif tool_name == "plot_price_chart":
        ticker = inputs["ticker"]
        period = inputs["period"]
        ma_windows = inputs.get("ma_windows", [20, 50])
        return f"""
import yfinance as yf

ticker = "{ticker}"
period = "{period}"
ma_windows = {ma_windows}

data = yf.download(ticker, period=period, progress=False)

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', linewidth=1.5)

for window in ma_windows:
    ma = data['Close'].rolling(window=window).mean()
    plt.plot(data.index, ma, label=f'MA{{window}}', linewidth=1)

plt.title(f'{{ticker}} Price Chart')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

_result_ = {{"ticker": ticker, "period": period, "last_close": float(data['Close'].iloc[-1])}}
"""

    elif tool_name == "run_backtest":
        ticker = inputs["ticker"]
        short_window = inputs["short_window"]
        long_window = inputs["long_window"]
        period = inputs["period"]
        return f"""
import yfinance as yf
import pandas as pd

ticker = "{ticker}"
short_window = {short_window}
long_window = {long_window}

data = yf.download(ticker, period="{period}", progress=False)
data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
data['SMA_long'] = data['Close'].rolling(window=long_window).mean()

data['Signal'] = 0
data.loc[data['SMA_short'] > data['SMA_long'], 'Signal'] = 1
data['Position'] = data['Signal'].diff()

data['Returns'] = data['Close'].pct_change()
data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']

total_return = (1 + data['Strategy_Returns'].dropna()).prod() - 1
buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1

# Plot backtest results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(data.index, data['Close'], label='Price', alpha=0.7)
ax1.plot(data.index, data['SMA_short'], label=f'SMA{{short_window}}')
ax1.plot(data.index, data['SMA_long'], label=f'SMA{{long_window}}')
ax1.scatter(data[data['Position'] == 1].index, data[data['Position'] == 1]['Close'], marker='^', color='g', label='Buy', s=100)
ax1.scatter(data[data['Position'] == -1].index, data[data['Position'] == -1]['Close'], marker='v', color='r', label='Sell', s=100)
ax1.set_title(f'{{ticker}} Backtest: SMA{{short_window}}/{{long_window}} Crossover')
ax1.legend()
ax1.grid(True, alpha=0.3)

cumulative_strategy = (1 + data['Strategy_Returns'].fillna(0)).cumprod()
cumulative_buyhold = (1 + data['Returns'].fillna(0)).cumprod()
ax2.plot(data.index, cumulative_strategy, label='Strategy')
ax2.plot(data.index, cumulative_buyhold, label='Buy & Hold')
ax2.set_title('Cumulative Returns')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

_result_ = {{
    "ticker": ticker,
    "strategy_return": round(float(total_return) * 100, 2),
    "buy_hold_return": round(float(buy_hold_return) * 100, 2),
    "num_trades": int(abs(data['Position']).sum() / 2)
}}
"""

    elif tool_name == "build_prediction_model":
        ticker = inputs["ticker"]
        features = inputs["features"]
        period = inputs["period"]
        return f"""
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

target = "{ticker}"
features = {features}
all_tickers = [target] + features

data = yf.download(all_tickers, period="{period}", progress=False)['Close']
returns = data.pct_change().dropna()

X = returns[features]
y = returns[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

coefficients = dict(zip(features, model.coef_.tolist()))

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.title(f'{{target}} Prediction (R² = {{r2:.4f}})')

plt.subplot(1, 2, 2)
plt.bar(features, model.coef_)
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.title('Feature Importance')
plt.xticks(rotation=45)

plt.tight_layout()

_result_ = {{
    "target": target,
    "features": features,
    "r2_score": round(float(r2), 4),
    "coefficients": coefficients
}}
"""

    else:
        return f"# Unknown tool: {tool_name}\n_result_ = {{'error': 'Unknown tool'}}"

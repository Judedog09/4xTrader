import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# === WINDOWS FIX ===
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# === CONFIGURATION ===
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def get_adaptive_signal(df):
    """Calculates signals based on ADX volatility switch."""
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['STD'] = df['close'].rolling(20).std()
    df['Lower'] = df['MA20'] - (df['STD'] * 2)
    
    # Manual ADX (Trend Strength)
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = df['low'].diff().clip(upper=0).abs()
    tr = pd.concat([df['high'] - df['low'], 
                    (df['high'] - df['close'].shift()).abs(), 
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['ADX'] = ((plus_dm.rolling(14).mean() - minus_dm.rolling(14).mean()).abs() / (atr + 1e-9)) * 100
    df['ADX'] = df['ADX'].rolling(14).mean().fillna(0)

    df['Signal'] = 0
    # TREND MODE (ADX > 25)
    df.loc[(df['ADX'] >= 25) & (df['EMA5'] > df['EMA13']), 'Signal'] = 1
    # CHOP MODE (ADX < 25)
    df.loc[(df['ADX'] < 25) & (df['close'] < df['Lower']), 'Signal'] = 1
    
    # EXITS
    df.loc[(df['ADX'] >= 25) & (df['EMA5'] < df['EMA13']), 'Signal'] = 0
    df.loc[(df['ADX'] < 25) & (df['close'] > df['MA20']), 'Signal'] = 0
    return df

def backtest_portfolio(symbols):
    results = []
    print(f"{'SYMBOL':<8} | {'TRADES':<6} | {'MKT %':<8} | {'STRAT %':<8} | {'PF'}")
    print("-" * 65)

    for ticker in symbols:
        try:
            bars = api.get_bars(ticker, '5Min', limit=2000).df
            if len(bars) < 50: continue
            
            df = get_adaptive_signal(bars)
            df['Market_Return'] = df['close'].pct_change()
            df['Position'] = df['Signal'].shift(1).fillna(0)
            df['Trade_Cost'] = df['Signal'].diff().abs() * 0.0002 # Slippage estimate
            df['P_and_L'] = (df['Position'] * df['Market_Return']) - df['Trade_Cost']
            
            wins = df[df['P_and_L'] > 0]['P_and_L'].sum()
            losses = abs(df[df['P_and_L'] < 0]['P_and_L'].sum())
            pf = round(wins / losses, 2) if losses > 0 else 9.99
            
            mkt_f = (df['Market_Return'] + 1).cumprod().iloc[-1] - 1
            str_f = (df['P_and_L'] + 1).cumprod().iloc[-1] - 1
            trades = (df['Signal'].diff() == 1).sum()
            
            print(f"{ticker:<8} | {trades:<6} | {mkt_f*100:>7.2f}% | {str_f*100:>7.2f}% | {pf}")
            results.append({'str': str_f, 'mkt': mkt_f, 'pf': pf, 'sym': ticker})
        except: continue

    if results:
        avg_str = np.mean([r['str'] for r in results]) * 100
        print("-" * 65)
        print(f"OVERALL PERFORMANCE: Avg Return: {avg_str:.2f}%")

if __name__ == "__main__":
    test_list = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "META", "GOOGL"]
    backtest_portfolio(test_list)
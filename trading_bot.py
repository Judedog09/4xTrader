import alpaca_trade_api as tradeapi
import time, pandas as pd, numpy as np, os, nltk, requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

# --- INITIALIZATION ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')

# === CONFIGURATION ===
api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"), 
    os.getenv("ALPACA_SECRET_KEY"), 
    os.getenv("ALPACA_BASE_URL"),
    api_version='v2'
)
sia = SentimentIntensityAnalyzer()

WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "AMD", "SPY", "QQQ", "PLTR"]

# RISK SETTINGS
MAX_POSITIONS = 5
STOP_LOSS_PCT = 0.98
TAKE_PROFIT_PCT = 1.04

def get_adaptive_signal(ticker):
    """ADX-based strategy switch logic."""
    try:
        bars = api.get_bars(ticker, '5Min', limit=50).df
        if bars.empty or len(bars) < 30: return "HOLD", 0
        
        df = bars.copy()
        df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['EMA13'] = df['close'].ewm(span=13).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['STD'] = df['close'].rolling(20).std()
        df['Lower'] = df['MA20'] - (df['STD'] * 2)
        
        # ADX Trend Strength Calculation
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = df['low'].diff().clip(upper=0).abs()
        tr = pd.concat([df['high'] - df['low'], 
                        (df['high'] - df['close'].shift()).abs(), 
                        (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df['ADX'] = ((plus_dm.rolling(14).mean() - minus_dm.rolling(14).mean()).abs() / (atr + 1e-9)) * 100
        df['ADX'] = df['ADX'].rolling(14).mean()

        current_adx = df['ADX'].iloc[-1]
        
        if current_adx >= 25: # TREND MODE
            if df['EMA5'].iloc[-1] > df['EMA13'].iloc[-1]: return "BUY", current_adx
        else: # CHOP MODE
            if df['close'].iloc[-1] < df['Lower'].iloc[-1]: return "BUY", current_adx
            
        return "HOLD", 0
    except: return "ERROR", 0

def is_autotrader_enabled():
    try:
        response = requests.get("http://localhost:8000/api/config", timeout=2)
        return response.json().get("auto_trader_active", True)
    except: return True 

def run():
    print(">>> Engine Active using ADX Strategy & Bracket Orders.")
    while True:
        try:
            clock = api.get_clock()
            if clock.is_open and is_autotrader_enabled():
                positions = api.list_positions()
                pos_symbols = [p.symbol for p in positions]

                for ticker in WATCHLIST:
                    if ticker in pos_symbols or len(positions) >= MAX_POSITIONS: continue

                    signal, strength = get_adaptive_signal(ticker)
                    
                    if signal == "BUY":
                        price = api.get_latest_trade(ticker).price
                        qty = max(1, int(1000 / price))
                        
                        # BRACKET ORDER: Entry + Automatic SL/TP
                        api.submit_order(
                            symbol=ticker, qty=qty, side='buy', type='limit',
                            limit_price=round(price * 1.001, 2),
                            time_in_force='gtc', order_class='bracket',
                            take_profit={'limit_price': round(price * TAKE_PROFIT_PCT, 2)},
                            stop_loss={'stop_price': round(price * STOP_LOSS_PCT, 2)}
                        )
                        print(f"ENTRY: {ticker} (ADX: {strength:.2f})")
                time.sleep(60)
            else:
                time.sleep(30)
        except Exception as e:
            print(f"Loop Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run()
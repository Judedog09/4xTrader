import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

try:
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
    account = api.get_account()
    
    print("✅ CONNECTION SUCCESSFUL!")
    print(f"Cash Available: ${account.cash}")
    
except Exception as e:
    print(f"❌ CONNECTION FAILED! Check your .env file. Error: {e}")
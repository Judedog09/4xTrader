import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

api = tradeapi.REST(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), os.getenv("ALPACA_BASE_URL"))

def reset_everything():
    print("--- STARTING FULL RESET ---")
    try:
        api.cancel_all_orders()
        api.close_all_positions()
        print("✅ Orders canceled and positions closed.")
        if os.path.exists('trades.csv'):
            with open('trades.csv', 'w') as f:
                f.write("Time,Symbol,Side,Price,Qty,Total\n")
            print("✅ trades.csv reset.")
    except Exception as e:
        print(f"❌ Reset Error: {e}")

if __name__ == "__main__":
    reset_everything()
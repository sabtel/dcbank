import time
import requests

def get_orderbook_bid_price(symbol):
    url = f"https://apiv2.nobitex.ir/v3/orderbook/{symbol}"
    response = requests.get(url, timeout=0.35)
    response.raise_for_status()
    data = response.json()
    if data.get("bids") and len(data["bids"]) > 0:
        best_bid = float(data["bids"][0][0])
        print(f"Bid:   {best_bid:.2f}")   # <-- print inside function
        return best_bid
    else:
        raise ValueError(f"No bid prices available for {symbol}")

def get_orderbook_ask_price(symbol):
    url = f"https://apiv2.nobitex.ir/v3/orderbook/{symbol}"
    response = requests.get(url, timeout=0.35)
    response.raise_for_status()
    data = response.json()
    if data.get("asks") and len(data["asks"]) > 0:
        best_ask = float(data["asks"][0][0])
        print(f"Ask:   {best_ask:.2f}")   # <-- print inside function
        return best_ask
    else:
        raise ValueError(f"No ask prices available for {symbol}")

def monitor_orderbook(symbol="BTCIRT", interval=0.3):
    """
    Monitor bid then ask, print under each other, then show % spread.
    """
    print(f"\n📊 Monitoring {symbol} orderbook every {interval:.1f}s (Ctrl+C to stop)\n")
    next_time = time.monotonic()
    try:
        while True:
            cycle_start = time.monotonic()
            try:
                bid = get_orderbook_bid_price(symbol)
                ask = get_orderbook_ask_price(symbol)

                spread = ask - bid
                pct = (spread * 100.0) / bid if bid != 0 else 0

                elapsed = time.monotonic() - cycle_start
                print(f"Spread: {spread:.2f} ({pct:.4f}%) | Cycle: {elapsed:.4f}s\n")

            except Exception as e:
                elapsed = time.monotonic() - cycle_start
                print(f"Error fetching prices: {e} | Cycle: {elapsed:.4f}s\n")

            # align to exact 0.4s schedule
            next_time += interval
            sleep_for = next_time - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_time = time.monotonic()

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user.")

if __name__ == "__main__":
    monitor_orderbook("BTCIRT", interval=0.3)

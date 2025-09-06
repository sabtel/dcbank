import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import GradientBoostingRegressor
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import Timeout as ReqTimeout
import math

# DAY-CHANGE FETCHER
symbols_day_change = [
    "DOGSIRT", "HMSTRIRT", "SIRT", "TIRT", "1B_BABYDOGEIRT",
    "1M_PEPEIRT", "100K_FLOKIIRT", "NOTIRT", "API3IRT", "ZRXIRT",
    "CVXIRT", "ENSIRT", "SNTIRT", "STORJIRT", "SLPIRT", "CRVIRT",
    "AAVEIRT", "XMRIRT", "LDOIRT", "DYDXIRT", "APTIRT", "MAGICIRT",
    "CHZIRT", "XAUTIRT", "ATOMIRT", "NEARIRT", "FILIRT", "LINKIRT",
    "DOTIRT", "UNIIRT", "XLMIRT", "GMTIRT", "POLIRT", "AVAXIRT",
    "SOLIRT", "ETCIRT", "BCHIRT", "LTCIRT", "TRXIRT", "ADAIRT",
    "DOGEIRT", "XRPIRT", "BNBIRT", "ETHIRT", "BTCIRT", "TONIRT",
    "ARBIRT", "1k_SHIBIRT", "ONEIRT", "HBARIRT","RENDERIRT"
]



def fetch_day_change(src_currency: str, dst_currency: str = "rls") -> float | None:
    url = "https://apiv2.nobitex.ir/market/stats"
    params = {"srcCurrency": src_currency, "dstCurrency": dst_currency}
    try:
        resp = requests.get(url, params=params, timeout=0.8)
        resp.raise_for_status()
        stats = resp.json().get("stats", {}).get(f"{src_currency}-{dst_currency}", {})
        return float(stats["dayChange"]) if stats and "dayChange" in stats else None
    except ReqTimeout:
        raise
    except Exception as e:
        print(f"Error fetching {src_currency}: {e}")
        return None

def fetch_and_display_changes():
    print("\n⏳  Daily %-changes (via Nobitex)  ")
    skipped = []
    all_changes = []

    for symbol in symbols_day_change:
        src = symbol[:-3].lower()
        try:
            change = fetch_day_change(src)
        except ReqTimeout:
            print(f"{symbol:<12}    ⏱️ Timeout, skipping for now")
            skipped.append(symbol)
            continue

        if change is not None:
            all_changes.append(change)
            print(f"{symbol:<12} {change:+6.2f}")
        else:
            print(f"{symbol:<12}    N/A")

    if skipped:
        print("\n🔄 Retrying timed-out symbols...")
        for symbol in skipped:
            src = symbol[:-3].lower()
            try:
                change = fetch_day_change(src)
            except ReqTimeout:
                print(f"{symbol:<12}    ❌ Still timed out")
                continue
            if change is not None:
                all_changes.append(change)
                print(f"{symbol:<12} {change:+6.2f}  (retry)")
            else:
                print(f"{symbol:<12}    N/A  (retry)")

    if not all_changes:
        return None

    q1 = np.percentile(all_changes, 25)
    q3 = np.percentile(all_changes, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_changes = [x for x in all_changes if lower_bound <= x <= upper_bound]

    if not filtered_changes:
        print("\n⚠️ All values filtered as outliers. No average calculated.")
        return None

    avg = sum(filtered_changes) / len(filtered_changes)
    print(f"\n📊 Average daily change (excluding outliers): {avg:+.2f}%")
    return avg

    

# TIMEZONE SETUP
TEHRAN_TZ = pytz.timezone("Asia/Tehran")

# CONFIGURATION
TRADING_TOKEN = "69dacf9c757f30678daf243c481772185cb388d1"
WALLET_BALANCE_URL = "https://apiv2.nobitex.ir/users/wallets/balance"
ORDER_URL = "https://apiv2.nobitex.ir/market/orders/add"
MARGIN_ORDER_URL = "https://apiv2.nobitex.ir/margin/orders/add"
MARGIN_CLOSE_URL = "https://apiv2.nobitex.ir/positions/{position_id}/close"

# TECHNICAL ANALYSIS FUNCTIONS
AUTO_SELECT_5 = True

def get_user_timeframe():
    """
    If AUTO_SELECT_5 is True this will immediately return the 5-minute timeframe
    (resolution "5", minutes_per_candle 5) without asking the user.
    If AUTO_SELECT_5 is False, it falls back to the original interactive prompt.
    """
    if AUTO_SELECT_5:
        print("Auto-selecting 5-minute timeframe (no prompt).")
        return ("5", 5)

    # original interactive prompt (kept for convenience)
    print("Select a timeframe:")
    print("1: 1 minute")
    print("2: 5 minutes")
    print("3: *15 minutes")
    print("4: 1 hour")
    print("5: 4 hours")
    print("6: 6 hours")
    print("7: 12 hours")
    print("8: 1 day")
    
    choice = input("Enter the number corresponding to your desired timeframe: ")
    
    timeframes = {
        "1": ("1", 1),
        "2": ("5", 5),
        "3": ("15", 15),
        "4": ("60", 60),
        "5": ("240", 240),
        "6": ("360", 360),
        "7": ("720", 720),
        "8": ("D", 1440)
    }
    
    return timeframes.get(choice, ("15", 15))

def get_nobitex_data(symbol, resolution, minutes_per_candle, limit=1000):
    end_time = int(pd.Timestamp.now(tz=TEHRAN_TZ).timestamp())
    start_time = end_time - limit * minutes_per_candle * 60
    url = f"https://apiv2.nobitex.ir/market/udf/history?symbol={symbol}&resolution={resolution}&from={start_time}&to={end_time}"
    try:
        response = requests.get(url, timeout=0.9)
    except ReqTimeout:
        raise
    data = response.json()

    if data.get("s") != "ok":
        raise ValueError(f"Error fetching data for {symbol}: {data}")

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["t"], unit="s", utc=True).tz_convert(TEHRAN_TZ),
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data["v"]
    })
    df.set_index("timestamp", inplace=True)
    return df

def calculate_indicators(df):
    df["20_MA"] = df["close"].rolling(window=20).mean()
    df["50_MA"] = df["close"].rolling(window=50).mean()
    df["200_MA"] = df["close"].rolling(window=200).mean()
    df["RSI"] = calculate_rsi(df, period=15)
    df = calculate_bollinger_bands(df)
    df = calculate_ichimoku(df)
    df = calculate_obv(df)
    return df

def calculate_rsi(df, period=15):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(df, window=20, num_std=2):
    df["BB_Middle"] = df["close"].rolling(window=window).mean()
    df["BB_Std"] = df["close"].rolling(window=window).std()
    df["BB_Upper"] = df["BB_Middle"] + num_std * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - num_std * df["BB_Std"]
    return df

def calculate_ichimoku(df):
    high = df['high']
    low = df['low']
    period9_high = high.rolling(window=9).max()
    period9_low = low.rolling(window=9).min()
    df['tenkan_sen'] = (period9_high + period9_low) / 2
    period26_high = high.rolling(window=26).max()
    period26_low = low.rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    period52_high = high.rolling(window=52).max()
    period52_low = low.rolling(window=52).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
    df['chikou_span'] = df['close'].shift(-26)
    return df

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    return df

def check_bullish_candlestick_patterns(df):
    if len(df) < 2:
        return []
    last_candle = df.iloc[-1]
    second_last_candle = df.iloc[-2]
    hammer = (last_candle["close"] > last_candle["open"]) and \
             (last_candle["low"] < last_candle["open"]) and \
             ((last_candle["high"] - last_candle["close"]) < (last_candle["close"] - last_candle["low"]) * 0.3)
    engulfing = (second_last_candle["close"] < second_last_candle["open"]) and \
                (last_candle["close"] > last_candle["open"]) and \
                (last_candle["close"] > second_last_candle["open"]) and \
                (last_candle["open"] < second_last_candle["close"])
    three_white_soldiers = len(df) >= 3 and all(df["close"].iloc[-i] > df["open"].iloc[-i] for i in range(1, 4))
    patterns = []
    if hammer:
        patterns.append("Hammer")
    if engulfing:
        patterns.append("Bullish Engulfing")
    if three_white_soldiers:
        patterns.append("Three White Soldiers")
    return patterns

def check_chart_patterns(df):
    close_prices = df["close"]
    double_bottom = len(close_prices) >= 3 and (close_prices.iloc[-3] > close_prices.iloc[-2]) and (close_prices.iloc[-1] > close_prices.iloc[-2])
    golden_cross = df["50_MA"].iloc[-1] > df["200_MA"].iloc[-1]
    rsi_oversold = df["RSI"].iloc[-1] < 30
    high_volume = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1]
    patterns = []
    if double_bottom:
        patterns.append("Double Bottom")
    if golden_cross:
        patterns.append("Golden Cross")
    if rsi_oversold:
        patterns.append("RSI Oversold")
    if high_volume:
        patterns.append("High Volume Confirmation")
    return patterns

def check_fibonacci_support_resistance(df):
    fib_levels, recent_high, recent_low = calculate_fibonacci_levels(df)
    current_price = df["close"].iloc[-1]
    fib_patterns = []
    for level, price in fib_levels.items():
        if abs(current_price - price) < (recent_high - recent_low) * 0.02:
            if price <= recent_low:
                fib_patterns.append(f"Near Support at {level} Level ({price:.2f})")
            elif price >= recent_high:
                fib_patterns.append(f"Near Resistance at {level} Level ({price:.2f})")
            else:
                fib_patterns.append(f"Near {level} Level ({price:.2f})")
    return fib_patterns

def calculate_fibonacci_levels(df):
    recent_high = df["high"].max()
    recent_low = df["low"].min()
    difference = recent_high - recent_low
    fib_levels = {
        "23.6%": recent_high - difference * 0.236,
        "38.2%": recent_high - difference * 0.382,
        "50.0%": recent_high - difference * 0.5,
        "61.8%": recent_high - difference * 0.618,
        "100%": recent_high
    }
    return fib_levels, recent_high, recent_low

def check_bollinger_signal(df):
    last_close = df["close"].iloc[-1]
    last_upper = df["BB_Upper"].iloc[-1]
    signal = []
    if last_close > last_upper:
        signal.append("Bollinger Breakout")
    return signal

def check_ichimoku_signal(df):
    last_row = df.iloc[-1]
    if pd.isna(last_row['senkou_span_a']) or pd.isna(last_row['senkou_span_b']):
        return []
    cloud_top = max(last_row['senkou_span_a'], last_row['senkou_span_b'])
    price_above_cloud = last_row['close'] > cloud_top
    tenkan_above_kijun = last_row['tenkan_sen'] > last_row['kijun_sen']
    signals = []
    if price_above_cloud and tenkan_above_kijun:
        signals.append("Ichimoku Bullish")
    return signals

def check_obv_signal(df, lookback=3):
    if len(df) < lookback + 1:
        return []
    if df["OBV"].iloc[-1] > df["OBV"].iloc[-lookback]:
        return ["OBV Bullish"]
    return []

def check_sma_signal(df, lookback=3):
    if len(df) < lookback + 1:
        return []
    if "20_MA" not in df.columns:
        df["20_MA"] = df["close"].rolling(window=20).mean()
    if df["close"].iloc[-1] > df["20_MA"].iloc[-1] and df["20_MA"].iloc[-1] > df["20_MA"].iloc[-lookback]:
        return ["SMA Bullish"]
    return []

def count_consecutive_bullish(df):
    count = 0
    for idx in range(len(df) - 1, -1, -1):
        candle = df.iloc[idx]
        if candle["close"] > candle["open"]:
            count += 1
        else:
            break
    return count

def count_consecutive_bearish(df):
    count = 0
    for idx in range(len(df) - 1, -1, -1):
        candle = df.iloc[idx]
        if candle["close"] < candle["open"]:
            count += 1
        else:
            break
    return count

def calculate_confirmation_candles(df, all_patterns, base_required=3):
    if "Three White Soldiers" in all_patterns:
        print("Three White Soldiers detected: confirmation complete.")
        return 0
    required = base_required
    print(f"Base confirmation requirement: {required}")
    recent_bearish = count_consecutive_bearish(df)
    required += recent_bearish
    print(f"Detected consecutive bearish candles: {recent_bearish}  --> Requirement now: {base_required} + {recent_bearish} = {required}")
    rsi_value = df["RSI"].iloc[-1] if "RSI" in df.columns else 50
    if rsi_value > 70:
        required += 1
        print(f"RSI is {rsi_value} (above 70)              --> Requirement increased by 1: {required}")
    else:
        print(f"RSI is {rsi_value} (normal)                --> No adjustment needed")
    avg_volume = df["volume"].rolling(window=20).mean().iloc[-1]
    latest_volume = df["volume"].iloc[-1]
    if latest_volume < avg_volume:
        required += 1
        print(f"Volume is below 20-candle average --> Requirement increased by 1: {required}")
    else:
        print(f"Volume is above 20-candle average --> No adjustment needed")
    volatility = df["close"].pct_change().rolling(window=20).std().iloc[-1]
    if volatility > 0.05:
        required += 1
        print(f"Volatility is high (>5%)           --> Requirement increased by 1: {required}")
    else:
        print(f"Volatility is normal              --> No adjustment needed")
    recent_bullish = count_consecutive_bullish(df)
    print(f"Detected consecutive bullish candles: {recent_bullish}")
    confirmation_candles = max(0, required - recent_bullish)
    print(f"Final confirmation candles needed: {confirmation_candles}\n")
    return confirmation_candles

def calculate_percentage_increase_gb(df):
    data = df.dropna()
    if len(data) < 2:
        return 0.0
    data = data.copy()
    data['target'] = data['close'].shift(-1)
    data['target'] = (data['target'] - data['close']) / data['close'] * 100
    data = data[:-1]
    features = ['open', 'high', 'low', 'close', 'volume', '20_MA', '50_MA', '200_MA', 'RSI',
                'BB_Middle', 'BB_Upper', 'BB_Lower', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']
    features = [f for f in features if f in data.columns]
    X = data[features]
    y = data['target']
    model = GradientBoostingRegressor()
    model.fit(X, y)
    last_row = df.iloc[-1]
    X_pred = last_row[features].values.reshape(1, -1)
    prediction = model.predict(X_pred)[0]
    return prediction

def calculate_stoch_rsi_blue(df, rsi_period=14, stoch_period=14, smooth_period=3):
    rsi = calculate_rsi(df, period=rsi_period)
    rsi_min = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
    denominator = rsi_max - rsi_min
    raw_stoch_rsi = (rsi - rsi_min) / denominator.replace(0, np.nan) * 100
    raw_stoch_rsi = raw_stoch_rsi.fillna(0)
    stoch_rsi_blue = raw_stoch_rsi.rolling(window=smooth_period).mean()
    return stoch_rsi_blue

# SYMBOL ANALYSIS STAGES
def compute_stage_one(symbol, resolution, minutes_per_candle):
    try:
        df = get_nobitex_data(symbol, resolution, minutes_per_candle)
    except ReqTimeout:
        print(f"{symbol}: ⏱️ data fetch timed out in Stage 1, skipping")
        return None
    except Exception as e:
        print(f"{symbol}: Error fetching data in Stage 1: {e}")
        return None

    if len(df) < 50:
        print(f"{symbol}: Not enough data in Stage 1.")
        return None

    df = calculate_indicators(df)
    all_patterns = (
        check_bullish_candlestick_patterns(df) +
        check_chart_patterns(df) +
        check_fibonacci_support_resistance(df) +
        check_bollinger_signal(df) +
        check_ichimoku_signal(df) +
        check_obv_signal(df) +
        check_sma_signal(df)
    )
    score = len(all_patterns)
    return {"symbol": symbol, "score": score, "patterns": all_patterns, "df": df}

def compute_stage_two(result):
    if result is None or "df" not in result or "patterns" not in result:
        print(f"{result['symbol'] if result else 'Unknown'}: Invalid input for Stage 2")
        return None
    df = result["df"]
    all_patterns = result["patterns"]
    confirm = calculate_confirmation_candles(df, all_patterns)
    return {"symbol": result["symbol"], "confirmation_candles": confirm}

def compute_stage_three(symbol, resolution, minutes_per_candle):
    try:
        df = get_nobitex_data(symbol, resolution, minutes_per_candle)
    except ReqTimeout:
        print(f"{symbol}: ⏱️ data fetch timed out in Stage 3, skipping")
        return None
    except Exception as e:
        print(f"{symbol}: Error fetching data in Stage 3: {e}")
        return None

    if len(df) < 50:
        print(f"{symbol}: Not enough data in Stage 3.")
        return None

    df = calculate_indicators(df)
    stoch5 = calculate_stoch_rsi_blue(df).iloc[-1]
    result = {"symbol": symbol, "stoch_rsi_blue": stoch5}

    if minutes_per_candle == 5:
        try:
            df15 = get_nobitex_data(symbol, "15", 15)
            if len(df15) >= 50:
                df15 = calculate_indicators(df15)
                stoch15v = calculate_stoch_rsi_blue(df15).iloc[-1]
                result["stoch_rsi_blue_15h"] = stoch15v
        except:
            pass
    return result

# TRADING FUNCTIONS
def get_wallet_balance_for(currency):
    headers = {"Authorization": f"Token {TRADING_TOKEN}"}
    query = currency.lower()
    url = f"https://apiv2.nobitex.ir/v2/wallets?currencies={query}&type=margin"
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            print(f"API returned non-ok status: {data}")
            return None
        wallets = data.get("wallets", {})
        wallet_data = wallets.get(currency.upper())
        if not wallet_data:
            print(f"No margin wallet found for {currency.upper()}")
            return None
        balance = float(wallet_data.get("balance", "0"))
        print(f"Margin wallet balance for {currency.upper()}: {balance}")
        return balance
    except Exception as e:
        print(f"Error fetching balance for {currency.upper()}: {e}")
        return None

def get_orderbook_ask_price(symbol):
    url = f"https://apiv2.nobitex.ir/v3/orderbook/{symbol}"
    response = requests.get(url)
    data = response.json()
    if data.get("asks") and len(data["asks"]) > 0:
        best_ask_price = float(data["asks"][0][0])
        print(f"Fetched best ask price for {symbol}: {best_ask_price}")
        return best_ask_price
    else:
        raise ValueError(f"No ask prices available for {symbol}")

def get_orderbook_bid_price(symbol):
    url = f"https://apiv2.nobitex.ir/v3/orderbook/{symbol}"
    response = requests.get(url)
    data = response.json()
    if data.get("bids") and len(data["bids"]) > 0:
        best_bid_price = float(data["bids"][0][0])
        print(f"Fetched best bid price for {symbol}: {best_bid_price}")
        return best_bid_price
    else:
        raise ValueError(f"No bid prices available for {symbol}")

def execute_buy_order_margin(symbol, leverage="5", price_selector=get_orderbook_bid_price, avg=None):
    try:
        lev = float(leverage)
    except ValueError:
        print(f"Invalid leverage '{leverage}', defaulting to 1x.")
        lev = 1.0

    balance = get_wallet_balance_for("rls")
    if balance is None or balance <= 0:
        print("Insufficient rial balance, skipping margin buy.")
        return None

    try:
        price = price_selector(symbol)
        order_type = "buy" if avg is not None and avg > 1 else "sell"
        amount = (balance * lev) / price
        exact_amount = round(amount, 8)
        formatted_amount = f"{exact_amount:.8f}"
        src_currency = symbol[:-3].lower()

        order_payload = {
            "type": order_type,
            "srcCurrency": src_currency,
            "dstCurrency": "rls",
            "leverage": f"{lev:.0f}",
            "amount": formatted_amount,
            "price": f"{price:.2f}",
            "execution": "market"
        }
        headers = {
            "Authorization": f"Token {TRADING_TOKEN}",
            "content-type": "application/json"
        }

        print(f"Placing MARGIN {order_type.upper()} order: {order_payload}")
        response = requests.post(MARGIN_ORDER_URL, json=order_payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"Margin {order_type} order placed: {result}")

        order = result.get("order", {})
        position_id = order.get("id")
        if not position_id:
            print("No positionId returned in response! Cannot proceed.")
            return None
        amount = float(order.get("amount", 0))
        return price, position_id, formatted_amount
    except Exception as e:
        print(f"Error placing margin {order_type} order for {symbol}: {e}")
        return None

def get_open_position_details(symbol):
    src_currency = symbol[:-3].lower()
    url = f"https://apiv2.nobitex.ir/positions/list?srcCurrency={src_currency}&status=active"
    headers = {"Authorization": f"Token {TRADING_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok":
            print("Failed to fetch position list:", data)
            return None, None
        for pos in data.get("positions", []):
            if pos["status"].lower() == "open":
                pos_id = pos["id"]
                liability = pos.get("liability") or pos.get("delegatedAmount")
                print(f"Found position for {symbol} → ID: {pos_id} | Liability: {liability}")
                return pos_id, liability
        print(f"No open position found for {symbol}")
        return None, None
    except Exception as e:
        print(f"Error getting position details: {e}")
        return None, None

def fetch_unrealized_pnl_percent(symbol: str, position_id: str) -> float:
    src = symbol[:-3].lower()
    url = f"https://apiv2.nobitex.ir/positions/list?srcCurrency={src}&status=active"
    headers = {"Authorization": f"Token {TRADING_TOKEN}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    for pos in resp.json().get("positions", []):
        if str(pos.get("id")) == str(position_id):
            return float(pos.get("unrealizedPNLPercent", 0))
    return 0.0


def compute_pct_from_orderbook(symbol: str, avg: float, position_id: str | None = None, price: float | None = None) -> float | None:
    """
    Returns a reliable pct value or None if not computable.
    - Validates orderbook prices (ask/bid > 0).
    - Protects against division by zero and spurious huge values.
    - Falls back to server unrealizedPNLPercent only if it looks sane.
    """
    try:
        ask = get_orderbook_ask_price(symbol)
        bid = get_orderbook_bid_price(symbol)

        # sanity checks
        if ask is None or bid is None:
            raise ValueError("Orderbook returned no bid/ask")

        if not (math.isfinite(ask) and math.isfinite(bid)):
            raise ValueError("Non-finite orderbook prices")

        # reject obviously bad prices (zero or negative)
        if ask <= 0 or bid <= 0:
            raise ValueError(f"Unreasonable orderbook prices: ask={ask}, bid={bid}")

        # prepare denom price (use provided buy price first)
        denom_price = None
        if price is not None:
            try:
                denom_price = float(f"{float(price):.2f}")
                if denom_price <= 0:
                    denom_price = None
            except Exception:
                denom_price = None

        pct = None
        if avg is not None and avg > 2.4:
            denom = denom_price if denom_price is not None else ask
            if denom <= 0:
                raise ZeroDivisionError("Denominator is zero/invalid")
            pct = ((bid - denom) * 100.0) / denom
        elif avg is not None and avg < -2.4:
            denom = denom_price if denom_price is not None else bid
            if denom <= 0:
                raise ZeroDivisionError("Denominator is zero/invalid")
            pct = ((ask - denom) * 100.0) / denom
        else:
            # avg neutral -> don't compute orderbook pct
            return None

        # sanity clamp: reject NaN/inf and extremely large numbers (likely API glitch)
        if pct is None or not math.isfinite(pct):
            raise ValueError("Computed pct not finite")
        if abs(pct) > 1000:   # arbitrary safety cap; adjust if you expect legitimate >1000%
            print(f"Rejected spurious pct={pct:.2f} (clamped).")
            return None

        return float(pct)

    except Exception as e:
        print(f"compute_pct_from_orderbook: orderbook error for {symbol}: {e}")
        # fallback: try server-side unrealized PNL only if position_id supplied
        if position_id is not None:
            try:
                fallback = fetch_unrealized_pnl_percent(symbol, position_id)
                # validate fallback
                if fallback is None or not math.isfinite(fallback):
                    print("Fallback unrealizedPNLPercent not usable.")
                    return None
                # clamp unrealistic fallback values too
                if abs(fallback) > 1000:
                    print("Fallback unrealizedPNLPercent out of reasonable range; ignoring.")
                    return None
                return float(fallback)
            except Exception as e2:
                print(f"Fallback fetch_unrealized_pnl_percent failed: {e2}")
        return None



def execute_close_position_margin(symbol, position_id, amount, price_selector=get_orderbook_ask_price):
    asset = symbol[:-3].lower()
    try:
        price = price_selector(symbol)
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return
    url = MARGIN_CLOSE_URL.format(position_id=position_id)
    payload = {
        "amount": amount,
        "price": f"{price:.2f}",
        "execution": "market",
    }
    headers = {
        "Authorization": f"Token {TRADING_TOKEN}",
        "content-type": "application/json"
    }
    print(f"Closing margin position {position_id} for {symbol}: {payload}")
    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        print(f"Position {position_id} closed: {resp.json()}")
    except Exception as e:
        print(f"Error closing position {position_id}: {e}")

# COUNTDOWN FUNCTION
def countdown(wait_seconds):
    start_time = time.time()
    avg = fetch_and_display_changes()
    while True:
        elapsed = time.time() - start_time
        remaining = int(wait_seconds - elapsed)
        if remaining <= 0:
            break
        print(f"Countdown: {remaining} seconds remaining...", end="\r")
        time.sleep(1)
    print("Countdown finished! Starting analysis...         ")
    return avg

# SYMBOLS LIST
symbols = [
    "DOGSIRT", "HMSTRIRT", "SIRT", "TIRT", "1B_BABYDOGEIRT",
    "1M_PEPEIRT", "100K_FLOKIIRT", "NOTIRT", "API3IRT", "ZRXIRT",
    "CVXIRT", "ENSIRT", "SNTIRT", "STORJIRT", "SLPIRT", "CRVIRT",
    "AAVEIRT", "XMRIRT", "LDOIRT", "DYDXIRT", "APTIRT", "MAGICIRT",
    "CHZIRT", "ATOMIRT", "NEARIRT", "FILIRT", "LINKIRT",
    "DOTIRT", "UNIIRT", "XLMIRT", "GMTIRT", "POLIRT", "AVAXIRT",
    "SOLIRT", "ETCIRT", "BCHIRT", "LTCIRT", "TRXIRT", "ADAIRT",
    "DOGEIRT", "XRPIRT", "BNBIRT", "TONIRT",
    "ARBIRT", "1k_SHIBIRT", "ONEIRT", "HBARIRT","RENDERIRT"
]


def process_symbol(symbol, resolution, minutes_per_candle):
    """Process all three stages sequentially for a single symbol"""
    try:
        # Stage 1
        stage1 = compute_stage_one(symbol, resolution, minutes_per_candle)
        if not stage1:
            return None
        
        # Stage 2
        stage2 = compute_stage_two(stage1)
        if not stage2:
            return None

        # Stage 3 (modified to reuse Stage 1 data)
        try:
            df = stage1["df"]
            stoch5 = calculate_stoch_rsi_blue(df).iloc[-1]
            result = {"symbol": symbol, "stoch_rsi_blue": stoch5}
            
            # Additional 15m check if needed
            if minutes_per_candle == 5:
                try:
                    df15 = get_nobitex_data(symbol, "15", 15)
                    if len(df15) >= 50:
                        df15 = calculate_indicators(df15)
                        stoch15v = calculate_stoch_rsi_blue(df15).iloc[-1]
                        result["stoch_rsi_blue_15h"] = stoch15v
                except Exception:
                    pass
        except Exception as e:
            print(f"{symbol}: Stage 3 failed: {e}")
            return None

        # Combine results
        combined_result = {
            "symbol": symbol,
            "score": stage1["score"],
            "patterns": stage1["patterns"],
            "confirmation_candles": stage2["confirmation_candles"],
            "stoch_rsi_blue": result["stoch_rsi_blue"],
            "current_price": stage1["df"]["close"].iloc[-1],
            "percentage_increase": calculate_percentage_increase_gb(stage1["df"])
        }
        if "stoch_rsi_blue_15h" in result:
            combined_result["stoch_rsi_blue_15h"] = result["stoch_rsi_blue_15h"]
            
        return combined_result
    except Exception as e:
        print(f"{symbol}: Pipeline error: {e}")
        return None
    

# MAIN LOOP
def main_loop():
    resolution, minutes_per_candle = get_user_timeframe()
    period_seconds = minutes_per_candle * 60

    while True:
        now = datetime.now(TEHRAN_TZ)
# the program tries to come in the the bazzaer with consistance trend 
        
        if 0 <= now.hour < 9  or 16 <= now.hour < 18 :
            print(f"Quiet hours: Current Iran time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(10)
            continue

        local_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_since_midnight = (now - local_midnight).total_seconds()
        base_candle_index = int(seconds_since_midnight // period_seconds)
        next_candle_time = local_midnight + timedelta(seconds=(base_candle_index + 1) * period_seconds)
        analysis_time = next_candle_time - timedelta(minutes=1.85)

        if now > analysis_time:
            print("Analysis window passed for this cycle. Waiting for next cycle...")
            time.sleep(10)
            continue

        wait_analysis = (analysis_time - now).total_seconds()
        print(f"\nCurrent Iran time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Upcoming candle closes at: {next_candle_time.strftime('%H:%M:%S')}")
        print(f"Analysis window scheduled at: {analysis_time.strftime('%H:%M:%S')}")
        print(f"Waiting {wait_analysis:.0f} seconds until analysis window...\n")

        avg = countdown(wait_analysis)
    
        # Tight timing synchronization
        while (analysis_time - datetime.now(TEHRAN_TZ)).total_seconds() > 1:
            time.sleep(0.5)
        while datetime.now(TEHRAN_TZ) < analysis_time:
            time.sleep(0.001)
            
        print("Starting parallel analysis...\n")

        analysis_results = []
        timed_out_symbols = []

        # Process all symbols in parallel with dynamic result handling
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = {executor.submit(process_symbol, sym, resolution, minutes_per_candle): sym for sym in symbols}
            
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    res = future.result(timeout=10)
                    if res:
                        analysis_results.append(res)
                        # Immediate result printing
                        print(f"{sym} analysis:")
                        print(f"  Score: {res['score']} | Price: {res['current_price']:.2f} rls | %Δ: {res['percentage_increase']:.2f}%")
                        print(f"  Patterns: {res['patterns']}")
                        print(f"  Confirm candles: {res['confirmation_candles']}")
                        summary = f"  StochRSI-5m: {res['stoch_rsi_blue']:.2f}"
                        if "stoch_rsi_blue_15h" in res:
                            summary += f" | StochRSI-15h: {res['stoch_rsi_blue_15h']:.2f}"
                        print(summary + "\n")
                    else:
                        timed_out_symbols.append(sym)
                except Exception as e:
                    print(f"{sym}: Pipeline failed: {e}")
                    timed_out_symbols.append(sym)

        # Retry logic for failed symbols
        if timed_out_symbols:
            print("\n🔄 Retrying timed-out symbols...")
            with ThreadPoolExecutor(max_workers=50) as retry_executor:
                retry_futures = {retry_executor.submit(process_symbol, sym, resolution, minutes_per_candle): sym 
                               for sym in timed_out_symbols}
                
                for future in as_completed(retry_futures):
                    sym = retry_futures[future]
                    try:
                        res = future.result(timeout=30)
                        if res:
                            analysis_results.append(res)
                            print(f"{sym} analysis (retry):")
                            print(f"  Score: {res['score']} | Price: {res['current_price']:.2f} rls | %Δ: {res['percentage_increase']:.2f}%")
                            print(f"  Patterns: {res['patterns']}")
                            print(f"  Confirm candles: {res['confirmation_candles']}")
                            summary = f"  StochRSI-5m: {res['stoch_rsi_blue']:.2f}"
                            if "stoch_rsi_blue_15h" in res:
                                summary += f" | StochRSI-15h: {res['stoch_rsi_blue_15h']:.2f}"
                            print(summary + "\n")
                    except Exception as e:
                        print(f"{sym}: Final retry failed: {e}")

        if avg is None:
            print("Couldn't calculate average change. Skipping this cycle.")
            time.sleep(10)
            continue

        if avg > 2.4:
            print("🟢 Strong positive market sentiment detected.")
            price_selector_buy = get_orderbook_ask_price
            price_selector_sell = get_orderbook_bid_price
            EPSILON = 1e-2 
            if minutes_per_candle == 5:
                candidates = [
                    r for r in analysis_results
                    if abs(r["stoch_rsi_blue"]) < EPSILON
                    and r.get("stoch_rsi_blue_15h") is not None
                    and r["stoch_rsi_blue_15h"] < 20.00
                ]
            else:
        
                candidates = [
                    r for r in analysis_results
                    if abs(r["stoch_rsi_blue"]) < EPSILON
                ]
            selected = [max(candidates, key=lambda x: (x["score"], -x["confirmation_candles"]))] if candidates else []

        elif avg < -2.4:
            print("🔴 Strong negative market sentiment detected.")
            price_selector_buy = get_orderbook_bid_price
            price_selector_sell = get_orderbook_ask_price
            if minutes_per_candle == 5:
                candidates = [
                    r for r in analysis_results
                    if r["stoch_rsi_blue"] == 100.00
                    and r.get("stoch_rsi_blue_15h") is not None
                    and r["stoch_rsi_blue_15h"] > 80.00
                ]
            else:
                candidates = [r for r in analysis_results if r["stoch_rsi_blue"] == 0.00]
            selected = [min(candidates, key=lambda x: (x["score"], -x["confirmation_candles"]))] if candidates else []

        else:
            print("\nCycle complete. Market neutral. Repeating for next cycle...\n")
            time.sleep(10)
            continue

        bought = []
        if selected:
            print("Executing BUY order for selected symbol(s)...")
            for res in selected:
                sym = res["symbol"]
                print(f"→ BUY {sym} at {datetime.now(TEHRAN_TZ).strftime('%H:%M:%S')}")
                buy = execute_buy_order_margin(sym, price_selector=price_selector_buy, avg=avg)
                if buy:
                    buy_price, position_id, amt = buy
                    res.update({"buy_price": buy_price, "position_id": position_id, "liability": amt})
                    bought.append(res)
        else:
            print("No symbols selected for trading this cycle.")
        if bought:
            for res in bought:
                sym = res["symbol"]
                submitted_id = res["position_id"]
                print(f"\nWaiting for position to open on Nobitex for {sym} (temp id: {submitted_id})…")
                while True:
                    pos_id, liability = get_open_position_details(sym)
                    if pos_id:
                        res["position_id"] = pos_id
                        res["liability"] = liability
                        print(f"→ Position opened: ID={pos_id}, amount={liability}")
                        break
                    time.sleep(1)
                    
            for res in bought:
                sym = res["symbol"]
                position_id = res["position_id"]
                liability = res["liability"]
                print(f"\nMonitoring {sym} for computed PCT every 2 seconds (Buy Price: {res['buy_price']:.2f} rls)…")
                monitor_start = datetime.now()
                while True:
                    try:

                        elapsed_hours = (datetime.now() - monitor_start).total_seconds() / 3600
                        days_passed = int(elapsed_hours // 24)

                        pos_threshold = 0.65 + (0.10 * days_passed)
                        neg_threshold = -0.65 - (0.10 * days_passed)

                        pct = compute_pct_from_orderbook(sym, avg, position_id, price=res.get("buy_price"))
                        if pct is None:
                            print(f"  Could not compute reliable PCT for {sym} — skipping this iteration.   ", end="\r")
                            time.sleep(0.4)
                            continue
                        print(f"  Computed PCT (orderbook-based): {pct:.2f}%   ", end="\r")
                        if (avg > 2.4 and pct > pos_threshold) or pct <= -4:
                            print(f"\nTarget reached ({pct:.2f}%). Closing {sym}…")
                            execute_close_position_margin(sym, position_id, amount=liability, price_selector=price_selector_sell)
                            break
                        elif (avg < -2.4 and pct < neg_threshold) or pct >= 4:
                            print(f"\nStop-loss triggered ({pct:.2f}%). Closing {sym}…")
                            execute_close_position_margin(sym, position_id, amount=liability, price_selector=price_selector_sell)
                            break
                    except ReqTimeout:    
                        print(f"  Timeout fetching orderbook for {sym} — retrying in 0.4s...   ", end="\r", flush=True)
                    except Exception as e:
                        print(f"  Error monitoring {sym}: {e} — retrying in 0.4s...   ", end="\r", flush=True)
                    time.sleep(0.4)
            else:
             print("No positions to monitor/close.")

        print("\nCycle complete. Repeating process for the next cycle...\n")
        time.sleep(10)

# RSI MODE
def show_stoch_rsi_blue():
    resolution, minutes_per_candle = get_user_timeframe()
    print("\nCalculating Stochastic RSI Blue Line for each symbol...\n")
    for symbol in symbols:
        try:
            df = get_nobitex_data(symbol, resolution, minutes_per_candle)
            if len(df) < 50:
                print(f"{symbol}: Not enough data to compute Stoch RSI reliably.")
                continue
            stoch_rsi_blue_series = calculate_stoch_rsi_blue(df)
            stoch_rsi_blue = stoch_rsi_blue_series.iloc[-1]
            print(f"{symbol}: Stochastic RSI Blue Line = {stoch_rsi_blue:.8f}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    print("\nStoch RSI (blue line) comparisons complete!\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "blue":
        show_stoch_rsi_blue()
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "sell":
        try:
            avg = float(input("Enter the daily average (avg): "))
        except ValueError:
            print("Invalid average. Please enter a numeric value.")
            sys.exit()
        symbol = input("Enter the symbol to trade (e.g., BTCIRT, ETHIRT, etc.): ").strip().upper()
        if symbol not in symbols:
            print("Invalid symbol.")
            sys.exit()
        if avg > 1:
            price_selector_buy = get_orderbook_ask_price
            price_selector_current = get_orderbook_bid_price
        elif avg < -1:
            price_selector_buy = get_orderbook_bid_price
            price_selector_current = get_orderbook_ask_price
        else:
            print("Market sentiment is neutral. No action taken.")
            sys.exit()
        buy_result = execute_buy_order_margin(symbol, price_selector=price_selector_buy, avg=avg)
        if not buy_result:
            print("❌ Buy order failed.")
            sys.exit()
        buy_price, submitted_position_id, _ = buy_result
        print(f"Buy order submitted at {buy_price:.2f} for {symbol[:-3]} (pos id: {submitted_position_id})")
        print("Waiting for position to open on Nobitex…")
        position_id = None
        liability = None
        while True:
            pos_id, liab = get_open_position_details(symbol)
            if pos_id:
                position_id = pos_id
                liability = liab
                print(f"→ Position opened: ID={position_id}, amount={liability}")
                break
            time.sleep(1)
        print("Entering continuous monitor of Unrealized PNL% every 3 seconds (Ctrl+C to stop)…")
        while True:
            try:
                pct = fetch_unrealized_pnl_percent(symbol, position_id)
                print(f"  Unrealized PNL%: {pct:.2f}%   ", end="\r")
            except Exception as e:
                print(f"Error fetching PNL%: {e}", end="\r")
            time.sleep(3)
    else:
        main_loop()




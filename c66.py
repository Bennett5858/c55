import asyncio
import json
import time
import numpy as np
import joblib
import pandas as pd
from datetime import datetime, timezone, time as dtime
from sklearn.ensemble import RandomForestClassifier
import websockets
from colorama import Fore, init
from rich.console import Console
from rich.table import Table
from rich.live import Live
from sklearn.model_selection import GridSearchCV

# === INIT ===
init(autoreset=True)
console = Console()

# === CONFIGURATION ===
APP_ID = "82214"
WS_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
SYMBOL = "1HZ10V"
AUTH_TOKEN = "2G2HGBSga15fSlS"  # Replace with your token

# === AI TRADING SYSTEM ===
class TradeDataCollector:
    def __init__(self):
        self.data = {
            "volatility_10": [],
            "velocity": [],
            "volume": [],
            "profit_10": [],
            "result": []
        }
        self.trade_history = []

    def add_trade(self, price, volume, result):
        if price <= 0 or volume < 0 or result not in [0, 1]:
            print("⚠️ Invalid trade data ignored.")
            return

        volatility_10 = 0.0
        profit_10 = 0.0
        if len(self.trade_history) >= 10:
            recent_prices = [trade["price"] for trade in self.trade_history[-10:]]
            volatility_10 = np.std(recent_prices)
            profit_10 = sum(trade["result"] for trade in self.trade_history[-10:])

        velocity = price - self.trade_history[-1]["price"] if self.trade_history else 0.0

        self.data["volatility_10"].append(volatility_10)
        self.data["velocity"].append(velocity)
        self.data["volume"].append(volume)
        self.data["profit_10"].append(profit_10)
        self.data["result"].append(result)

        self.trade_history.append({"price": price, "result": result})

    def get_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.data)

class HybridTradingSystem:
    def __init__(self):
        self.data_collector = TradeDataCollector()
        self.ai_model = None
        self.trade_count = 0
        self.ai_ready = False

    def train_model(self, df):
        features = ["volatility_10", "velocity", "volume", "profit_10"]
        X = df[features]
        y = df["result"]

        # Advanced: Use class weights to handle imbalance, and grid search for best params

        param_grid = {
            "n_estimators": [100, 150, 200],
            "max_depth": [8, 10, 12],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": ["balanced"]
        }

        rf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring="accuracy")
        grid.fit(X, y)
        best_model = grid.best_estimator_

        joblib.dump(best_model, "ai_trade_model.pkl")
        print(f"✅ Model trained (best params: {grid.best_params_}) and saved.")
        # return best_model
        return best_model

    def evaluate_model(self, model, df):
        features = ["volatility_10", "velocity", "volume", "profit_10"]
        X = df[features]
        y = df["result"]
        y_pred = model.predict(X)
        accuracy = (y_pred == y).mean()
        print(f"📊 Model Accuracy: {accuracy:.2%}")
        return accuracy

    def ai_filter(self, features):
        feature_order = ["volatility_10", "velocity", "volume", "profit_10"]
        import pandas as pd
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        X = pd.DataFrame([features], columns=feature_order)
        proba = self.ai_model.predict_proba(X)[0][1]
        return proba > 0.65, proba

    def decide_trade(self, price, volume, heuristic_decision):
        self.trade_count += 1
        result = 1 if heuristic_decision else 0
        self.data_collector.add_trade(price, volume, result)
        if self.ai_model is None:
            try:
                self.ai_model = joblib.load("ai_trade_model.pkl")
                print("🔄 AI model loaded successfully.")
            except FileNotFoundError:
                print(Fore.RED + "❌ AI model not found. Using heuristic only.")
                return heuristic_decision
        if distance_from_center(price, CENTERLINE_POSITION) > MAX_CENTER_DRIFT:
            print(Fore.YELLOW + "⚠️ Centerline drift too high. Skipping trade.")
            return False

        # Only check AI confidence after it is defined below

        if not self.ai_ready:
            if self.trade_count >= 100:
                self.ai_model = self.train_model(self.data_collector.get_dataframe())
                self.ai_ready = True
                print("🔓 AI model activated!")
            return heuristic_decision

        features = {
            "volatility_10": np.std([trade["price"] for trade in self.data_collector.trade_history[-10:]]) if len(self.data_collector.trade_history) >= 10 else 0.0,
            "velocity": price - self.data_collector.trade_history[-1]["price"] if self.data_collector.trade_history else 0.0,
            "volume": volume,
            "profit_10": sum(trade["result"] for trade in self.data_collector.trade_history[-10:]) if len(self.data_collector.trade_history) >= 10 else 0.0
        }
        ok, confidence = self.ai_filter(features)
        if confidence < 0.7:
            print(Fore.YELLOW + f"⚠️ Low AI confidence in proposal: {confidence:.2f}. Skipping.")
            return False
        if distance_from_center(price, CENTERLINE_POSITION) > 0.4:
            print(Fore.YELLOW + "⚠️ Centerline drift too high. Skipping proposal.")
            return False
        # Removed undefined accuracy and response checks in this context
        if 'previous_price' in globals() and distance_from_center(previous_price, CENTERLINE_POSITION) > MAX_CENTER_DRIFT:
            print(Fore.YELLOW + "⚠️ Centerline drift too high. Skipping proposal.")
            return None
        if (volatility := calculate_volatility(recent_prices)) > 0.22:
            print(Fore.YELLOW + f"⚠️ High volatility detected: {volatility:.5f}. Skipping proposal.")
            return None
        if hit_daily_profit_target():
            print(Fore.YELLOW + "🏁 Daily profit target hit. Stopping trades.")
            return
        if not is_within_trading_session():
            print(Fore.LIGHTBLACK_EX + "⛔ Outside trading session. Skipping.")
            return

        print(f"🤖 AI Confidence: {confidence:.2%}")
        return ok

# === DERIV INTERACTION ===
async def get_account_balance():
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"authorize": AUTH_TOKEN}))
        while True:
            auth_response = json.loads(await ws.recv())
            if auth_response.get("msg_type") == "authorize":
                break
            if "error" in auth_response:
                print(Fore.RED + f"❌ Auth Error: {auth_response['error']['message']}")
                return 0.0
        await ws.send(json.dumps({"balance": 1}))
        while True:
            response = json.loads(await ws.recv())
            if response.get("msg_type") == "balance" and "balance" in response:
                return float(response["balance"].get("balance", 0.0))
            if "error" in response:
                print(Fore.RED + f"❌ Balance Error: {response['error']['message']}")
                return 0.0

# === PARAMETERS & GLOBALS ===
COOLDOWN_SECONDS = 10
BARRIER_DELTA = 0.33
CENTER_TOLERANCE_MIN = 0.05
CENTER_TOLERANCE_MAX = 0.50
MAX_CENTER_DRIFT = 1.5
RESET_CENTERLINE_EVERY = 5
MOVING_AVG_WINDOW = 20
DAILY_PROFIT_TARGET = 100.0

previous_price = previous_upper = previous_lower = None
last_trade_time = 0
CENTERLINE_POSITION = None
centerline_prices = []
trade_count = 0
wins = losses = 0
profit_total = 0.0
trade_log = []
daily_profit = 0.0
recent_prices = []

dashboard_data = {
    "Price": "-",
    "Centerline": "-",
    "Distance": "-",
    "Cooldown": "-",
    "Signal": "-",
    "Wins": 0,
    "Losses": 0,
    "Profit": "$0.00",
    "Balance": "$0.00"
}

# === UTILS ===
def calculate_barriers(price, delta):
    return round(price + delta, 5), round(price - delta, 5)

def is_within_range(price, lower, upper):
    return lower <= price <= upper

def distance_from_center(price, center):
    return abs(price - center)

def update_centerline(current_price):
    centerline_prices.append(current_price)
    if len(centerline_prices) > MOVING_AVG_WINDOW:
        centerline_prices.pop(0)
    return sum(centerline_prices) / len(centerline_prices)

def auto_tune_parameters():
    global CENTER_TOLERANCE_MAX, BARRIER_DELTA
    if wins > 5 and losses == 0:
        CENTER_TOLERANCE_MAX *= 0.97
        BARRIER_DELTA *= 0.3
    elif losses > wins:
        CENTER_TOLERANCE_MAX *= 0.3
        BARRIER_DELTA *= 0.3

def build_dashboard():
    table = Table(title="📊 Accumulator Strategy Dashboard")
    for key in dashboard_data:
        table.add_row(key, str(dashboard_data[key]))
    return table

def hit_daily_profit_target():
    return daily_profit >= DAILY_PROFIT_TARGET

def is_within_trading_session():
    now_utc = datetime.now(timezone.utc).time()
    return not (dtime(0, 0) <= now_utc <= dtime(4, 0))

def calculate_volatility(prices, window=10):
    if len(prices) < window:
        return 0.0
    diffs = [abs(prices[i] - prices[i-1]) for i in range(1, window)]
    return round(sum(diffs) / len(diffs), 5)

def get_max_bet(balance, pct=0.02):
    return round(balance * pct, 2)

current_bet = None

async def send_authorization(ws):
    await ws.send(json.dumps({"authorize": AUTH_TOKEN}))
    response = json.loads(await ws.recv())
    if "error" in response:
        print(Fore.RED + f"❌ Auth Error: {response['error']['message']}")
        return False
    print(Fore.GREEN + "🔐 Authorized successfully.")
    return True

async def send_proposal(ws):
    await ws.send(json.dumps({
        "proposal": 1,
        "amount": current_bet,
        "basis": "stake",
        "contract_type": "ACCU",
        "currency": "USD",
        "symbol": SYMBOL,
        "product_type": "basic",
        "growth_rate": 0.05,
        "limit_order": { "take_profit": 0.01 }
    }))
    response = json.loads(await ws.recv())
    trading_system = HybridTradingSystem()
    features = {
        "volatility_10": np.std(recent_prices[-10:]) if len(recent_prices) >= 10 else 0.0,
        "velocity": recent_prices[-1] - recent_prices[-2] if len(recent_prices) >= 2 else 0.0,
        "volume": 1,
        "profit_10": sum([t['pnl'] for t in trade_log[-10:]]) if len(trade_log) >= 10 else 0.0
    }
    if volatility := calculate_volatility(recent_prices) > 0.22:
        print(Fore.YELLOW + f"⚠️ High volatility detected: {volatility:.5f}. Adjusting parameters.")
        BARRIER_DELTA = 0.32        
    if trading_system.ai_model is None:
        try:
            trading_system.ai_model = joblib.load("ai_trade_model.pkl")
        except FileNotFoundError:
            pass
    if trading_system.ai_model is not None:
        # Use features and model loaded above for confidence and evaluation
        _, confidence = trading_system.ai_filter(features)
        print(Fore.CYAN + f"🤖 AI Is Analysing...: {confidence:.2%}")
        try:
            # Load data from the model's training data if available, else use current features
            if hasattr(trading_system.ai_model, "X_") and hasattr(trading_system.ai_model, "y_"):
                X = trading_system.ai_model.X_
                y = trading_system.ai_model.y_
                df = pd.DataFrame(X, columns=["volatility_10", "velocity", "volume", "profit_10"])
                df["result"] = y
                
                
            else:
                df = trading_system.data_collector.get_dataframe()
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ Could not load model training data: {e}")
            df = trading_system.data_collector.get_dataframe()
        if df.empty or len(df) < 1:
            print(Fore.YELLOW + "⚠️ Not enough data for AI model evaluation. Skipping proposal.")
            # return None
        try:
            trading_system.ai_model = joblib.load("ai_trade_model.pkl")
        except FileNotFoundError:
            print(Fore.RED + "❌ AI model not found. Skipping proposal.")
            return None
        # Only evaluate accuracy if df is not empty
        # accuracy = trading_system.evaluate_model(trading_system.ai_model, df) if not df.empty else 1.0
    if confidence < 0.7:
        print(Fore.YELLOW + f"⚠️ Low AI confidence in proposal: {confidence:.2f}. Skipping.")
        # if not df.empty and len(df) > 0:
        # trading_system.ai_model = trading_system.train_model(df)
        return None
    if distance_from_center(previous_price, CENTERLINE_POSITION) > 0.4:
        print(Fore.YELLOW + "⚠️ Centerline drift too high. Skipping proposal.")
        return None
    if accuracy < 0.7:
        print(Fore.YELLOW + f"⚠️ AI model accuracy too low: {accuracy:.2f}. Retraining model.")
        # if not df.empty and len(df) > 0:
        trading_system.ai_model = trading_system.train_model(df)
    if "error" in response:
        print(Fore.RED + f"❌ Proposal Error: {response['error']['message']}")
        return None
    if "proposal" not in response or "id" not in response["proposal"]:
        print(Fore.RED + f"❌ Proposal response missing 'proposal' or 'id': {response}")
        return None
    if distance_from_center(previous_price, CENTERLINE_POSITION) > MAX_CENTER_DRIFT:
        print(Fore.YELLOW + "⚠️ Centerline drift too high. Skipping proposal.")
        return None
    if (volatility := calculate_volatility(recent_prices)) > 0.22:
        print(Fore.YELLOW + f"⚠️ High volatility detected: {volatility:.5f}. Skipping proposal.")
        return None
    if hit_daily_profit_target():
        print(Fore.YELLOW + "🏁 Daily profit target hit. Stopping trades.")
        return
    if not is_within_trading_session():
        print(Fore.LIGHTBLACK_EX + "⛔ Outside trading session. Skipping.")
        return
    return response["proposal"]["id"]

async def wait_for_contract_result(ws, contract_id):
    global wins, losses, profit_total, trade_log, recent_prices

    print(Fore.YELLOW + "⏳ Waiting for contract result (live updates)...")
    last_status = None

    while True:
        await ws.send(json.dumps({
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1
        }))

        async for msg in ws:
            response = json.loads(msg)

            if "error" in response:
                print(Fore.RED + f"❌ Contract Check Error: {response['error']['message']}")
                return

            if response.get("msg_type") == "proposal_open_contract":
                contract = response["proposal_open_contract"]
                status = contract.get("status")
                profit = float(contract.get("profit", 0.0))

                if status != last_status:
                    print(Fore.CYAN + f"Live Status: {status} | Profit: {profit:.2f}")
                    last_status = status

                if contract.get("is_expired") or contract.get("is_sold"):
                    buy_price = contract.get("buy_price", 0.0)
                    pnl = profit
                    profit_total += pnl

                    epsilon = 1e-6
                    if pnl > epsilon:
                        wins += 1
                        result = "WIN"
                        color = Fore.GREEN
                    elif pnl < -epsilon:
                        losses += 1
                        result = "LOSS"
                        color = Fore.RED
                    else:
                        result = "BREAKEVEN"
                        color = Fore.LIGHTYELLOW_EX

                    velocity = recent_prices[-1] - recent_prices[-2] if len(recent_prices) >= 2 else 0.0
                    volatility = np.std(recent_prices[-10:]) if len(recent_prices) >= 10 else 0.0
                    profit_10 = sum([t['pnl'] for t in trade_log[-10:]]) if len(trade_log) >= 10 else 0.0

                    trade_features = {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "price": buy_price,
                        "volatility_10": round(volatility, 5),
                        "velocity": round(velocity, 5),
                        "profit_10": round(profit_10, 5),
                        "result": result,
                        "pnl": round(pnl, 5)
                    }

                    trade_log.append(trade_features)

                    print(
                        color +
                        f"{result} | Profit: {pnl:.2f} | "
                        f"Velocity: {velocity:.5f} | "
                        f"Volatility: {volatility:.5f} | "
                        f"Profit 10: {profit_10:.5f} | "
                        f"Time: {trade_features['time']} | "
                        f"Price: {buy_price:.5f} | "
                        f"Distance: {distance_from_center(buy_price, CENTERLINE_POSITION):.5f} | "
                        f"Status: {status} | "
                        f"Recent Prices: {recent_prices[-5:]}"
                    )
                    return

        await asyncio.sleep(3)

async def send_buy_request(ws, proposal_id, trading_system):
    global wins, losses, profit_total
    await ws.send(json.dumps({
        "buy": proposal_id,
        "price": current_bet
    }))

    # Wait for a response with msg_type == "buy"
    while True:
        response = json.loads(await ws.recv())
        if response.get("msg_type") == "buy":
            break
        if "error" in response:
            print(Fore.RED + f"❌ Buy Error: {response['error']['message']}")
            return False
        # Ignore unrelated messages (e.g., ticks)
        # Optionally, add a timeout or max attempts to avoid infinite loop
    
    if distance_from_center(previous_price, CENTERLINE_POSITION) > MAX_CENTER_DRIFT:
        print(Fore.YELLOW + "⚠️ Centerline drift too high. Skipping buy.")
        return False
    if "error" in response:
        print(Fore.RED + f"❌ Buy Error: {response['error']['message']}")
        return False
    if "buy" not in response or "contract_id" not in response["buy"]:
        print(Fore.RED + f"❌ Unexpected buy response: {response}")
        return False
    contract_id = response["buy"]["contract_id"]
    await wait_for_contract_result(ws, contract_id)
    accuracy = trading_system.evaluate_model(trading_system.ai_model, trading_system.data_collector.get_dataframe())
    volatility = calculate_volatility(recent_prices)
    dashboard_data["Price"] = f"{previous_price:.5f}"
    dashboard_data["Centerline"] = f"{CENTERLINE_POSITION:.5f}" if CENTERLINE_POSITION is not None else "-"
    dashboard_data["Distance"] = f"{distance_from_center(previous_price, CENTERLINE_POSITION):.5f}"
    dashboard_data["Signal"] = "✅ TRADE EXECUTED"      
    dashboard_data["Cooldown"] = f"{max(0, COOLDOWN_SECONDS - (time.time() - last_trade_time)):.1f}s"
    dashboard_data["Volatility"] = f"{volatility:.5f}"
    dashboard_data["Accuracy"] = f"{accuracy:.2%}"
    dashboard_data["Trade Count"] = trade_count 
    dashboard_data["Daily Profit"] = f"${daily_profit:.2f}"
    dashboard_data["Max Bet"] = f"${current_bet:.2f}"
    
    dashboard_data["Wins"] = wins
    dashboard_data["Losses"] = losses
    dashboard_data["Profit"] = f"${profit_total:.2f}"
    dashboard_data["Balance"] = f"${account_balance + profit_total:.2f}"
    auto_tune_parameters()
    return True

async def accumulator_barrier_strategy():
    global previous_price, previous_upper, previous_lower
    global last_trade_time, CENTERLINE_POSITION, trade_count, BARRIER_DELTA

    trading_system = HybridTradingSystem()
    async with websockets.connect(WS_URL) as ws:
        if not await send_authorization(ws):
            return
        
        await ws.send(json.dumps({ "ticks": SYMBOL, "subscribe": 1 }))
        with Live(build_dashboard(), refresh_per_second=2) as live:
            async for msg in ws:
                data = json.loads(msg)
                if data.get("msg_type") != "tick":
                    continue
                current_price = float(data["tick"]["quote"])
                current_upper, current_lower = calculate_barriers(current_price, BARRIER_DELTA)

                if previous_lower is None or previous_upper is None:
                    previous_lower = current_lower
                    previous_upper = current_upper
                    previous_price = current_price
                    continue

                CENTERLINE_POSITION = update_centerline(current_price)
                center_distance = distance_from_center(current_price, CENTERLINE_POSITION)
                cooldown_remaining = max(0, COOLDOWN_SECONDS - (time.time() - last_trade_time))
                recent_prices.append(current_price)
                if len(recent_prices) > 20:
                    recent_prices.pop(0)
                volatility = calculate_volatility(recent_prices)
                if volatility > 0.22:
                    print(Fore.YELLOW + f"⚠️ High volatility detected: {volatility:.5f}")

                dashboard_data["Price"] = f"{current_price:.5f}"
                dashboard_data["Centerline"] = f"{CENTERLINE_POSITION:.5f}" if CENTERLINE_POSITION is not None else "-"
                dashboard_data["Distance"] = f"{center_distance:.5f}"
                dashboard_data["Cooldown"] = f"{cooldown_remaining:.1f}s"

                heuristic_decision = (
                    is_within_range(current_price, current_lower, current_upper) and
                    is_within_range(current_price, previous_lower, previous_upper) and
                    (CENTER_TOLERANCE_MIN <= center_distance <= CENTER_TOLERANCE_MAX) and
                    (time.time() - last_trade_time >= COOLDOWN_SECONDS)
                )

                final_decision = trading_system.decide_trade(current_price, 1, heuristic_decision)

                should_trade = True
                confidence = None

                # Evaluate all trade-blocking conditions
                if not final_decision:
                    should_trade = False

                if trading_system.ai_model is not None:
                    features = {
                        "volatility_10": np.std(recent_prices[-10:]) if len(recent_prices) >= 10 else 0.0,
                        "velocity": recent_prices[-1] - recent_prices[-2] if len(recent_prices) >= 2 else 0.0,
                        "volume": 1,
                        "profit_10": sum([t['pnl'] for t in trade_log[-10:]]) if len(trade_log) >= 10 else 0.0
                    }
                    _, confidence = trading_system.ai_filter(features)
                    if confidence is not None and confidence < 0.7:
                        print(Fore.YELLOW + f"⚠️ Low AI confidence in proposal: {confidence:.2f}. Skipping.")
                        should_trade = False
                    if confidence is not None:
                        print(f"🤖 AI Confidence: {confidence:.2%}")

                if distance_from_center(current_price, CENTERLINE_POSITION) > 0.4:
                    print(Fore.YELLOW + "⚠️ Centerline drift too high. Skipping proposal.")
                    should_trade = False

                if previous_price is not None and distance_from_center(previous_price, CENTERLINE_POSITION) > MAX_CENTER_DRIFT:
                    print(Fore.YELLOW + "⚠️ Centerline drift too high. Skipping proposal.")
                    should_trade = False
                    # Network quality and signal check before trading
                async def check_network_quality():
                    try:
                        # Try a quick ping to Deriv API (authorize call)
                        async with websockets.connect(WS_URL, ping_timeout=2, close_timeout=2) as test_ws:
                            await test_ws.send(json.dumps({"ping": 1}))
                            pong = await asyncio.wait_for(test_ws.recv(), timeout=2)
                            pong_data = json.loads(pong)
                            return pong_data.get("msg_type") == "ping"
                    except Exception as e:
                        print(Fore.RED + f"❌ Network check failed: {e}")
                        return False
                network_ok = await check_network_quality()
                if not network_ok:
                    print(Fore.RED + "❌ Poor network connectivity detected. Skipping proposal to avoid losses.")
                    should_trade = False

                volatility = calculate_volatility(recent_prices)
                if volatility > 0.22:
                    print(Fore.YELLOW + f"⚠️ High volatility detected: {volatility:.5f}. Skipping proposal.")
                    should_trade = False

                if hit_daily_profit_target():
                    print(Fore.YELLOW + "🏁 Daily profit target hit. Stopping trades.")
                    should_trade = False

                if not is_within_trading_session():
                    print(Fore.LIGHTBLACK_EX + "⛔ Outside trading session. Skipping.")
                    should_trade = False

                if not should_trade:
                    dashboard_data["Signal"] = "❌ No Signal"
                else:
                    dashboard_data["Signal"] = "✅ TRADE SIGNAL"
                    proposal_id = await send_proposal(ws)
                    if proposal_id and await send_buy_request(ws, proposal_id, trading_system):
                        last_trade_time = time.time()
                        trade_count += 1

                previous_price = current_price
                previous_upper = current_upper
                previous_lower = current_lower
                live.update(build_dashboard())

                if trading_system.ai_ready and trading_system.trade_count % 500 == 0:
                    df = trading_system.data_collector.get_dataframe()
                    accuracy = trading_system.evaluate_model(trading_system.ai_model, df)
                    if accuracy < 0.6:
                        trading_system.ai_model = trading_system.train_model(df)

if __name__ == "__main__":
    trading_system = HybridTradingSystem()
    df = trading_system.data_collector.get_dataframe()
    accuracy = 1.0  # Default to 1.0 if not enough data
    if not df.empty and trading_system.ai_model is not None:
        accuracy = trading_system.evaluate_model(trading_system.ai_model, df)
    if not df.empty and (trading_system.ai_model is None or accuracy < 0.6):
        trading_system.ai_model = trading_system.train_model(df)
    # if not is_within_trading_session():
    #     print(Fore.RED + "⛔ Outside trading session. Exiting.")
    #     exit(0)

    if CENTERLINE_POSITION is None:
        print(Fore.YELLOW + "🔄 Initializing centerline position...")
        CENTERLINE_POSITION = previous_price
        centerline_prices.clear()

    volatility = calculate_volatility(recent_prices)
    if volatility > 0.22:
        print(Fore.YELLOW + f"⚠️ High volatility detected: {volatility:.5f}. Adjusting parameters.")
        BARRIER_DELTA = 0.32
        CENTER_TOLERANCE_MAX = 0.32

    print(Fore.MAGENTA + "📡 Running AI-Augmented Accumulator Strategy...")

    account_balance = asyncio.run(get_account_balance())
    current_bet = get_max_bet(account_balance)

    try:
        asyncio.run(accumulator_barrier_strategy())
    except KeyboardInterrupt:
        print(Fore.RED + "\n🛑 Stopped by user.")
    except Exception as e:
        print(Fore.RED + f"❌ Unexpected error: {e}")

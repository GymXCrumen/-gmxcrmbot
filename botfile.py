#!/usr/bin/env python3
"""
XGBOOST AI Trading Bot for Telegram
Deployed on Railway
"""

import os
import asyncio
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import csv
import logging
import sys
from datetime import datetime

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ================= CONFIG =================

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
DATA_FILE = "trades.csv"
MIN_TRADES_FOR_TRAINING = 50
CONFIDENCE_THRESHOLD = 0.65
SCAN_INTERVAL = 10

# ================= DATA =================

def get_data(symbol, tf="1m"):
    """Fetch historical data for a symbol"""
    mapping = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
        "AUDUSD": "AUDUSD=X"
    }
    
    try:
        df = yf.download(mapping[symbol], interval=tf, period="1d", progress=False)
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        df.columns = df.columns.str.lower()
        return df.tail(120)
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# ================= SESSION =================

def in_session():
    """Check if current time is within trading sessions"""
    hour = datetime.utcnow().hour
    return (7 <= hour <= 10) or (12 <= hour <= 16)

def good_entry_time():
    """Check if entry time is optimal"""
    return datetime.utcnow().second >= 50

# ================= AI ENGINE =================

class LearningEngine:
    """Machine learning engine"""
    
    def __init__(self):
        self.initialize_csv()
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42
        )
        
        self.trained = False
        self.training_samples = 0
        self.train()
    
    def initialize_csv(self):
        """Initialize CSV file if it doesn't exist"""
        if not os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "trend", "liquidity", "rejection",
                        "momentum", "rsi", "result", "timestamp"
                    ])
                logger.info(f"Created new trade log file: {DATA_FILE}")
            except Exception as e:
                logger.error(f"Error creating CSV file: {e}")
    
    def save_trade(self, features, result):
        """Save trade outcome"""
        try:
            timestamp = datetime.utcnow().isoformat()
            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    int(features["trend"]),
                    int(features["liquidity"]),
                    int(features["rejection"]),
                    int(features["momentum"]),
                    int(features["rsi"]),
                    result,
                    timestamp
                ])
            logger.info(f"Trade saved - Result: {'WIN' if result else 'LOSS'}")
            return True
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return False
    
    def train(self):
        """Train the model"""
        try:
            if not os.path.exists(DATA_FILE):
                logger.info("No trade data available for training")
                return
            
            df = pd.read_csv(DATA_FILE)
            
            if len(df) < MIN_TRADES_FOR_TRAINING:
                logger.info(f"Not enough data for training. Need {MIN_TRADES_FOR_TRAINING} samples, have {len(df)}")
                return
            
            feature_cols = ["trend", "liquidity", "rejection", "momentum", "rsi"]
            X = df[feature_cols]
            y = df["result"]
            
            self.model.fit(X, y)
            self.trained = True
            self.training_samples = len(df)
            
            predictions = self.model.predict(X)
            accuracy = (predictions == y).mean()
            
            logger.info(f"Model trained on {len(df)} samples. Accuracy: {accuracy:.2%}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def predict(self, features):
        """Predict probability"""
        if not self.trained:
            return 0.5
        
        try:
            X = [[
                int(features["trend"]),
                int(features["liquidity"]),
                int(features["rejection"]),
                int(features["momentum"]),
                int(features["rsi"])
            ]]
            
            proba = self.model.predict_proba(X)[0][1]
            return float(proba)
        except Exception as e:
            logger.error(f"Error predicting: {e}")
            return 0.5

# ================= STRATEGY =================

class Strategy:
    """Trading strategy"""
    
    def __init__(self, learner):
        self.learner = learner
    
    def liquidity_grab(self, df):
        """Detect liquidity grab"""
        try:
            if len(df) < 10:
                return False
            lows = df['low'].rolling(10).min()
            last = df.iloc[-1]
            return last['low'] < lows.iloc[-2] and last['close'] > lows.iloc[-2]
        except:
            return False
    
    def rejection(self, df):
        """Detect rejection"""
        try:
            if len(df) < 1:
                return False
            last = df.iloc[-1]
            body = abs(last['close'] - last['open'])
            wick = min(last['open'], last['close']) - last['low']
            return body > 0 and wick > body * 3
        except:
            return False
    
    def momentum(self, df):
        """Check momentum"""
        try:
            if len(df) < 2:
                return False
            return df.iloc[-1]['close'] > df.iloc[-2]['close']
        except:
            return False
    
    def analyze(self, symbol):
        """Analyze symbol"""
        try:
            df = get_data(symbol)
            
            if df.empty or len(df) < 200:
                return None
            
            df['ema'] = ta.ema(df['close'], length=200)
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            last = df.iloc[-1]
            
            if pd.isna(last['ema']) or pd.isna(last['rsi']):
                return None
            
            features = {
                "trend": last['close'] > last['ema'],
                "liquidity": self.liquidity_grab(df),
                "rejection": self.rejection(df),
                "momentum": self.momentum(df),
                "rsi": last['rsi'] < 35
            }
            
            score = sum(features.values())
            if score < 4:
                return None
            
            probability = self.learner.predict(features)
            
            if probability < CONFIDENCE_THRESHOLD:
                return None
            
            return {
                "symbol": symbol,
                "price": last['close'],
                "direction": "CALL",
                "features": features,
                "probability": probability,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

# ================= RESULT TRACKING =================

async def track_result(symbol, entry_price, direction, features, learner):
    """Track trade result"""
    await asyncio.sleep(30)
    
    try:
        df = get_data(symbol)
        if df.empty:
            logger.error(f"Could not fetch data for tracking {symbol}")
            return
        
        last = df.iloc[-1]
        result = 1 if last['close'] > entry_price else 0
        
        learner.save_trade(features, result)
        learner.train()
        
        result_text = "WIN" if result else "LOSS"
        logger.info(f"Tracked {symbol}: {result_text}")
        
    except Exception as e:
        logger.error(f"Error tracking result: {e}")

# ================= EXECUTION HOOK =================

def execute_trade_stub(signal):
    """Execute trade (placeholder)"""
    logger.info(f"🔔 SIGNAL: {signal['symbol']} @ {signal['price']:.5f} | Confidence: {signal['probability']*100:.1f}%")
    
    try:
        with open("signals.txt", "a") as f:
            f.write(f"{signal['timestamp']} | {signal['symbol']} | {signal['direction']} | {signal['price']:.5f} | {signal['probability']*100:.1f}%\n")
    except Exception as e:
        logger.error(f"Error writing signal: {e}")

# ================= TELEGRAM BOT =================

class SniperBot:
    """Main bot class"""
    
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set!")
        
        logger.info("Bot token loaded successfully")
        self.learner = LearningEngine()
        self.strategy = Strategy(self.learner)
        self.running = False
        self.update = None
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command"""
        welcome_msg = (
            "🤖 *XGBOOST AI TRADING BOT*\n\n"
            "I analyze forex pairs using machine learning.\n\n"
            "*Commands:*\n"
            "🔍 `/scan` - Start scanning\n"
            "🛑 `/stop` - Stop scanning\n"
            "📊 `/status` - Bot status\n"
            "📈 `/stats` - View statistics\n\n"
            "*Active Pairs:* " + ", ".join(PAIRS)
        )
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')
        logger.info(f"User {update.effective_user.id} started the bot")
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Status command"""
        status_text = f"📊 *Bot Status*\n\n"
        status_text += f"*Scanning:* {'🟢 Active' if self.running else '🔴 Inactive'}\n"
        status_text += f"*Model Trained:* {'✅ Yes' if self.learner.trained else '❌ No'}\n"
        status_text += f"*Training Samples:* {self.learner.training_samples}\n"
        status_text += f"*Trading Session:* {'🟢 Open' if in_session() else '🔴 Closed'}"
        
        await update.message.reply_text(status_text, parse_mode='Markdown')
    
    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Statistics command"""
        try:
            if not os.path.exists(DATA_FILE):
                await update.message.reply_text("No trade data available yet.")
                return
            
            df = pd.read_csv(DATA_FILE)
            if len(df) == 0
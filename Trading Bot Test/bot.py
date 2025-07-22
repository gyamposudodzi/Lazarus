import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
import json
import os
import threading
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import yfinance as yf
import sqlite3
from pathlib import Path

class OrderType(Enum):
    BUY = "SELL"
    SELL = "BUY"
    BUY_LIMIT = "SELL_LIMIT"
    SELL_LIMIT = "BUY_LIMIT"
    BUY_STOP = "SELL_STOP"
    SELL_STOP = "BUY_STOP"

class TradingState(Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"

@dataclass
class TradeSignal:
    symbol: str
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    reason: str
    timestamp: datetime
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Position:
    ticket: int
    symbol: str
    type: str
    volume: float
    open_price: float
    stop_loss: float
    take_profit: float
    profit: float
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class DatabaseManager:
    """Handles database operations for trade history and logs"""
    
    def __init__(self, db_path: str = "mt5_trading.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    type TEXT NOT NULL,
                    volume REAL NOT NULL,
                    open_price REAL NOT NULL,
                    close_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    profit REAL DEFAULT 0,
                    profit_pips REAL DEFAULT 0,
                    open_time TIMESTAMP NOT NULL,
                    close_time TIMESTAMP,
                    reason TEXT,
                    status TEXT DEFAULT 'OPEN',
                    ticket INTEGER,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Daily stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    profit REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_trade(self, trade_data: Dict):
        """Save trade to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (symbol, type, volume, open_price, close_price, 
                                  stop_loss, take_profit, profit, profit_pips, 
                                  open_time, close_time, reason, status, ticket, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['symbol'], trade_data['type'], trade_data['volume'],
                trade_data['open_price'], trade_data.get('close_price'),
                trade_data['stop_loss'], trade_data['take_profit'],
                trade_data['profit'], trade_data.get('profit_pips', 0),
                trade_data['open_time'], trade_data.get('close_time'),
                trade_data['reason'], trade_data.get('status', 'OPEN'),
                trade_data.get('ticket'), trade_data.get('confidence', 0)
            ))
            conn.commit()
    
    def update_daily_stats(self, date: str, stats: Dict):
        """Update daily statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO daily_stats 
                (date, trades, wins, losses, profit, win_rate, max_drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                date, stats['trades'], stats['wins'], stats['losses'],
                stats['profit'], stats['win_rate'], stats.get('max_drawdown', 0)
            ))
            conn.commit()

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.daily_loss = 0
        self.daily_trades = 0
        self.max_drawdown = 0
        self.peak_equity = 0
        
    def can_trade(self, account_balance: float, open_positions: int) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk parameters"""
        
        # Check daily loss limit
        if abs(self.daily_loss) >= self.config['max_daily_loss']:
            return False, "Daily loss limit exceeded"
        
        # Check maximum trades per day
        max_daily_trades = self.config.get('max_daily_trades', 20)
        if self.daily_trades >= max_daily_trades:
            return False, "Daily trade limit exceeded"
        
        # Check maximum concurrent positions
        if open_positions >= self.config['max_concurrent_trades']:
            return False, "Maximum concurrent positions reached"
        
        # Check account balance
        min_balance = self.config.get('min_balance', 1000)
        if account_balance < min_balance:
            return False, "Account balance below minimum threshold"
        
        return True, "OK"
    
    def calculate_position_size(self, symbol: str, account_balance: float, 
                              risk_amount: float, stop_loss_pips: float,
                              symbol_info: Dict) -> float:
        """Calculate optimal position size based on risk management"""
        
        # Risk amount in currency
        risk_money = account_balance * risk_amount
        
        # Get pip value
        pip_value = symbol_info.get('pip_value', 10)
        contract_size = symbol_info.get('contract_size', 100000)
        
        # Calculate lot size
        if stop_loss_pips > 0:
            lot_size = risk_money / (stop_loss_pips * pip_value)
        else:
            lot_size = symbol_info['min_lot']
        
        # Apply limits
        min_lot = symbol_info['min_lot']
        max_lot = symbol_info['max_lot']
        lot_step = symbol_info.get('lot_step', 0.01)
        
        # Round to lot step
        lot_size = max(min_lot, min(lot_size, max_lot))
        lot_size = round(lot_size / lot_step) * lot_step
        
        return lot_size
    
    def update_drawdown(self, current_equity: float):
        """Update maximum drawdown tracking"""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

class MT5TradingBot:
    def __init__(self, config_file: str = 'mt5_config.json'):
        self.config_file = config_file
        self.load_config()
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.risk_manager = RiskManager(self.config['risk'])
        
        # Trading state
        self.state = TradingState.STOPPED
        self.is_connected = False
        self.positions: Dict[str, Position] = {}
        self.trade_history = []
        self.daily_stats = {'trades': 0, 'profit': 0, 'wins': 0, 'losses': 0}
        
        # Threading
        self.trading_thread = None
        self.stop_event = threading.Event()
        
        # Symbol mapping (Yahoo Finance to MT5)
        self.symbol_map = {
            'EURUSD=X': 'EURUSD',
            'GBPUSD=X': 'GBPUSD',
            'USDJPY=X': 'USDJPY',
            'AUDUSD=X': 'AUDUSD',
            'USDCAD=X': 'USDCAD',
            'USDCHF=X': 'USDCHF',
            'NZDUSD=X': 'NZDUSD',
            'XAUUSD=X': 'XAUUSD'
        }
        
        # Setup logging
        self.setup_logging()
        
        # Load existing positions
        self.load_positions()
    
    def load_config(self):
        """Load configuration with better error handling"""
        default_config = {
            'mt5': {
                'login': 5038366265,
                'password': "-0NfSdPi",
                'server': "MetaQuotes-Demo",
                'path': r"C:\Program Files\MetaTrader 5\terminal64.exe"
            },
            'risk': {
                'max_daily_loss': 1000,
                'max_daily_trades': 100,
                'max_concurrent_trades': 30,
                'risk_per_trade': 0.05,
                'min_confidence': 0,
                'max_spread': 3.0,
                'slippage': 3,
                'min_balance': 0
            },
            'trading': {
                'magic_number': 123456,
                'comment': "AutoBot_v1",
                'enable_partial_close': True,
                'trailing_stop': True,
                'break_even': True,
                'break_even_pips': 20,
                'break_even_offset': 5
            },
            'symbols': {
                'EURUSD': {'min_lot': 0.01, 'max_lot': 1.0, 'pip_value': 10},
                'GBPUSD': {'min_lot': 0.01, 'max_lot': 1.0, 'pip_value': 10},
                'USDJPY': {'min_lot': 0.01, 'max_lot': 1.0, 'pip_value': 9.09},
                'AUDUSD': {'min_lot': 0.01, 'max_lot': 1.0, 'pip_value': 10},
                'USDCAD': {'min_lot': 0.01, 'max_lot': 1.0, 'pip_value': 7.69},
                'USDCHF': {'min_lot': 0.01, 'max_lot': 1.0, 'pip_value': 10},
                'NZDUSD': {'min_lot': 0.01, 'max_lot': 1.0, 'pip_value': 10},
                'XAUUSD': {'min_lot': 0.01, 'max_lot': 1.0, 'pip_value': 1}
            },
            'notifications': {
                'email_enabled': False,
                'email_address': "",
                'webhook_url': ""
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                self.config = self.merge_configs(default_config, loaded_config)
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            print(f" Error loading config: {e}")
            self.config = default_config
    
    def merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge configuration dictionaries"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('MT5TradingBot')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"mt5_trading_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def connect_mt5(self) -> bool:
        """Connect to MT5 with better error handling"""
        try:
            # Validate config
            if not self.config['mt5']['login'] or not self.config['mt5']['password']:
                self.logger.error("MT5 credentials not configured")
                return False
            
            # Initialize MT5
            if not mt5.initialize(path=self.config['mt5']['path']):
                error = mt5.last_error()
                self.logger.error(f"MT5 initialize failed: {error}")
                return False
            
            # Login
            if not mt5.login(
                login=self.config['mt5']['login'],
                password=self.config['mt5']['password'],
                server=self.config['mt5']['server']
            ):
                error = mt5.last_error()
                self.logger.error(f"MT5 login failed: {error}")
                return False
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                return False
            
            self.logger.info(f" Connected to MT5")
            self.logger.info(f"Account: {account_info.login}")
            self.logger.info(f"Balance: {account_info.balance} {account_info.currency}")
            self.logger.info(f"Server: {account_info.server}")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            self.state = TradingState.ERROR
            return False
    
    def disconnect_mt5(self):
        """Safely disconnect from MT5"""
        try:
            if self.is_connected:
                mt5.shutdown()
                self.is_connected = False
                self.logger.info("Disconnected from MT5")
        except Exception as e:
            self.logger.error(f"Error disconnecting from MT5: {e}")
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive symbol information"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            # Get tick info for spread
            tick = mt5.symbol_info_tick(symbol)
            spread = (tick.ask - tick.bid) if tick else 0
            
            return {
                'symbol': symbol,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'spread': spread,
                'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max,
                'lot_step': symbol_info.volume_step,
                'contract_size': symbol_info.trade_contract_size,
                'pip_value': self.config['symbols'].get(symbol, {}).get('pip_value', 10),
                'margin_required': symbol_info.margin_initial,
                'trade_allowed': symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED
            }
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current price with validation"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            # Validate prices
            if tick.bid <= 0 or tick.ask <= 0:
                return None
            
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'time': datetime.fromtimestamp(tick.time),
                'mid': (tick.bid + tick.ask) / 2
            }
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def validate_trade_conditions(self, symbol: str, signal: TradeSignal) -> Tuple[bool, str]:
        """Comprehensive trade validation"""
        try:
            # Check if trading is enabled
            if self.state != TradingState.RUNNING:
                return False, "Trading not active"
            
            # Check connection
            if not self.is_connected:
                return False, "Not connected to MT5"
            
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                return False, "Cannot get account info"
            
            # Risk management checks
            can_trade, risk_reason = self.risk_manager.can_trade(
                account_info.balance, len(self.positions)
            )
            if not can_trade:
                return False, risk_reason
            
            # Check if symbol already has position
            if symbol in self.positions:
                return False, f"{symbol} already has open position"
            
            # Check confidence level
            if signal.confidence < self.config['risk']['min_confidence']:
                return False, f"Confidence too low: {signal.confidence}%"
            
            # Check symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return False, f"Cannot get {symbol} info"
            
            if not symbol_info['trade_allowed']:
                return False, f"{symbol} trading not allowed"
            
            # Check spread
            if symbol_info['spread'] > 0:
                spread_pips = symbol_info['spread'] / symbol_info['point'] / 10
                if spread_pips > self.config['risk']['max_spread']:
                    return False, f"Spread too high: {spread_pips:.1f} pips"
            
            # Check market hours
            if not self.is_market_open():
                return False, "Market closed"
            
            # Check margin requirements
            required_margin = symbol_info['margin_required'] * signal.lot_size
            if required_margin > account_info.margin_free:
                return False, "Insufficient margin"
            
            return True, "OK"
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"
    
    def is_market_open(self) -> bool:
        """Check if forex market is open (improved)"""
        try:
            # Get current GMT time
            now_utc = datetime.now(timezone.utc)
            weekday = now_utc.weekday()
            hour = now_utc.hour
            
            # Forex market hours (GMT):
            # Closed: Friday 22:00 - Sunday 22:00
            if weekday == 4 and hour >= 22:  # Friday 22:00+
                return False
            elif weekday == 5:  # Saturday (all day)
                return False
            elif weekday == 6 and hour < 22:  # Sunday before 22:00
                return False
            
            # Check if specific symbols have trading sessions
            # This is a simplified check - you might want to add more specific session times
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            return True  # Default to open if we can't determine
    
    def execute_trade(self, signal: TradeSignal) -> Optional[int]:
        """Execute trade with comprehensive error handling"""
        try:
            symbol = signal.symbol
            
            # Validate trade
            is_valid, reason = self.validate_trade_conditions(symbol, signal)
            if not is_valid:
                self.logger.warning(f" Trade validation failed for {symbol}: {reason}")
                return None
            
            # Get current price
            price_info = self.get_current_price(symbol)
            if not price_info:
                self.logger.error(f" Cannot get price for {symbol}")
                return None
            
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f" Cannot get symbol info for {symbol}")
                return None
            
            # Calculate position size
            account_info = mt5.account_info()
            stop_loss_pips = abs(signal.entry_price - signal.stop_loss) / symbol_info['point'] / 10
            
            lot_size = self.risk_manager.calculate_position_size(
                symbol, account_info.balance, self.config['risk']['risk_per_trade'],
                stop_loss_pips, symbol_info
            )
            
            # Determine order type and price
            if signal.action.upper() == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = price_info['ask']
            elif signal.action.upper() == "SELL":
                order_type = mt5.ORDER_TYPE_SELL
                price = price_info['bid']
            else:
                self.logger.error(f"Invalid signal action: {signal.action}")
                return None
            
            # Prepare order request
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': lot_size,
                'type': order_type,
                'price': price,
                'sl': signal.stop_loss,
                'tp': signal.take_profit,
                'deviation': self.config['risk']['slippage'],
                'magic': self.config['trading']['magic_number'],
                'comment': 'AutoTrade for Dodzi',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_FOK,
            }
            
            # Log order attempt
            self.logger.info(f"Placing {signal.action} order for {symbol}")
            self.logger.info(f"Size: {lot_size} | Price: {price}")
            self.logger.info(f"SL: {signal.stop_loss} | TP: {signal.take_profit}")
            
            # Send order
            result = mt5.order_send(request)

# Check if result is None (failed trade send)
            if result is None:
                error_code, description = mt5.last_error()
                self.logger.error("Order send failed: result is None")
                self.logger.error(f"MT5 Error [{error_code}]: {description}")
                self.logger.debug(f"Request: {request}")
                return None

            # Check result retcode
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f" Order failed: {result.comment} (Code: {result.retcode})")
                return None
            
            # Create position object
            position = Position(
                ticket=result.order,
                symbol=symbol,
                type=signal.action.upper(),
                volume=lot_size,
                open_price=result.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                profit=0,
                timestamp=datetime.now()
            )
            
            # Store position
            self.positions[symbol] = position
            
            # Update statistics
            self.daily_stats['trades'] += 1
            self.risk_manager.daily_trades += 1
            
            # Save to database
            trade_data = {
                'symbol': symbol,
                'type': signal.action.upper(),
                'volume': lot_size,
                'open_price': result.price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'profit': 0,
                'open_time': datetime.now().isoformat(),
                'reason': signal.reason,
                'status': 'OPEN',
                'ticket': result.order,
                'confidence': signal.confidence
            }
            self.db_manager.save_trade(trade_data)
            
            self.logger.info(f" {signal.action} {symbol} executed successfully")
            self.logger.info(f"   Ticket: {result.order} | Actual Price: {result.price}")
            
            return result.order
            
        except Exception as e:
            self.logger.error(f" Error executing trade: {e}")
            return None
    
    def close_position(self, symbol: str, reason: str = "Manual close") -> bool:
        """Close position with comprehensive error handling"""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            # Get current price
            price_info = self.get_current_price(symbol)
            if not price_info:
                self.logger.error(f"Cannot get price for {symbol}")
                return False
            
            # Determine close price and order type
            if position.type == "BUY":
                close_price = price_info['bid']
                order_type = mt5.ORDER_TYPE_SELL
            else:
                close_price = price_info['ask']
                order_type = mt5.ORDER_TYPE_BUY
            
            # Prepare close request
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': position.volume,
                'type': order_type,
                'position': position.ticket,
                'price': close_price,
                'deviation': self.config['risk']['slippage'],
                'magic': self.config['trading']['magic_number'],
                'comment': f"Close - {reason[:20]}",
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Close failed for {symbol}: {result.comment}")
                return False
            
            # Calculate profit in pips
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                if position.type == "BUY":
                    profit_pips = (close_price - position.open_price) / symbol_info['point'] / 10
                else:
                    profit_pips = (position.open_price - close_price) / symbol_info['point'] / 10
            else:
                profit_pips = 0
            
            # Update statistics
            self.daily_stats['profit'] += result.profit
            self.risk_manager.daily_loss += result.profit
            
            if result.profit > 0:
                self.daily_stats['wins'] += 1
            else:
                self.daily_stats['losses'] += 1
            
            # Add to trade history
            trade_record = {
                'symbol': symbol,
                'type': position.type,
                'volume': position.volume,
                'open_price': position.open_price,
                'close_price': close_price,
                'profit': result.profit,
                'profit_pips': profit_pips,
                'open_time': position.timestamp,
                'close_time': datetime.now(),
                'reason': reason,
                'ticket': position.ticket
            }
            self.trade_history.append(trade_record)
            
            # Update database
            self.db_manager.save_trade({
                **trade_record,
                'open_time': position.timestamp.isoformat(),
                'close_time': datetime.now().isoformat(),
                'status': 'CLOSED'
            })
            
            # Remove from positions
            del self.positions[symbol]
            
            self.logger.info(f" Closed {symbol}")
            self.logger.info(f"   Profit: {result.profit:.2f} | Pips: {profit_pips:.1f}")
            self.logger.info(f"   Reason: {reason}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    def check_break_even(self, symbol: str, position: Position, mt5_pos):
        """Check and set break-even for profitable positions"""
        try:
            if not self.config['trading']['break_even']:
                return
            
            break_even_pips = self.config['trading']['break_even_pips']
            break_even_offset = self.config['trading']['break_even_offset']
            
            # Get symbol info for pip calculation
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return
            
            pip_size = symbol_info['point'] * 10
            current_price = self.get_current_price(symbol)
            if not current_price:
                return
            
            # Calculate current profit in pips
            if position.type == "BUY":
                profit_pips = (current_price['bid'] - position.open_price) / pip_size
                new_sl = position.open_price + (break_even_offset * pip_size)
            else:
                profit_pips = (position.open_price - current_price['ask']) / pip_size
                new_sl = position.open_price - (break_even_offset * pip_size)
            
            # Check if we should move to break-even
            if profit_pips >= break_even_pips and position.stop_loss != new_sl:
                # Only move SL if it's better than current SL
                should_update = False
                if position.type == "BUY" and new_sl > position.stop_loss:
                    should_update = True
                elif position.type == "SELL" and new_sl < position.stop_loss:
                    should_update = True
                
                if should_update:
                    self.modify_position(symbol, stop_loss=new_sl, reason="Break-even")
                    
        except Exception as e:
            self.logger.error(f"Error checking break-even for {symbol}: {e}")
    
    def modify_position(self, symbol: str, stop_loss: float = None, 
                       take_profit: float = None, reason: str = "Modification") -> bool:
        """Modify existing position"""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            
            # Prepare modification request
            request = {
                'action': mt5.TRADE_ACTION_SLTP,
                'symbol': symbol,
                'position': position.ticket,
                'sl': stop_loss if stop_loss is not None else position.stop_loss,
                'tp': take_profit if take_profit is not None else position.take_profit,
                'magic': self.config['trading']['magic_number'],
                'comment': f"Modify - {reason}"
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Update position object
                if stop_loss is not None:
                    position.stop_loss = stop_loss
                if take_profit is not None:
                    position.take_profit = take_profit
                
                self.logger.info(f"Modified {symbol} - {reason}")
                return True
            else:
                self.logger.error(f"Modify failed for {symbol}: {result.comment}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error modifying position {symbol}: {e}")
            return False
    
    def sync_positions(self):
        """Synchronize positions with MT5"""
        try:
            mt5_positions = mt5.positions_get()
            if mt5_positions is None:
                mt5_positions = []
            
            # Convert to dict for easier lookup
            mt5_pos_dict = {pos.symbol: pos for pos in mt5_positions}
            
            # Check for positions that exist in MT5 but not in our tracking
            for mt5_pos in mt5_positions:
                if mt5_pos.symbol not in self.positions:
                    # Add missing position
                    position = Position(
                        ticket=mt5_pos.ticket,
                        symbol=mt5_pos.symbol,
                        type="BUY" if mt5_pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                        volume=mt5_pos.volume,
                        open_price=mt5_pos.price_open,
                        stop_loss=mt5_pos.sl,
                        take_profit=mt5_pos.tp,
                        profit=mt5_pos.profit,
                        timestamp=datetime.fromtimestamp(mt5_pos.time)
                    )
                    self.positions[mt5_pos.symbol] = position
                    self.logger.info(f"Added missing position: {mt5_pos.symbol}")
            
            # Check for positions that exist in our tracking but not in MT5
            symbols_to_remove = []
            for symbol, position in self.positions.items():
                if symbol not in mt5_pos_dict:
                    symbols_to_remove.append(symbol)
            
            # Remove closed positions
            for symbol in symbols_to_remove:
                self.logger.info(f"Removing closed position: {symbol}")
                del self.positions[symbol]
            
            # Update profit for existing positions
            for symbol, position in self.positions.items():
                if symbol in mt5_pos_dict:
                    position.profit = mt5_pos_dict[symbol].profit
                    
                    # Check break-even
                    self.check_break_even(symbol, position, mt5_pos_dict[symbol])
                    
        except Exception as e:
            self.logger.error(f"Error syncing positions: {e}")
    
    def load_positions(self):
        """Load existing positions from MT5 on startup"""
        if self.is_connected:
            self.sync_positions()
    
    def get_market_data(self, symbol: str, timeframe: str = 'H1', periods: int = 100) -> Optional[pd.DataFrame]:
        """Get historical market data"""
        try:
            # Convert timeframe string to MT5 constant
            timeframes = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            tf = timeframes.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Get data
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, periods)
            if rates is None or len(rates) == 0:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def technical_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform technical analysis on price data"""
        try:
            if df is None or len(df) < 20:
                return {}
            
            analysis = {}
            
            # Moving Averages
            df['MA_20'] = df['close'].rolling(window=20).mean()
            df['MA_50'] = df['close'].rolling(window=50).mean()
            df['MA_200'] = df['close'].rolling(window=200).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            df['BB_Middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            analysis = {
                'price': latest['close'],
                'ma_20': latest['MA_20'],
                'ma_50': latest['MA_50'],
                'rsi': latest['RSI'],
                'macd': latest['MACD'],
                'macd_signal': latest['MACD_Signal'],
                'bb_upper': latest['BB_Upper'],
                'bb_lower': latest['BB_Lower'],
                'bb_middle': latest['BB_Middle'],
                'trend': self.determine_trend(df),
                'strength': self.calculate_signal_strength(df)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return {}
    
    def determine_trend(self, df: pd.DataFrame) -> str:
        """Determine market trend"""
        try:
            if len(df) < 50:
                return "UNKNOWN"
            
            latest = df.iloc[-1]
            
            # Check MA alignment
            ma_20 = latest['MA_20']
            ma_50 = latest['MA_50']
            price = latest['close']
            
            if price > ma_20 > ma_50:
                return "BULLISH"
            elif price < ma_20 < ma_50:
                return "BEARISH"
            else:
                return "SIDEWAYS"
                
        except Exception as e:
            return "UNKNOWN"
    
    def calculate_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength (0-100)"""
        try:
            if len(df) < 20:
                return 50.0
            
            latest = df.iloc[-1]
            strength = 50.0  # Base strength
            
            # RSI contribution
            rsi = latest['RSI']
            if rsi < 30:
                strength += 15  # Oversold - bullish
            elif rsi > 70:
                strength += 15  # Overbought - bearish
            
            # MACD contribution
            if latest['MACD'] > latest['MACD_Signal']:
                strength += 10
            
            # Bollinger Bands contribution
            if latest['close'] < latest['BB_Lower']:
                strength += 10  # Potential bounce
            elif latest['close'] > latest['BB_Upper']:
                strength += 10  # Potential reversal
            
            # Trend strength
            trend = self.determine_trend(df)
            if trend in ['BULLISH', 'BEARISH']:
                strength += 10
            
            return min(100, max(0, strength))
            
        except Exception as e:
            return 50.0
    
    def generate_trade_signal(self, symbol: str) -> Optional[TradeSignal]:
        """Generate trade signal based on technical analysis"""
        try:
            # Get market data
            df = self.get_market_data(symbol, 'H1', 200)
            if df is None:
                return None
            
            # Perform analysis
            analysis = self.technical_analysis(df)
            if not analysis:
                return None
            
            # Get current price
            price_info = self.get_current_price(symbol)
            if not price_info:
                return None
            
            # Signal generation logic
            signal = None
            confidence = analysis.get('strength', 50)
            reason = ""
            
            price = analysis['price']
            rsi = analysis.get('rsi', 50)
            macd = analysis.get('macd', 0)
            macd_signal = analysis.get('macd_signal', 0)
            bb_upper = analysis.get('bb_upper', price)
            bb_lower = analysis.get('bb_lower', price)
            trend = analysis.get('trend', 'SIDEWAYS')
            
            # Bullish signals
            if (rsi < 30 and price < bb_lower and macd > macd_signal and trend != 'BEARISH'):
                action = "BUY"
                entry_price = price_info['ask']
                stop_loss = price * 0.99  # 1% stop loss
                take_profit = price * 1.02  # 2% take profit
                reason = "RSI oversold + BB bounce + MACD bullish"
                confidence += 20
                
            # Bearish signals
            elif (rsi > 70 and price > bb_upper and macd < macd_signal and trend != 'BULLISH'):
                action = "SELL"
                entry_price = price_info['bid']
                stop_loss = price * 1.01  # 1% stop loss
                take_profit = price * 0.98  # 2% take profit
                reason = "RSI overbought + BB rejection + MACD bearish"
                confidence += 20
                
            # Trend following signals
            elif trend == 'BULLISH' and rsi < 60 and macd > macd_signal:
                action = "BUY"
                entry_price = price_info['ask']
                stop_loss = analysis.get('ma_20', price * 0.99)
                take_profit = price * 1.015  # 1.5% take profit
                reason = "Bullish trend continuation"
                confidence += 10
                
            elif trend == 'BEARISH' and rsi > 40 and macd < macd_signal:
                action = "SELL"
                entry_price = price_info['bid']
                stop_loss = analysis.get('ma_20', price * 1.01)
                take_profit = price * 0.985  # 1.5% take profit
                reason = "Bearish trend continuation"
                confidence += 10
            else:
                return None  # No signal
            
            # Create signal
            signal = TradeSignal(
                symbol=symbol,
                action=action,
                confidence=min(95, confidence),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=0.01,  # Will be recalculated in execute_trade
                reason=reason,
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def scan_markets(self) -> List[TradeSignal]:
        """Scan all configured symbols for trade opportunities"""
        signals = []
        
        try:
            symbols = list(self.config['symbols'].keys())
            
            for symbol in symbols:
                # Skip if we already have a position
                if symbol in self.positions:
                    continue
                
                # Generate signal
                signal = self.generate_trade_signal(symbol)
                if signal and signal.confidence >= self.config['risk']['min_confidence']:
                    signals.append(signal)
                    self.logger.info(f" Signal: {signal.action} {symbol} ({signal.confidence:.1f}%)")
                
                # Small delay to avoid overloading
                time.sleep(0.5)
                
        except Exception as e:
            self.logger.error(f"Error scanning markets: {e}")
        
        return signals
    
    def trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting trading loop")
        
        last_scan_time = 0
        scan_interval = 300  # 5 minutes
        
        while not self.stop_event.is_set() and self.state == TradingState.RUNNING:
            try:
                current_time = time.time()
                
                # Sync positions every loop
                self.sync_positions()
                
                # Update account info
                account_info = mt5.account_info()
                if account_info:
                    self.risk_manager.update_drawdown(account_info.equity)
                
                # Scan for new opportunities
                if current_time - last_scan_time >= scan_interval:
                    if self.is_market_open():
                        signals = self.scan_markets()
                        
                        # Execute signals
                        for signal in signals[:3]:  # Limit concurrent signals
                            if len(self.positions) < self.config['risk']['max_concurrent_trades']:
                                self.execute_trade(signal)
                            else:
                                break
                    
                    last_scan_time = current_time
                
                # Update daily stats
                today = datetime.now().strftime('%Y-%m-%d')
                win_rate = (self.daily_stats['wins'] / max(1, self.daily_stats['wins'] + self.daily_stats['losses'])) * 100
                
                stats = {
                    'trades': self.daily_stats['trades'],
                    'wins': self.daily_stats['wins'],
                    'losses': self.daily_stats['losses'],
                    'profit': self.daily_stats['profit'],
                    'win_rate': win_rate,
                    'max_drawdown': self.risk_manager.max_drawdown
                }
                self.db_manager.update_daily_stats(today, stats)
                
                # Sleep before next iteration
                time.sleep(30)  # 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait longer on error
        
        self.logger.info("Trading loop stopped")
    
    def start_trading(self):
        """Start the trading system"""
        try:
            if self.state == TradingState.RUNNING:
                self.logger.warning("Trading already running")
                return
            
            # Connect to MT5
            if not self.connect_mt5():
                return
            
            # Load existing positions
            self.load_positions()
            
            # Set state to running
            self.state = TradingState.RUNNING
            self.stop_event.clear()
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            self.trading_thread.start()
            
            self.logger.info("Trading system started")
            
        except Exception as e:
            self.logger.error(f"Error starting trading: {e}")
            self.state = TradingState.ERROR
    
    def stop_trading(self, close_positions: bool = False):
        """Stop the trading system"""
        try:
            if self.state != TradingState.RUNNING:
                self.logger.warning("Trading not running")
                return
            
            self.logger.info("Stopping trading system...")
            
            # Set stop event
            self.stop_event.set()
            self.state = TradingState.STOPPED
            
            # Close all positions if requested
            if close_positions:
                self.logger.info("Closing all positions...")
                for symbol in list(self.positions.keys()):
                    self.close_position(symbol, "System shutdown")
            
            # Wait for trading thread to finish
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Disconnect from MT5
            self.disconnect_mt5()
            
            self.logger.info("Trading system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
    
    def get_status(self) -> Dict:
        """Get current system status"""
        try:
            account_info = mt5.account_info() if self.is_connected else None
            
            status = {
                'state': self.state.value,
                'connected': self.is_connected,
                'positions': len(self.positions),
                'daily_stats': self.daily_stats.copy(),
                'account': {
                    'balance': account_info.balance if account_info else 0,
                    'equity': account_info.equity if account_info else 0,
                    'margin': account_info.margin if account_info else 0,
                    'free_margin': account_info.margin_free if account_info else 0,
                    'currency': account_info.currency if account_info else 'USD'
                },
                'risk': {
                    'daily_loss': self.risk_manager.daily_loss,
                    'daily_trades': self.risk_manager.daily_trades,
                    'max_drawdown': self.risk_manager.max_drawdown
                },
                'positions_detail': [pos.to_dict() for pos in self.positions.values()]
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {'state': 'ERROR', 'error': str(e)}
    
    def print_status(self):
        """Print formatted status"""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("MT5 TRADING BOT STATUS")
        print("="*60)
        print(f"State: {status['state']}")
        print(f"Connected: {'' if status['connected'] else ''}")
        print(f"Open Positions: {status['positions']}")
        
        print(f"\n ACCOUNT INFO:")
        acc = status['account']
        print(f"Balance: {acc['balance']:.2f} {acc['currency']}")
        print(f"Equity: {acc['equity']:.2f} {acc['currency']}")
        print(f"Free Margin: {acc['free_margin']:.2f} {acc['currency']}")
        
        print(f"\n DAILY STATS:")
        stats = status['daily_stats']
        print(f"Trades: {stats['trades']}")
        print(f"Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"Profit: {stats['profit']:.2f}")
        
        if stats['wins'] + stats['losses'] > 0:
            win_rate = stats['wins'] / (stats['wins'] + stats['losses']) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        
        print(f"\n  RISK INFO:")
        risk = status['risk']
        print(f"Daily Loss: {risk['daily_loss']:.2f}")
        print(f"Daily Trades: {risk['daily_trades']}")
        print(f"Max Drawdown: {risk['max_drawdown']:.2f}%")
        
        if status['positions_detail']:
            print(f"\n OPEN POSITIONS:")
            for pos in status['positions_detail']:
                print(f"  {pos['symbol']} {pos['type']} {pos['volume']} @ {pos['open_price']:.5f}")
                print(f"    SL: {pos['stop_loss']:.5f} | TP: {pos['take_profit']:.5f}")
                print(f"    Profit: {pos['profit']:.2f}")
        
        print("="*60)

# Example usage and main execution
def main():
    """Main function to run the trading bot"""
    bot = MT5TradingBot()
    
    try:
        # Print initial status
        bot.print_status()
        
        # Start trading
        bot.start_trading()
        
        # Keep running and print status periodically
        while True:
            time.sleep(300)  # 5 minutes
            bot.print_status()
            
            # Check if bot is still running
            if bot.state != TradingState.RUNNING:
                break
                
    except KeyboardInterrupt:
        print("\n Received shutdown signal...")
        bot.stop_trading(close_positions=True)
    except Exception as e:
        print(f" Unexpected error: {e}")
        bot.stop_trading(close_positions=False)
    finally:
        print("Trading bot shutdown complete")

if __name__ == "__main__":
    main()
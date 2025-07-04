import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import time
import threading
import json
import os
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings('ignore')

class EnhancedForexTradingBot:
    def __init__(self, symbols=None, config_file='bot_config.json'):
        # Default forex symbols (Yahoo Finance format)
        if symbols is None:
            self.symbols = [
                'EURUSD=X',    # EUR/USD
                'GBPUSD=X',    # GBP/USD
                'USDJPY=X',    # USD/JPY
                'AUDUSD=X',    # AUD/USD
                'USDCAD=X',    # USD/CAD
                'USDCHF=X',    # USD/CHF
                'NZDUSD=X',    # NZD/USD
                'XAUUSD=X',    # Gold
            ]
        else:
            self.symbols = symbols
            
        self.config_file = config_file
        self.load_config()
        
        self.data = {}
        self.signals = {}
        self.win_rates = {}
        self.trade_history = []
        self.is_running = False
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'total_pips': 0,
            'best_pair': '',
            'worst_pair': ''
        }
        
    def load_config(self):
        """Load configuration from file or create default"""
        default_config = {
            'risk_management': {
                'max_risk_per_trade': 0.02,  # 2% risk per trade
                'stop_loss_atr_multiplier': 2.0,
                'take_profit_atr_multiplier': 3.0,
                'max_concurrent_trades': 3
            },
            'signal_thresholds': {
                'high_confidence_threshold': 70,
                'low_confidence_threshold': 40,
                'minimum_signal_strength': 3
            },
            'time_filters': {
                'avoid_news_hours': True,
                'london_session_boost': True,
                'new_york_session_boost': True
            },
            'alert_settings': {
                'enable_alerts': True,
                'alert_threshold': 4  # Signal strength threshold for alerts
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = default_config
                self.save_config()
        except:
            self.config = default_config
            
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f" Error saving config: {e}")
    
    def get_market_session(self) -> str:
        """Determine current market session"""
        utc_hour = datetime.utcnow().hour
        
        if 0 <= utc_hour < 8:
            return "TOKYO"
        elif 8 <= utc_hour < 16:
            return "LONDON"
        elif 16 <= utc_hour < 24:
            return "NEW_YORK"
        else:
            return "OFF_HOURS"
    
    def is_high_impact_news_time(self) -> bool:
        """Check if current time is during high-impact news (simplified)"""
        utc_now = datetime.utcnow()
        utc_hour = utc_now.hour
        utc_minute = utc_now.minute
        
        # Avoid common news times (8:30 UTC, 12:30 UTC, 14:00 UTC)
        news_times = [(8, 30), (12, 30), (14, 0)]
        
        for hour, minute in news_times:
            if utc_hour == hour and abs(utc_minute - minute) <= 30:
                return True
        return False
    
    def fetch_extended_data(self, symbol: str, period: str = '10d') -> Optional[pd.DataFrame]:
        """Fetch extended forex data with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval='1h')
            if len(data) < 100:  # Need more data for better analysis
                print(f"Insufficient data for {symbol}")
                return None
            return data
        except Exception as e:
            print(f" Error fetching {symbol}: {e}")
            return None
    
    def calculate_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators"""
        df = data.copy()
        
        # Multiple timeframe EMAs
        df['EMA_8'] = df['Close'].ewm(span=8).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        
        # Trend strength
        df['Trend_Strength'] = (df['EMA_8'] - df['EMA_21']) / df['EMA_21'] * 100
        
        # RSI with divergence detection
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_MA'] = df['RSI'].rolling(window=3).mean()
        
        # MACD with histogram momentum
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Momentum'] = df['MACD_Histogram'].diff()
        
        # Dynamic Bollinger Bands
        bb_period = 20
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic with smoothing
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # ATR and volatility
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # Support/Resistance levels
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        
        # Price momentum and acceleration
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Momentum'] = df['Close'].pct_change(5)
        df['Price_Acceleration'] = df['Price_Momentum'].diff()
        
        # Volume analysis (if available)
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        else:
            df['Volume_Ratio'] = 1
            
        return df
    
    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate risk and position sizing metrics"""
        latest = data.iloc[-1]
        atr = latest['ATR']
        price = latest['Close']
        
        # Calculate stop loss and take profit levels
        stop_loss_distance = atr * self.config['risk_management']['stop_loss_atr_multiplier']
        take_profit_distance = atr * self.config['risk_management']['take_profit_atr_multiplier']
        
        return {
            'atr': atr,
            'stop_loss_distance': stop_loss_distance,
            'take_profit_distance': take_profit_distance,
            'risk_reward_ratio': take_profit_distance / stop_loss_distance,
            'volatility_rank': self.get_volatility_rank(data)
        }
    
    def get_volatility_rank(self, data: pd.DataFrame) -> str:
        """Rank current volatility vs historical"""
        current_atr = data['ATR'].iloc[-1]
        avg_atr = data['ATR'].rolling(window=50).mean().iloc[-1]
        
        if current_atr > avg_atr * 1.5:
            return "HIGH"
        elif current_atr < avg_atr * 0.7:
            return "LOW"
        else:
            return "NORMAL"
    
    def generate_enhanced_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced trading signals with multiple confirmations"""
        df = data.copy()
        
        # Initialize signal columns
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        df['Confidence'] = 0
        df['Entry_Reason'] = ''
        
        # Trend conditions
        strong_uptrend = (df['EMA_8'] > df['EMA_21']) & (df['EMA_21'] > df['EMA_50'])
        strong_downtrend = (df['EMA_8'] < df['EMA_21']) & (df['EMA_21'] < df['EMA_50'])
        
        # Momentum conditions
        rsi_bullish = df['RSI'] > 45
        rsi_bearish = df['RSI'] < 55
        rsi_oversold = df['RSI'] < 30
        rsi_overbought = df['RSI'] > 70
        
        # MACD conditions
        macd_bullish = df['MACD'] > df['MACD_Signal']
        macd_bearish = df['MACD'] < df['MACD_Signal']
        macd_momentum_up = df['MACD_Momentum'] > 0
        macd_momentum_down = df['MACD_Momentum'] < 0
        
        # Bollinger Band conditions
        bb_squeeze = df['BB_Width'] < df['BB_Width'].rolling(window=20).mean() * 0.8
        bb_expansion = df['BB_Width'] > df['BB_Width'].rolling(window=20).mean() * 1.2
        bb_upper_break = df['Close'] > df['BB_Upper']
        bb_lower_break = df['Close'] < df['BB_Lower']
        
        # Stochastic conditions
        stoch_bullish = (df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'] > 20)
        stoch_bearish = (df['Stoch_K'] < df['Stoch_D']) & (df['Stoch_K'] < 80)
        
        # Volatility conditions
        normal_volatility = (df['ATR_Ratio'] > 0.001) & (df['ATR_Ratio'] < 0.01)
        
        # Build signal conditions with scoring
        buy_conditions = [
            strong_uptrend,
            rsi_bullish | rsi_oversold,
            macd_bullish & macd_momentum_up,
            bb_expansion | bb_lower_break,
            stoch_bullish,
            normal_volatility,
            df['Price_Momentum'] > 0,
            df['Close'] > df['EMA_200']  # Long-term trend filter
        ]
        
        sell_conditions = [
            strong_downtrend,
            rsi_bearish | rsi_overbought,
            macd_bearish & macd_momentum_down,
            bb_expansion | bb_upper_break,
            stoch_bearish,
            normal_volatility,
            df['Price_Momentum'] < 0,
            df['Close'] < df['EMA_200']  # Long-term trend filter
        ]
        
        # Calculate scores
        df['Buy_Score'] = sum(buy_conditions)
        df['Sell_Score'] = sum(sell_conditions)
        
        # Apply session boost
        session = self.get_market_session()
        if session in ['LONDON', 'NEW_YORK'] and self.config['time_filters'][f'{session.lower()}_session_boost']:
            df['Buy_Score'] = df['Buy_Score'] * 1.2
            df['Sell_Score'] = df['Sell_Score'] * 1.2
        
        # Generate final signals
        min_strength = self.config['signal_thresholds']['minimum_signal_strength']
        strong_buy = df['Buy_Score'] >= min_strength
        strong_sell = df['Sell_Score'] >= min_strength
        
        df.loc[strong_buy, 'Signal'] = 1
        df.loc[strong_sell, 'Signal'] = -1
        df.loc[strong_buy, 'Signal_Strength'] = df.loc[strong_buy, 'Buy_Score']
        df.loc[strong_sell, 'Signal_Strength'] = df.loc[strong_sell, 'Sell_Score']
        
        # Calculate confidence based on multiple factors
        df['Confidence'] = np.where(
            df['Signal'] != 0,
            np.minimum(df['Signal_Strength'] / 8 * 100, 100),
            0
        )
        
        return df
    
    def calculate_enhanced_win_rate(self, signals_data: pd.DataFrame, lookforward_hours: int = 6) -> Dict:
        """Calculate enhanced win rate with additional metrics"""
        df = signals_data.copy()
        results = {
            'total_signals': 0,
            'correct_predictions': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'best_win': 0,
            'worst_loss': 0,
            'profit_factor': 0
        }
        
        signal_rows = df[df['Signal'] != 0].copy()
        profits = []
        losses = []
        
        for idx, row in signal_rows.iterrows():
            try:
                current_idx = df.index.get_loc(idx)
                if current_idx + lookforward_hours < len(df):
                    future_price = df.iloc[current_idx + lookforward_hours]['Close']
                    current_price = row['Close']
                    price_change = (future_price - current_price) / current_price
                    
                    if row['Signal'] == 1:  # BUY signal
                        if price_change > 0.001:  # Profitable
                            results['correct_predictions'] += 1
                            profits.append(price_change)
                        else:
                            losses.append(abs(price_change))
                    elif row['Signal'] == -1:  # SELL signal
                        if price_change < -0.001:  # Profitable
                            results['correct_predictions'] += 1
                            profits.append(abs(price_change))
                        else:
                            losses.append(price_change)
                    
                    results['total_signals'] += 1
            except:
                continue
        
        if results['total_signals'] > 0:
            results['win_rate'] = (results['correct_predictions'] / results['total_signals']) * 100
            
            if profits:
                results['avg_profit'] = np.mean(profits) * 100
                results['best_win'] = max(profits) * 100
                
            if losses:
                results['avg_loss'] = np.mean(losses) * 100
                results['worst_loss'] = max(losses) * 100
                
            if profits and losses:
                total_profit = sum(profits)
                total_loss = sum(losses)
                results['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return results
    
    def analyze_symbol_enhanced(self, symbol: str) -> Optional[Dict]:
        """Enhanced symbol analysis with risk management"""
        # Skip if news time and configured to avoid
        if self.config['time_filters']['avoid_news_hours'] and self.is_high_impact_news_time():
            print(f" Skipping {symbol} - High impact news time")
            return None
        
        # Fetch data
        data = self.fetch_extended_data(symbol)
        if data is None:
            return None
        
        # Calculate indicators
        data_with_indicators = self.calculate_advanced_indicators(data)
        
        # Generate signals
        signals_data = self.generate_enhanced_signals(data_with_indicators)
        
        # Calculate performance metrics
        performance = self.calculate_enhanced_win_rate(signals_data)
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(signals_data)
        
        # Get latest signal
        latest = signals_data.iloc[-1]
        raw_signal = latest['Signal']
        
        # Determine final trading decision with enhanced logic
        final_signal, trade_reason = self.determine_trade_decision(
            symbol, raw_signal, performance, latest, risk_metrics
        )
        
        # Store data
        self.data[symbol] = signals_data
        self.win_rates[symbol] = performance['win_rate']
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'price': round(latest['Close'], 5),
            'raw_signal': 'BUY' if raw_signal == 1 else ('SELL' if raw_signal == -1 else 'HOLD'),
            'signal_strength': int(latest['Signal_Strength']),
            'confidence': round(latest['Confidence'], 1),
            'win_rate': round(performance['win_rate'], 1),
            'profit_factor': round(performance['profit_factor'], 2),
            'final_signal': final_signal,
            'trade_reason': trade_reason,
            'entry_reason': latest['Entry_Reason'],
            'risk_metrics': risk_metrics,
            'market_session': self.get_market_session(),
            'rsi': round(latest['RSI'], 1),
            'trend_strength': round(latest['Trend_Strength'], 2),
            'volatility': risk_metrics['volatility_rank']
        }
        
        return result
    
    def determine_trade_decision(self, symbol: str, raw_signal: int, performance: Dict, 
                               latest_data: pd.Series, risk_metrics: Dict) -> Tuple[str, str]:
        """Enhanced trade decision logic"""
        win_rate = performance['win_rate']
        profit_factor = performance['profit_factor']
        confidence = latest_data['Confidence']
        
        # High confidence threshold
        high_threshold = self.config['signal_thresholds']['high_confidence_threshold']
        low_threshold = self.config['signal_thresholds']['low_confidence_threshold']
        
        # Decision logic
        if win_rate > high_threshold and profit_factor > 1.5 and confidence > 70:
            return ('BUY' if raw_signal == 1 else 'SELL'), f"HIGH_CONFIDENCE (WR:{win_rate:.1f}% PF:{profit_factor:.1f})"
        
        elif win_rate < low_threshold and profit_factor < 0.8:
            opposite_signal = 'SELL' if raw_signal == 1 else 'BUY'
            return opposite_signal, f"REVERSE_TRADE (WR:{win_rate:.1f}% PF:{profit_factor:.1f})"
        
        elif risk_metrics['volatility_rank'] == 'HIGH':
            return 'NO_TRADE', f"HIGH_VOLATILITY_SKIP (Vol:{risk_metrics['volatility_rank']})"
        
        elif confidence < 50:
            return 'NO_TRADE', f"LOW_CONFIDENCE (Conf:{confidence:.1f}%)"
        
        else:
            return 'NO_TRADE', f"MODERATE_CONDITIONS (WR:{win_rate:.1f}% Conf:{confidence:.1f}%)"
    
    def run_enhanced_scan(self) -> List[Dict]:
        """Run enhanced market scan"""
        print(f"\n ENHANCED FOREX SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Market Session: {self.get_market_session()}")
        print("=" * 80)
        
        trading_opportunities = []
        high_priority_alerts = []
        
        for symbol in self.symbols:
            try:
                result = self.analyze_symbol_enhanced(symbol)
                if result:
                    print(f"\n {result['symbol']} | Session: {result['market_session']}")
                    print(f" Price: {result['price']} | Volatility: {result['volatility']}")
                    print(f" Win Rate: {result['win_rate']}% | Profit Factor: {result['profit_factor']}")
                    print(f" Confidence: {result['confidence']}% | Trend: {result['trend_strength']}%")
                    print(f" DECISION: {result['final_signal']} - {result['trade_reason']}")
                    
                    if result['final_signal'] != 'NO_TRADE':
                        trading_opportunities.append(result)
                        
                        # Check for high priority alerts
                        if (result['confidence'] > self.config['alert_settings']['alert_threshold'] * 20 and
                            self.config['alert_settings']['enable_alerts']):
                            high_priority_alerts.append(result)
                
            except Exception as e:
                print(f" Error analyzing {symbol}: {e}")
        
        # Display results
        if high_priority_alerts:
            print(f"\n HIGH PRIORITY ALERTS: {len(high_priority_alerts)}")
            for alert in high_priority_alerts:
                print(f" {alert['symbol']}: {alert['final_signal']} | Confidence: {alert['confidence']}%")
        
        if trading_opportunities:
            print(f"\n TRADING OPPORTUNITIES: {len(trading_opportunities)}")
            for opp in sorted(trading_opportunities, key=lambda x: x['confidence'], reverse=True):
                print(f"• {opp['symbol']}: {opp['final_signal']} | Conf: {opp['confidence']}% | WR: {opp['win_rate']}%")
        else:
            print(f"\n NO TRADING OPPORTUNITIES FOUND")
        
        # Update performance tracking
        self.update_performance_metrics(trading_opportunities)
        
        return trading_opportunities
    
    def update_performance_metrics(self, opportunities: List[Dict]):
        """Update bot performance metrics"""
        if opportunities:
            self.performance_metrics['total_signals'] += len(opportunities)
            
            # Find best and worst performing pairs
            win_rates = {opp['symbol']: opp['win_rate'] for opp in opportunities}
            if win_rates:
                best_pair = max(win_rates.items(), key=lambda x: x[1])
                worst_pair = min(win_rates.items(), key=lambda x: x[1])
                
                self.performance_metrics['best_pair'] = f"{best_pair[0]} ({best_pair[1]:.1f}%)"
                self.performance_metrics['worst_pair'] = f"{worst_pair[0]} ({worst_pair[1]:.1f}%)"
    
    def get_detailed_performance_report(self) -> str:
        """Generate detailed performance report"""
        if not self.win_rates:
            return " No performance data available yet."
        
        report = "\n" + "="*60
        report += "\n DETAILED PERFORMANCE REPORT"
        report += "\n" + "="*60
        
        # Overall stats
        total_pairs = len(self.win_rates)
        avg_win_rate = sum(self.win_rates.values()) / len(self.win_rates)
        
        report += f"\n Overall Statistics:"
        report += f"\n   • Total Pairs Analyzed: {total_pairs}"
        report += f"\n   • Average Win Rate: {avg_win_rate:.1f}%"
        report += f"\n   • Total Signals Generated: {self.performance_metrics['total_signals']}"
        
        # Performance by pair
        report += f"\n\n Performance by Pair:"
        sorted_pairs = sorted(self.win_rates.items(), key=lambda x: x[1], reverse=True)
        
        for symbol, win_rate in sorted_pairs:
            status = " TRADE" if win_rate > 70 else (" REVERSE" if win_rate < 40 else " SKIP")
            report += f"\n   • {symbol}: {win_rate:.1f}% {status}"
        
        # Best/Worst performers
        if self.performance_metrics['best_pair']:
            report += f"\n\n Best Performer: {self.performance_metrics['best_pair']}"
            report += f"\n  Worst Performer: {self.performance_metrics['worst_pair']}"
        
        # Current market session
        report += f"\n\n Current Session: {self.get_market_session()}"
        report += f"\n Next News Check: {'⚠ HIGH IMPACT TIME' if self.is_high_impact_news_time() else ' Clear'}"
        
        return report

# Usage example with enhanced features
if __name__ == "__main__":
    # Initialize enhanced bot
    enhanced_bot = EnhancedForexTradingBot()
    
    print(" Enhanced Forex Trading Bot Initialized")
    print(" Configuration loaded with risk management")
    
    # Run enhanced scan
    opportunities = enhanced_bot.run_enhanced_scan()
    
    # Show detailed performance report
    print(enhanced_bot.get_detailed_performance_report())
    
    # Save configuration
    enhanced_bot.save_config()
    print("\n Configuration saved")
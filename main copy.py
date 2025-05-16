import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import os

# Настройка логирования
if not os.path.exists('logs'):
    os.makedirs('logs')

log_filename = f'logs/trading_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

class FuturesTrader:
    def __init__(self, symbol="BTCUSDT", timeframe="1h", 
                 initial_balance=1000, fee=0.0004, leverage=20, 
                 slippage=0.0005, risk_per_trade=1):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.fee = fee  # 0.04% fee
        self.leverage = leverage
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade  # % от баланса на риск
        self.strategies = []
        logging.info(f"Инициализация торгового бота: {symbol}, таймфрейм: {timeframe}, начальный баланс: {initial_balance}, плечо: {leverage}")
    
    def add_strategy(self, strategy):
        """Добавление стратегии в систему"""
        self.strategies.append(strategy)
        logging.info(f"Добавлена стратегия: {strategy.name}")
    
    def get_historical_data(self, days=30):
        """Получение исторических данных с Binance"""
        logging.info(f"Загрузка исторических данных для {self.symbol}, период: {days} дней")
        
        # Преобразование таймфрейма в формат Binance
        binance_timeframe = self.timeframe
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        url = f"https://api.binance.com/api/v3/klines?symbol={self.symbol}&interval={binance_timeframe}&startTime={start_time}&endTime={end_time}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if not data or isinstance(data, dict) and 'code' in data:
                logging.error(f"API Error: {data}")
                return None
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                             'close_time', 'quote_asset_volume', 'trades', 
                                             'taker_buy_base', 'taker_buy_quote', 'ignore'])
            
            # Преобразование типов данных
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logging.info(f"Загружено {len(df)} записей исторических данных")
            return df
        
        except Exception as e:
            logging.error(f"Ошибка при загрузке исторических данных: {e}")
            return None
    
    def _combine_strategy_signals(self, signals_with_reasons):
        """
        Взвешенное комбинирование сигналов от разных стратегий
        """
        if not signals_with_reasons:
            return 0, "Нет сигналов"
        
        # Веса для каждой стратегии
        strategy_weights = {
            "Trend Following Strategy": 0.4,
            "Breakout Strategy": 0.6
        }
        
        # Взвешенное суммирование сигналов
        weighted_signal = 0
        all_reasons = []
        
        for strategy_name, (signal, reason) in signals_with_reasons.items():
            weight = strategy_weights.get(strategy_name, 0.5)
            weighted_signal += signal * weight
            if signal != 0:
                all_reasons.append(f"{strategy_name}: {reason}")
        
        # Определение итогового сигнала по порогу
        if weighted_signal >= 0.3:
            final_signal = 1
        elif weighted_signal <= -0.3:
            final_signal = -1
        else:
            final_signal = 0
        
        reason = ", ".join(all_reasons) if all_reasons else "Недостаточно сильный сигнал"
        
        return final_signal, reason

    def run_backtest(self, df):
        """Запуск бэктестинга на исторических данных"""
        if df is None or len(df) == 0:
            logging.error("Нет данных для бэктестинга")
            return None
        
        logging.info(f"Начало бэктестинга: баланс={self.initial_balance}, левередж={self.leverage}, комиссия={self.fee}")
        
        # Инициализация переменных
        balance = self.initial_balance
        equity = [self.initial_balance]  # История капитала
        position = 0  # 0 - нет позиции, 1 - лонг, -1 - шорт
        position_size = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trades = []  # История сделок
        
        # Расчет индикаторов для всех стратегий
        for strategy in self.strategies:
            df = strategy.calculate_indicators(df)
        
        # Добавление колонок для сигналов и баланса
        df['Signal'] = 0
        df['Balance'] = self.initial_balance
        
        # Перебор исторических данных
        for i in tqdm(range(1, len(df)), desc="Бэктестинг"):
            current_time = df.index[i]
            price = df['close'].iloc[i]
            
            # Получение сигналов от всех стратегий
            signals_with_reasons = {}
            
            for strategy in self.strategies:
                signal, reason = strategy.generate_signal(df.iloc[:i+1])
                signals_with_reasons[strategy.name] = (signal, reason)
            
            # Комбинирование сигналов
            signal, signal_reason = self._combine_strategy_signals(signals_with_reasons)
            
            df.iloc[i, df.columns.get_loc('Signal')] = signal
            
            # Проверка баланса
            if balance <= 0:
                logging.warning(f"Баланс <= 0 ({balance}), торговля остановлена")
                break
            
            # Проверка стоп-лосса и тейк-профита для открытых позиций
            if position != 0:
                if position == 1:  # Лонг
                    if price <= stop_loss:  # Стоп-лосс
                        profit = position_size * (stop_loss / entry_price - 1) * self.leverage
                        realized_pnl = balance * profit - (position_size * entry_price * self.leverage * self.fee * 2)
                        balance += realized_pnl
                        
                        trades.append({
                            'entry_time': position_entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'position': position,
                            'size': position_size,
                            'pnl': realized_pnl,
                            'exit_reason': 'Stop Loss',
                            'entry_reason': position_reason
                        })
                        
                        logging.info(f"Стоп-лосс (ЛОНГ): цена={stop_loss}, прибыль={realized_pnl:.2f}, баланс={balance:.2f}")
                        position = 0
                        position_size = 0
                    
                    elif price >= take_profit:  # Тейк-профит
                        profit = position_size * (take_profit / entry_price - 1) * self.leverage
                        realized_pnl = balance * profit - (position_size * entry_price * self.leverage * self.fee * 2)
                        balance += realized_pnl
                        
                        trades.append({
                            'entry_time': position_entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'position': position,
                            'size': position_size,
                            'pnl': realized_pnl,
                            'exit_reason': 'Take Profit',
                            'entry_reason': position_reason
                        })
                        
                        logging.info(f"Тейк-профит (ЛОНГ): цена={take_profit}, прибыль={realized_pnl:.2f}, баланс={balance:.2f}")
                        position = 0
                        position_size = 0
                
                elif position == -1:  # Шорт
                    if price >= stop_loss:  # Стоп-лосс
                        profit = position_size * (1 - stop_loss / entry_price) * self.leverage
                        realized_pnl = balance * profit - (position_size * entry_price * self.leverage * self.fee * 2)
                        balance += realized_pnl
                        
                        trades.append({
                            'entry_time': position_entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'position': position,
                            'size': position_size,
                            'pnl': realized_pnl,
                            'exit_reason': 'Stop Loss',
                            'entry_reason': position_reason
                        })
                        
                        logging.info(f"Стоп-лосс (ШОРТ): цена={stop_loss}, прибыль={realized_pnl:.2f}, баланс={balance:.2f}")
                        position = 0
                        position_size = 0
                    
                    elif price <= take_profit:  # Тейк-профит
                        profit = position_size * (1 - take_profit / entry_price) * self.leverage
                        realized_pnl = balance * profit - (position_size * entry_price * self.leverage * self.fee * 2)
                        balance += realized_pnl
                        
                        trades.append({
                            'entry_time': position_entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'position': position,
                            'size': position_size,
                            'pnl': realized_pnl,
                            'exit_reason': 'Take Profit',
                            'entry_reason': position_reason
                        })
                        
                        logging.info(f"Тейк-профит (ШОРТ): цена={take_profit}, прибыль={realized_pnl:.2f}, баланс={balance:.2f}")
                        position = 0
                        position_size = 0
            
            # Обработка новых сигналов, если нет активной позиции
            if position == 0 and signal != 0:
                # Расчет риска в долларах
                risk_amount = balance * self.risk_per_trade
                
                # Вычисление ATR для динамического стоп-лосса
                atr = df['atr'].iloc[i] if 'atr' in df.columns else price * 0.01  # 1% по умолчанию
                
                if signal == 1:  # Сигнал на покупку
                    # Расчет стоп-лосса и тейк-профита на основе ATR
                    sl_distance = max(atr * 2, price * 0.01)  # увеличено до 2 ATR, минимум 1%
                    stop_loss = price - sl_distance
                    take_profit = price + sl_distance * 2  # RR уменьшено до 1:2 для повышения процента выигрышных сделок
                    
                    # Расчет размера позиции на основе риска
                    max_position_size = balance * self.leverage / price  # Максимальный размер в базовой валюте
                    risk_based_size = risk_amount / (sl_distance * self.leverage)  # Размер на основе риска
                    position_size = min(max_position_size, risk_based_size)
                    
                    # Учет проскальзывания
                    entry_price = price * (1 + self.slippage)
                    
                    position = 1
                    position_entry_time = current_time
                    position_reason = signal_reason
                    
                    logging.info(f"ОТКРЫТИЕ ЛОНГ: цена={entry_price:.2f}, размер={position_size:.4f}, стоп={stop_loss:.2f}, профит={take_profit:.2f}, баланс={balance:.2f}, причина: {signal_reason}")
                
                elif signal == -1:  # Сигнал на продажу
                    # Расчет стоп-лосса и тейк-профита на основе ATR
                    sl_distance = max(atr * 2, price * 0.01)  # увеличено до 2 ATR, минимум 1%
                    stop_loss = price + sl_distance
                    take_profit = price - sl_distance * 2  # RR уменьшено до 1:2 для повышения процента выигрышных сделок
                    
                    # Расчет размера позиции на основе риска
                    max_position_size = balance * self.leverage / price  # Максимальный размер в базовой валюте
                    risk_based_size = risk_amount / (sl_distance * self.leverage)  # Размер на основе риска
                    position_size = min(max_position_size, risk_based_size)
                    
                    # Учет проскальзывания
                    entry_price = price * (1 - self.slippage)
                    
                    position = -1
                    position_entry_time = current_time
                    position_reason = signal_reason
                    
                    logging.info(f"ОТКРЫТИЕ ШОРТ: цена={entry_price:.2f}, размер={position_size:.4f}, стоп={stop_loss:.2f}, профит={take_profit:.2f}, баланс={balance:.2f}, причина: {signal_reason}")
            
            # Обновление баланса
            df.iloc[i, df.columns.get_loc('Balance')] = balance
            
            # Обновление виртуального капитала (с учетом открытых позиций)
            if position != 0:
                # Расчет нереализованной прибыли/убытка
                if position == 1:  # Лонг
                    unrealized_pnl = position_size * (price / entry_price - 1) * self.leverage
                else:  # Шорт
                    unrealized_pnl = position_size * (1 - price / entry_price) * self.leverage
                
                equity_balance = balance + (balance * unrealized_pnl)
                equity.append(equity_balance)
            else:
                equity.append(balance)
        
        # Закрытие позиции в конце периода, если она открыта
        if position != 0:
            # Последняя цена
            last_price = df['close'].iloc[-1]
            
            if position == 1:  # Лонг
                profit = position_size * (last_price / entry_price - 1) * self.leverage
            else:  # Шорт
                profit = position_size * (1 - last_price / entry_price) * self.leverage
            
            realized_pnl = balance * profit - (position_size * entry_price * self.leverage * self.fee * 2)
            balance += realized_pnl
            
            trades.append({
                'entry_time': position_entry_time,
                'exit_time': df.index[-1],
                'entry_price': entry_price,
                'exit_price': last_price,
                'position': position,
                'size': position_size,
                'pnl': realized_pnl,
                'exit_reason': 'End of Backtest',
                'entry_reason': position_reason
            })
            
            logging.info(f"Закрытие позиции в конце теста: {'ЛОНГ' if position == 1 else 'ШОРТ'}, прибыль={realized_pnl:.2f}, итоговый баланс={balance:.2f}")
        
        # Расчет метрик
        stats = self.calculate_statistics(trades, equity, df)
        
        # Возвращаем результаты
        results = {
            'dataframe': df,
            'trades': trades,
            'equity': equity,
            'stats': stats,
            'final_balance': balance
        }
        
        return results
    
    def calculate_statistics(self, trades, equity, df):
        """Расчет статистических показателей торговой системы"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'max_win': 0,
                'max_loss': 0,
                'avg_trade': 0,
                'avg_bars': 0
            }
        
        # Общие метрики
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        # Процент выигрышных сделок
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Прибыль и убытки
        total_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        
        # Соотношение прибыли и убытков
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Средние значения
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        avg_trade = (total_profit - total_loss) / total_trades if total_trades > 0 else 0
        
        # Максимальная прибыль и убыток
        max_win = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
        max_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Среднее количество баров в сделке
        def bars_in_trade(trade):
            entry = pd.Timestamp(trade['entry_time'])
            exit_time = pd.Timestamp(trade['exit_time'])
            return len(df[(df.index >= entry) & (df.index <= exit_time)])
        
        avg_bars = sum(bars_in_trade(t) for t in trades) / total_trades if total_trades > 0 else 0
        
        # Расчет просадки
        equity_array = np.array(equity)
        max_equity = np.maximum.accumulate(equity_array)
        drawdown = (max_equity - equity_array) / max_equity
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Расчет коэффициента Шарпа
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,  # в процентах
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,  # в процентах
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_trade': avg_trade,
            'avg_bars': avg_bars
        }
    
    def plot_results(self, results):
        """Визуализация результатов бэктестинга"""
        if not results or 'dataframe' not in results:
            logging.error("Нет данных для визуализации")
            return
        
        df = results['dataframe']
        trades = results['trades']
        equity = results['equity']
        stats = results['stats']
        
        # Создание директории для результатов, если ее нет
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # Масштабирование массива equity
        equity_values = np.array(equity)
        
        # Создаем фигуру с подграфиками
        fig = plt.figure(figsize=(14, 16))
        
        # График цены с сигналами
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
        ax1.plot(df.index, df['close'], label='Цена', color='blue')
        
        # Добавляем SMA, если она есть
        if 'sma20' in df.columns:
            ax1.plot(df.index, df['sma20'], label='SMA20', color='orange', alpha=0.7)
        if 'sma50' in df.columns:
            ax1.plot(df.index, df['sma50'], label='SMA50', color='green', alpha=0.7)
        if 'sma200' in df.columns:
            ax1.plot(df.index, df['sma200'], label='SMA200', color='red', alpha=0.7)
        
        # Отметим точки входа и выхода на графике
        for trade in trades:
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            
            if trade['position'] == 1:  # Лонг
                ax1.scatter(entry_time, entry_price, color='green', marker='^', s=100, label='_nolegend_')
                ax1.scatter(exit_time, exit_price, color='red', marker='x', s=100, label='_nolegend_')
            else:  # Шорт
                ax1.scatter(entry_time, entry_price, color='red', marker='v', s=100, label='_nolegend_')
                ax1.scatter(exit_time, exit_price, color='green', marker='x', s=100, label='_nolegend_')
        
        ax1.set_title('Цена и сигналы')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # График индикаторов
        ax2 = plt.subplot2grid((4, 1), (2, 0))
        
        # Добавляем RSI, если он есть
        if 'rsi' in df.columns:
            ax2.plot(df.index, df['rsi'], label='RSI', color='purple')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.set_ylim(0, 100)
            ax2.set_title('RSI')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        # График капитала
        ax3 = plt.subplot2grid((4, 1), (3, 0))
        ax3.plot(df.index, equity_values[:len(df)], label='Капитал', color='green')
        ax3.set_title('Динамика капитала')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Создание таблицы с метриками
        stats_text = f"""
        Всего сделок: {stats['total_trades']}
        Процент выигрышных: {stats['win_rate']:.2f}%
        Прибыль/Убыток: {stats['profit_factor']:.2f}
        Коэффициент Шарпа: {stats['sharpe_ratio']:.2f}
        Макс. просадка: {stats['max_drawdown']:.2f}%
        Средняя прибыль: {stats['avg_profit']:.2f}
        Средний убыток: {stats['avg_loss']:.2f}
        Макс. прибыль: {stats['max_win']:.2f}
        Макс. убыток: {stats['max_loss']:.2f}
        """
        
        fig.text(0.15, 0.01, stats_text, fontsize=10)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('results/backtest_results.png', dpi=300)
        plt.close()
        
        logging.info(f"График сохранен в results/backtest_results.png")
        
        # Сохранение сделок в CSV
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv('results/trades.csv', index=False)
        
        # Сохранение статистики в JSON
        with open('results/stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        return fig

# Стратегии торговли
class Strategy:
    """Базовый класс для торговых стратегий"""
    def __init__(self, name):
        self.name = name
    
    def calculate_indicators(self, df):
        """Расчет индикаторов (должен быть переопределен в дочерних классах)"""
        return df
    
    def generate_signal(self, df):
        """Генерация торгового сигнала (должен быть переопределен в дочерних классах)"""
        return 0, "No signal"

class TrendFollowingStrategy(Strategy):
    """Стратегия следования за трендом"""
    

    def _calculate_sma(self, data, period):
        """Расчет простой скользящей средней"""
        return data.rolling(window=period).mean()

    def _calculate_ema(self, data, period):
        """Расчет экспоненциальной скользящей средней"""
        return data.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, data, period=14):
        """Расчет индикатора относительной силы (RSI)"""
        delta = data.diff()
        
        # Отдельные серии для положительных и отрицательных изменений
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Расчет среднего выигрыша и средней потери
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Расчет относительной силы
        rs = avg_gain / avg_loss
        
        # Расчет RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_adx(self, df, period=14):
        """Расчет индекса направленного движения (ADX)"""
        # Клонирование исходного DataFrame
        df = df.copy()
        
        # Вычисление True Range (TR)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Вычисление Directional Movement (DM)
        df['up_move'] = df['high'].diff()
        df['down_move'] = df['low'].diff()
        
        # Положительное и отрицательное направленное движение
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Сглаживание TR, +DM, -DM
        df['tr_smoothed'] = df['tr'].rolling(window=period).sum()
        df['plus_dm_smoothed'] = df['plus_dm'].rolling(window=period).sum()
        df['minus_dm_smoothed'] = df['minus_dm'].rolling(window=period).sum()
        
        # Вычисление +DI и -DI
        df['plus_di'] = 100 * (df['plus_dm_smoothed'] / df['tr_smoothed'])
        df['minus_di'] = 100 * (df['minus_dm_smoothed'] / df['tr_smoothed'])
        
        # Вычисление DX
        df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
        
        # Вычисление ADX (сглаженный DX)
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # Удаление временных колонок
        df = df.drop(['high_low', 'high_close', 'low_close', 'up_move', 'down_move', 
                    'plus_dm', 'minus_dm', 'tr_smoothed', 'plus_dm_smoothed', 
                    'minus_dm_smoothed', 'dx'], axis=1, errors='ignore')
        
        return df

    def _calculate_atr(self, df, period=14):
        """Расчет Average True Range (ATR)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Расчет True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Расчет ATR как скользящего среднего от TR
        atr = tr.rolling(window=period).mean()
        
        return atr

    def __init__(self, ema_short=20, ema_long=50, rsi_period=14, rsi_oversold=40, rsi_overbought=60, adx_period=14, adx_threshold=25):
        super().__init__("Trend Following")
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
    

    def _calculate_sma(self, data, period):
    
        return data.rolling(window=period).mean()

    def _calculate_ema(self, data, period):
    
        return data.ewm(span=period, adjust=False).mean()
    def calculate_indicators(self, df):
        
        # Клонирование исходного DataFrame
        df = df.copy()
    
    # Расчет EMA/SMA
        df['sma20'] = self._calculate_sma(df['close'], self.ema_short)
        df['sma50'] = self._calculate_sma(df['close'], self.ema_long)
        
        # Расчет RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # Расчет ADX
        df = self._calculate_adx(df, self.adx_period)
        
        # Расчет ATR для стоп-лоссов
        df['atr'] = self._calculate_atr(df, 14)
        
        return df
    
    def generate_signal(self, df):
        """Генерация сигналов для тренд-следящей стратегии с улучшенными фильтрами"""
        if len(df) < max(self.ema_short, self.ema_long, self.rsi_period) + 5:
            return 0, "Недостаточно данных"
        
        # Получение последних значений индикаторов
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Проверка силы тренда через ADX
        strong_trend = current['adx'] > self.adx_threshold if 'adx' in current else False
        
        # ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ
        # Проверка направления тренда за последние N баров
        last_n_bars = 5
        uptrend_count = sum(1 for i in range(1, min(last_n_bars+1, len(df))) 
                            if df['close'].iloc[-i] > df['close'].iloc[-i-1])
        downtrend_count = sum(1 for i in range(1, min(last_n_bars+1, len(df))) 
                            if df['close'].iloc[-i] < df['close'].iloc[-i-1])
        
        uptrend_consistency = uptrend_count / last_n_bars if last_n_bars > 0 else 0
        downtrend_consistency = downtrend_count / last_n_bars if last_n_bars > 0 else 0
        
        # Проверка объема
        volume_increasing = current['volume'] > df['volume'].iloc[-5:].mean() * 1.1
        
        # Проверка сигнала на покупку с дополнительными фильтрами
        if (current['sma20'] > current['sma50'] and 
            previous['rsi'] < self.rsi_oversold and 
            current['rsi'] > self.rsi_oversold and
            strong_trend and
            uptrend_consistency >= 0.6 and  # Должен быть преимущественно восходящий тренд
            volume_increasing):            # Объем должен быть выше среднего
            return 1, f"EMA: восходящий тренд, RSI: выход из перепроданности ({current['rsi']:.2f}), ADX: {current['adx']:.2f}, Сила тренда: {uptrend_consistency:.2f}"
        
        # Проверка сигнала на продажу с дополнительными фильтрами
        elif (current['sma20'] < current['sma50'] and 
            previous['rsi'] > self.rsi_overbought and 
            current['rsi'] < self.rsi_overbought and
            strong_trend and
            downtrend_consistency >= 0.6 and  # Должен быть преимущественно нисходящий тренд
            volume_increasing):             # Объем должен быть выше среднего
            return -1, f"EMA: нисходящий тренд, RSI: выход из перекупленности ({current['rsi']:.2f}), ADX: {current['adx']:.2f}, Сила тренда: {downtrend_consistency:.2f}"
        
        return 0, "Нет сигнала"
  
def _calculate_sma(self, data, period):
       """Расчет простой скользящей средней"""
       return data.rolling(window=period).mean()
   
def _calculate_rsi(self, data, period=14):
       """Расчет индикатора относительной силы (RSI)"""
       delta = data.diff()
       
       # Отдельные серии для положительных и отрицательных изменений
       gain = delta.where(delta > 0, 0)
       loss = -delta.where(delta < 0, 0)
       
       # Расчет среднего выигрыша и средней потери
       avg_gain = gain.rolling(window=period).mean()
       avg_loss = loss.rolling(window=period).mean()
       
       # Расчет относительной силы
       rs = avg_gain / avg_loss
       
       # Расчет RSI
       rsi = 100 - (100 / (1 + rs))
       
       return rsi
   
def _calculate_adx(self, df, period=14):
       """Расчет индекса направленного движения (ADX)"""
       # Клонирование исходного DataFrame
       df = df.copy()
       
       # Вычисление True Range (TR)
       df['high_low'] = df['high'] - df['low']
       df['high_close'] = abs(df['high'] - df['close'].shift(1))
       df['low_close'] = abs(df['low'] - df['close'].shift(1))
       df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
       
       # Вычисление Directional Movement (DM)
       df['up_move'] = df['high'].diff()
       df['down_move'] = df['low'].diff()
       
       # Положительное и отрицательное направленное движение
       df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
       df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
       
       # Сглаживание TR, +DM, -DM
       df['tr_smoothed'] = df['tr'].rolling(window=period).sum()
       df['plus_dm_smoothed'] = df['plus_dm'].rolling(window=period).sum()
       df['minus_dm_smoothed'] = df['minus_dm'].rolling(window=period).sum()
       
       # Вычисление +DI и -DI
       df['plus_di'] = 100 * (df['plus_dm_smoothed'] / df['tr_smoothed'])
       df['minus_di'] = 100 * (df['minus_dm_smoothed'] / df['tr_smoothed'])
       
       # Вычисление DX
       df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
       
       # Вычисление ADX (сглаженный DX)
       df['adx'] = df['dx'].rolling(window=period).mean()
       
       # Удаление временных колонок
       df = df.drop(['high_low', 'high_close', 'low_close', 'up_move', 'down_move', 
                     'plus_dm', 'minus_dm', 'tr_smoothed', 'plus_dm_smoothed', 
                     'minus_dm_smoothed', 'dx'], axis=1, errors='ignore')
       
       return df
   
def _calculate_atr(self, df, period=14):
       """Расчет Average True Range (ATR)"""
       high = df['high']
       low = df['low']
       close = df['close']
       
       # Расчет True Range
       tr1 = high - low
       tr2 = abs(high - close.shift())
       tr3 = abs(low - close.shift())
       
       tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
       
       # Расчет ATR как скользящего среднего от TR
       atr = tr.rolling(window=period).mean()
       
       return atr

class BreakoutStrategy(Strategy):
   """Стратегия прорыва уровней поддержки/сопротивления"""
   def __init__(self, lookback=100, significance=3, volume_threshold=1.5, bb_period=20, bb_std=2):
       super().__init__("Breakout Strategy")
       self.lookback = lookback  # Период для поиска уровней
       self.significance = significance  # Минимальное количество касаний для уровня
       self.volume_threshold = volume_threshold  # Пороговое значение объема
       self.bb_period = bb_period  # Период для полос Боллинджера
       self.bb_std = bb_std  # Стандартное отклонение для полос Боллинджера
       
       # Для хранения уровней
       self.support_levels = []
       self.resistance_levels = []
   
   def calculate_indicators(self, df):
       """Расчет индикаторов для стратегии прорыва"""
       # Клонирование исходного DataFrame
       df = df.copy()
       
       # Расчет полос Боллинджера
       df['bb_middle'], df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(
           df['close'], period=self.bb_period, std_dev=self.bb_std
       )
       
       # Расчет OBV (On-Balance Volume)
       df['obv'] = self._calculate_obv(df)
       
       # Расчет ATR для стоп-лоссов
       df['atr'] = self._calculate_atr(df, 14)
       
       # Поиск уровней поддержки и сопротивления
       if len(df) >= self.lookback:
           self.support_levels, self.resistance_levels = self._find_support_resistance_levels(
               df.tail(self.lookback), significance=self.significance
           )
       
       return df
   
   def generate_signal(self, df):
       """Генерация сигналов для стратегии прорыва уровней"""
       if len(df) < self.lookback:
           return 0, "Недостаточно данных"
       
       # Получение последних значений
       current = df.iloc[-1]
       previous = df.iloc[-2]
       
       # Проверка пробоя уровня сопротивления
       for level in self.resistance_levels:
           if previous['close'] < level and current['close'] > level:
               # Проверка объема
               avg_volume = df['volume'].tail(20).mean()
               volume_increased = current['volume'] > avg_volume * self.volume_threshold
               
               # Проверка OBV (подтверждение движения объемом)
               obv_confirms = current['obv'] > previous['obv']
               
               # Проверка полос Боллинджера (пробой верхней полосы)
               bb_confirms = current['close'] > current['bb_upper']
               
               if volume_increased and obv_confirms and bb_confirms:
                   return 1, f"Пробой сопротивления {level:.2f} с увеличенным объемом, OBV подтверждает"
       
       # Проверка пробоя уровня поддержки
       for level in self.support_levels:
           if previous['close'] > level and current['close'] < level:
               # Проверка объема
               avg_volume = df['volume'].tail(20).mean()
               volume_increased = current['volume'] > avg_volume * self.volume_threshold
               
               # Проверка OBV (подтверждение движения объемом)
               obv_confirms = current['obv'] < previous['obv']
               
               # Проверка полос Боллинджера (пробой нижней полосы)
               bb_confirms = current['close'] < current['bb_lower']
               
               if volume_increased and obv_confirms and bb_confirms:
                   return -1, f"Пробой поддержки {level:.2f} с увеличенным объемом, OBV подтверждает"
       
       return 0, "Нет сигнала"
   
   def _calculate_bollinger_bands(self, data, period=20, std_dev=2):
       """Расчет полос Боллинджера"""
       # Средняя линия (SMA)
       middle_band = data.rolling(window=period).mean()
       
       # Стандартное отклонение
       rolling_std = data.rolling(window=period).std()
       
       # Верхняя и нижняя полосы
       upper_band = middle_band + (rolling_std * std_dev)
       lower_band = middle_band - (rolling_std * std_dev)
       
       return middle_band, upper_band, lower_band
   
   def _calculate_obv(self, df):
       """Расчет On-Balance Volume (OBV)"""
       obv = np.zeros(len(df))
       obv[0] = df['volume'].iloc[0]
       
       for i in range(1, len(df)):
           if df['close'].iloc[i] > df['close'].iloc[i-1]:
               obv[i] = obv[i-1] + df['volume'].iloc[i]
           elif df['close'].iloc[i] < df['close'].iloc[i-1]:
               obv[i] = obv[i-1] - df['volume'].iloc[i]
           else:
               obv[i] = obv[i-1]
       
       return pd.Series(obv, index=df.index)
   
   def _calculate_atr(self, df, period=14):
       """Расчет Average True Range (ATR)"""
       high = df['high']
       low = df['low']
       close = df['close']
       
       # Расчет True Range
       tr1 = high - low
       tr2 = abs(high - close.shift())
       tr3 = abs(low - close.shift())
       
       tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
       
       # Расчет ATR как скользящего среднего от TR
       atr = tr.rolling(window=period).mean()
       
       return atr
   
   def _find_support_resistance_levels(self, df, significance=3, threshold_percent=0.03):
       """Нахождение уровней поддержки и сопротивления"""
       # Находим локальные минимумы и максимумы
       local_min = []
       local_max = []
       
       for i in range(2, len(df) - 2):
           # Локальный минимум
           if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
               df['low'].iloc[i] < df['low'].iloc[i-2] and
               df['low'].iloc[i] < df['low'].iloc[i+1] and 
               df['low'].iloc[i] < df['low'].iloc[i+2]):
               local_min.append(df['low'].iloc[i])
           
           # Локальный максимум
           if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
               df['high'].iloc[i] > df['high'].iloc[i-2] and
               df['high'].iloc[i] > df['high'].iloc[i+1] and 
               df['high'].iloc[i] > df['high'].iloc[i+2]):
               local_max.append(df['high'].iloc[i])
       
       # Группируем близкие уровни
       def cluster_levels(levels, threshold):
           if not levels:
               return []
           
           levels = sorted(levels)
           clusters = []
           current_cluster = [levels[0]]
           
           for level in levels[1:]:
               if abs(level - current_cluster[0]) / current_cluster[0] <= threshold:
                   current_cluster.append(level)
               else:
                   if len(current_cluster) >= significance:
                       clusters.append(sum(current_cluster) / len(current_cluster))
                   current_cluster = [level]
           
           # Проверяем последний кластер
           if len(current_cluster) >= significance:
               clusters.append(sum(current_cluster) / len(current_cluster))
           
           return clusters
       
       # Получаем кластеризованные уровни
       support_levels = cluster_levels(local_min, threshold_percent)
       resistance_levels = cluster_levels(local_max, threshold_percent)
       
       # Сортируем по возрастанию
       support_levels.sort()
       resistance_levels.sort()
       
       return support_levels, resistance_levels

# Основная функция
# ЗАМЕНИТЬ весь код main() на:
def main():
    """Основная функция для запуска торгового бота"""
    symbol = "BTCUSDT"
    
    # Тестирование на разных таймфреймах
    timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    best_results = {}
    
    for timeframe in timeframes:
        print(f"\n\n==== Тестирование на таймфрейме {timeframe} ====")
        
        trader = FuturesTrader(
            symbol=symbol, 
            timeframe=timeframe,
            initial_balance=1000,
            leverage=15,           # Уменьшено плечо
            fee=0.0004,
            slippage=0.0005,
            risk_per_trade=0.005  # Уменьшен риск на сделку
        )
        
        # Создание стратегий с улучшенными параметрами
        trend_strategy = TrendFollowingStrategy(
            ema_short=10,         # Уменьшено
            ema_long=30,          # Уменьшено
            rsi_period=14,
            rsi_oversold=40,      # Более агрессивное значение
            rsi_overbought=60,    # Более агрессивное значение
            adx_period=14,
            adx_threshold=20      # Снижен порог
        )
        
        breakout_strategy = BreakoutStrategy(
            lookback=30,           # Уменьшен период
            significance=1,        # Снижен порог значимости
            volume_threshold=1.2,  # Снижен порог объема
            bb_period=20,
            bb_std=1.8
        )
        
        # Добавляем обе стратегии
        trader.add_strategy(trend_strategy)
        trader.add_strategy(breakout_strategy)
        
        # Загрузка исторических данных (последние 90 дней)
        df = trader.get_historical_data(days=180)
        
        if df is None or len(df) == 0:
            logging.error(f"Не удалось загрузить исторические данные для таймфрейма {timeframe}.")
            continue
        
        # Запуск бэктестинга
        results = trader.run_backtest(df)
        
        if results:
            # Вывод результатов
            stats = results['stats']
            final_balance = results['final_balance']
            profit_percent = ((final_balance / trader.initial_balance) - 1) * 100
            
            print(f"\nРезультаты бэктестинга для {symbol} (таймфрейм: {timeframe}):")
            print(f"Начальный баланс: {trader.initial_balance:.2f} USDT")
            print(f"Конечный баланс: {final_balance:.2f} USDT")
            print(f"Прибыль: {profit_percent:.2f}%")
            print(f"Всего сделок: {stats['total_trades']}")
            print(f"Процент выигрышных: {stats['win_rate']:.2f}%")
            print(f"Прибыль/Убыток: {stats['profit_factor']:.2f}")
            print(f"Коэффициент Шарпа: {stats['sharpe_ratio']:.2f}")
            print(f"Максимальная просадка: {stats['max_drawdown']:.2f}%")
            
            # Сохраняем результаты для сравнения
            best_results[timeframe] = {
                'profit': profit_percent,
                'win_rate': stats['win_rate'],
                'sharpe': stats['sharpe_ratio'],
                'drawdown': stats['max_drawdown'],
                'trades': stats['total_trades'],
                'final_balance': final_balance
            }
            
            # Визуализация результатов
            trader.plot_results(results)
            
            logging.info(f"Бэктестинг для {timeframe} завершен. Прибыль: {profit_percent:.2f}%, Конечный баланс: {final_balance:.2f} USDT")
        else:
            logging.error(f"Бэктестинг для {timeframe} не удался.")
    
    # Вывод сравнительных результатов
    if best_results:
        print("\n\n==== Сравнение результатов по таймфреймам ====")
        for tf, result in best_results.items():
            print(f"{tf}: Прибыль: {result['profit']:.2f}%, Win Rate: {result['win_rate']:.2f}%, Sharpe: {result['sharpe']:.2f}, Просадка: {result['drawdown']:.2f}%, Сделок: {result['trades']}")

if __name__ == "__main__":
   main()
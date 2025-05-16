"""
Модуль для бэктестинга торговых стратегий.
Позволяет тестировать стратегии на исторических данных и оценивать их эффективность.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import uuid
from tqdm import tqdm
import sys
import os

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

logger = logging.getLogger('backtest_engine')

class BacktestEngine:
    """Класс для бэктестинга торговых стратегий"""
    
    def __init__(self, config, data_provider, risk_manager):
        """
        Инициализация движка бэктестинга.
        
        Args:
            config (dict): Конфигурация бэктестинга
            data_provider (DataProvider): Провайдер данных
            risk_manager (RiskManager): Менеджер рисков
        """
        self.config = config
        self.data_provider = data_provider
        self.risk_manager = risk_manager
        
        # Параметры бэктестинга
        self.initial_balance = config.get('initial_balance', 10000)
        self.fee_rate = config.get('fee_rate', 0.0004)  # 0.04% комиссия
        self.start_date = config.get('start_date')
        self.end_date = config.get('end_date')
        
        # Состояние бэктестинга
        self.current_balance = self.initial_balance
        self.equity_curve = []
        self.trades = []
        self.open_positions = {}
        
        logger.info(f"Инициализирован движок бэктестинга с балансом {self.initial_balance} USD")
    
    def run_backtest(self, strategies, symbol, timeframes):
        """
        Запуск бэктестинга для списка стратегий.
        
        Args:
            strategies (list): Список стратегий
            symbol (str): Символ торговой пары
            timeframes (dict): Словарь таймфреймов для каждой стратегии
            
        Returns:
            dict: Результаты бэктестинга
        """
        logger.info(f"Запуск бэктестинга для {len(strategies)} стратегий на {symbol}")
        
        # Получение исторических данных для всех нужных таймфреймов
        data = {}
        unique_timeframes = set()
        
        # Сбор всех уникальных таймфреймов из всех стратегий
        for strategy in strategies:
            for tf in strategy.get_required_timeframes():
                unique_timeframes.add(tf)
        
        # Загрузка данных для каждого таймфрейма
        for tf in unique_timeframes:
            data[tf] = self.data_provider.get_complete_historical_data(
                symbol, tf, self.start_date, self.end_date
            )
            logger.info(f"Загружено {len(data[tf])} свечей для таймфрейма {tf}")
        
        # Определение базового таймфрейма для итерации (самый маленький таймфрейм)
        base_timeframe = min(unique_timeframes, key=lambda x: self._get_timeframe_minutes(x))
        base_data = data[base_timeframe]
        
        # Инициализация состояния бэктестинга
        self.current_balance = self.initial_balance
        self.equity_curve = []
        self.trades = []
        self.open_positions = {}
        
        # Добавление начальной точки в equity curve
        self.equity_curve.append({
            'timestamp': base_data.index[0],
            'balance': self.current_balance,
            'equity': self.current_balance
        })
        
        # Итерация по историческим данным
        logger.info("Начало бэктестинга...")
        
        # Используем tqdm для отображения прогресса
        for i in tqdm(range(len(base_data)), desc="Бэктестинг"):
            current_time = base_data.index[i]
            
            # Подготовка данных для каждого таймфрейма до текущего момента
            current_data = {}
            for tf, tf_data in data.items():
                # Фильтрация данных до текущего момента
                current_data[tf] = tf_data[tf_data.index <= current_time]
            
            # Обновление данных для всех стратегий
            for strategy in strategies:
                for tf in strategy.get_required_timeframes():
                    if tf in current_data:
                        strategy.update_data(tf, current_data[tf])
            
            # Обработка текущих открытых позиций
            self._process_open_positions(current_time, current_data, strategies, symbol)
            
            # Получение сигналов от всех стратегий
            signals_with_reasons = {}
            
            for strategy in strategies:
                # Получение основного таймфрейма стратегии
                main_tf = strategy.timeframes.get('entry', base_timeframe)
                
                # Проверка, есть ли достаточно данных
                if main_tf in current_data and len(current_data[main_tf]) > 0:
                    # Расчет индикаторов
                    data_with_indicators = strategy.calculate_indicators(current_data[main_tf])
                    
                    # Проверка сигнала на вход
                    signal, reason = strategy.generate_signal(data_with_indicators)
                    signals_with_reasons[strategy.name] = (signal, reason)
            
            # Комбинирование сигналов
            signal, signal_reason = self._combine_strategy_signals(signals_with_reasons)
            
            # Обработка сигнала, если он есть
            if signal != 0:
                self._process_entry_signal(signal, signal_reason, symbol, current_time, current_data[base_timeframe].iloc[-1]['close'], strategies)
            
            # Обновление equity curve на каждом шаге
            equity = self._calculate_current_equity(current_time, current_data[base_timeframe].iloc[-1]['close'], symbol)
            self.equity_curve.append({
                'timestamp': current_time,
                'balance': self.current_balance,
                'equity': equity
            })
        
        # Закрытие всех оставшихся позиций в конце бэктестинга
        logger.info("Закрытие оставшихся позиций...")
        latest_price = base_data.iloc[-1]['close']
        for position_id, position in list(self.open_positions.items()):
            self._close_position(position_id, latest_price, base_data.index[-1], "End of backtest")
        
        # Расчет статистики
        stats = self._calculate_backtest_statistics()
        
        logger.info(f"Бэктестинг завершен. Конечный баланс: {self.current_balance:.2f} USD")
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'profit_percent': ((self.current_balance / self.initial_balance) - 1) * 100,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'stats': stats
        }
    
    def _combine_strategy_signals(self, signals_with_reasons):
        """
        Взвешенное комбинирование сигналов от разных стратегий.
        
        Args:
            signals_with_reasons (dict): Словарь сигналов от разных стратегий
            
        Returns:
            tuple: (итоговый сигнал, причина)
        """
        if not signals_with_reasons:
            return 0, "Нет сигналов"
        
        # Веса для каждой стратегии
        strategy_weights = settings.STRATEGY_WEIGHTS
        
        # Взвешенное суммирование сигналов
        weighted_signal = 0
        all_reasons = []
        
        for strategy_name, (signal, reason) in signals_with_reasons.items():
            weight = strategy_weights.get(strategy_name, 0.5)
            weighted_signal += signal * weight
            if signal != 0:
                all_reasons.append(f"{strategy_name}: {reason}")
        
        # Определение итогового сигнала по порогу
        buy_threshold = settings.SIGNAL_THRESHOLDS.get('buy', 0.3)
        sell_threshold = settings.SIGNAL_THRESHOLDS.get('sell', -0.3)
        
        if weighted_signal >= buy_threshold:
            final_signal = 1
        elif weighted_signal <= sell_threshold:
            final_signal = -1
        else:
            final_signal = 0
        
        reason = ", ".join(all_reasons) if all_reasons else "Недостаточно сильный сигнал"
        
        return final_signal, reason
    
    def _process_entry_signal(self, signal, signal_reason, symbol, current_time, current_price, strategies):
        """
        Обработка сигнала на вход.
        
        Args:
            signal (int): Сигнал (1 для покупки, -1 для продажи)
            signal_reason (str): Причина сигнала
            symbol (str): Символ торговой пары
            current_time (datetime): Текущее время
            current_price (float): Текущая цена
            strategies (list): Список стратегий
        """
        # Подготовка сигнала для риск-менеджера
        signal_dict = {
            'type': 'buy' if signal == 1 else 'sell',
            'price': current_price,
            'symbol': symbol,
            'time': current_time
        }
        
        # Получаем волатильность (ATR)
        volatility = None
        for strategy in strategies:
            if hasattr(strategy, 'latest_data') and strategy.latest_data:
                for tf_data in strategy.latest_data.values():
                    if 'atr' in tf_data.columns:
                        volatility = tf_data['atr'].iloc[-1]
                        break
                if volatility is not None:
                    break
        
        # Проверка возможности открытия позиции
        if not self.risk_manager.can_open_position(signal_dict, self.current_balance, current_time):
            return
        
        # Расчет параметров позиции
        position_params = self.risk_manager.calculate_position_size(signal_dict, self.current_balance, current_price, volatility)
        
        # Генерация уникального ID позиции
        position_id = str(uuid.uuid4())
        
        # Создание позиции
        position = {
            'id': position_id,
            'symbol': symbol,
            'type': signal_dict['type'],  # 'buy' или 'sell'
            'entry_price': current_price,
            'quantity': position_params['quantity'],
            'leverage': position_params['leverage'],
            'stop_loss': position_params['stop_loss'],
            'take_profit': position_params['take_profit'],
            'open_time': current_time,
            'margin': position_params['position_size_usd'] / position_params['leverage'],
            'reason': signal_reason
        }
        
        # Расчет комиссии
        fee = position_params['position_size_usd'] * self.fee_rate
        
        # Вычитание комиссии из баланса
        self.current_balance -= fee
        
        # Регистрация позиции
        self.open_positions[position_id] = position
        
        logger.info(f"Открыта позиция {position_id}: {signal_dict['type']} {position_params['quantity']} {symbol} по цене {current_price} с плечом {position_params['leverage']}x")
    
    def _process_open_positions(self, current_time, current_data, strategies, symbol):
        """
        Обработка открытых позиций.
        
        Args:
            current_time (datetime): Текущее время
            current_data (dict): Данные для всех таймфреймов
            strategies (list): Список стратегий
            symbol (str): Символ торговой пары
        """
        # Если нет открытых позиций, ничего не делаем
        if not self.open_positions:
            return
        
        # Получение текущей цены
        base_tf = min(current_data.keys(), key=lambda x: self._get_timeframe_minutes(x))
        current_price = current_data[base_tf].iloc[-1]['close']
        
        # Проверка всех открытых позиций
        for position_id, position in list(self.open_positions.items()):
            # Поиск соответствующей стратегии по имени в причине позиции
            strategy_name = position['reason'].split(':')[0] if ':' in position['reason'] else None
            strategy = next((s for s in strategies if s.name == strategy_name), None)
            
            # Проверка на стоп-лосс и тейк-профит
            if position['stop_loss'] is not None:
                if (position['type'] == 'buy' and current_price <= position['stop_loss']) or \
                   (position['type'] == 'sell' and current_price >= position['stop_loss']):
                    self._close_position(position_id, current_price, current_time, "Stop Loss")
                    continue
            
            if position['take_profit'] is not None:
                if (position['type'] == 'buy' and current_price >= position['take_profit']) or \
                   (position['type'] == 'sell' and current_price <= position['take_profit']):
                    self._close_position(position_id, current_price, current_time, "Take Profit")
                    continue
            
            # Проверка сигнала на выход от стратегии (если есть стратегия)
            if strategy:
                # Получение основного таймфрейма стратегии
                main_tf = strategy.timeframes.get('entry', base_tf)
                
                if main_tf in current_data:
                    data_with_indicators = strategy.calculate_indicators(current_data[main_tf])
                    
                    # Проверка сигнала на выход
                    # Этот метод должен быть реализован в стратегии, если нужно особое условие выхода
                    if hasattr(strategy, 'check_exit_signal'):
                        exit_signal = strategy.check_exit_signal(data_with_indicators, position)
                        
                        if exit_signal:
                            self._close_position(position_id, current_price, current_time, exit_signal.get('reason', 'Strategy exit signal'))
    
    def _close_position(self, position_id, close_price, close_time, reason):
        """
        Закрытие позиции.
        
        Args:
            position_id (str): Идентификатор позиции
            close_price (float): Цена закрытия
            close_time (datetime): Время закрытия
            reason (str): Причина закрытия
        """
        if position_id not in self.open_positions:
            return
        
        position = self.open_positions[position_id]
        
        # Расчет PnL
        entry_price = position['entry_price']
        quantity = position['quantity']
        leverage = position['leverage']
        position_type = position['type']
        
        if position_type == 'buy':
            pnl = (close_price - entry_price) * quantity * leverage
        else:  # sell
            pnl = (entry_price - close_price) * quantity * leverage
        
        # Расчет комиссии
        position_value = close_price * quantity * leverage
        fee = position_value * self.fee_rate
        
        # Обновление баланса
        self.current_balance += pnl - fee
        
        # Запись информации о сделке
        trade = {
            'id': position_id,
            'symbol': position['symbol'],
            'type': position_type,
            'strategy': position['reason'].split(':')[0] if ':' in position['reason'] else 'Unknown',
            'entry_price': entry_price,
            'exit_price': close_price,
            'quantity': quantity,
            'leverage': leverage,
            'open_time': position['open_time'],
            'close_time': close_time,
            'duration': (close_time - position['open_time']).total_seconds() / 60,  # в минутах
            'pnl': pnl,
            'fee': fee,
            'net_pnl': pnl - fee,
            'pnl_percent': pnl / (position['margin']) * 100,  # % от начального капитала
            'reason': reason,
            'entry_reason': position['reason']
        }
        
        self.trades.append(trade)
        
        # Удаление позиции из открытых
        del self.open_positions[position_id]
        
        logger.info(f"Закрыта позиция {position_id}: PnL = {pnl:.2f} USD ({trade['pnl_percent']:.2f}%), причина: {reason}")
    
    def _calculate_current_equity(self, current_time, current_price, symbol):
        """
        Расчет текущего капитала (баланс + нереализованная прибыль/убыток).
        
        Args:
            current_time (datetime): Текущее время
            current_price (float): Текущая цена
            symbol (str): Символ торговой пары
            
        Returns:
            float: Текущий капитал
        """
        # Начинаем с текущего баланса
        equity = self.current_balance
        
        # Добавляем нереализованную прибыль/убыток от открытых позиций
        for position_id, position in self.open_positions.items():
            if position['symbol'] == symbol:
                entry_price = position['entry_price']
                quantity = position['quantity']
                leverage = position['leverage']
                position_type = position['type']
                
                if position_type == 'buy':
                    unrealized_pnl = (current_price - entry_price) * quantity * leverage
                else:  # sell
                    unrealized_pnl = (entry_price - current_price) * quantity * leverage
                
                equity += unrealized_pnl
        
        return equity
    
    def _calculate_backtest_statistics(self):
        """
        Расчет статистики бэктестинга.
        
        Returns:
            dict: Статистические показатели
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_profit': 0,
                'average_loss': 0,
                'max_drawdown': 0,
                'max_drawdown_percent': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'avg_trade': 0,
                'avg_bars': 0,
                'max_win': 0,
                'max_loss': 0
            }
        
        # Расчет основных метрик
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['net_pnl'] > 0]
        losing_trades = [t for t in self.trades if t['net_pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t['net_pnl'] for t in winning_trades)
        total_loss = abs(sum(t['net_pnl'] for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_profit = total_profit / len(winning_trades) if winning_trades else 0
        average_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        avg_trade = (total_profit - total_loss) / total_trades
        
        max_win = max([t['net_pnl'] for t in self.trades]) if self.trades else 0
        max_loss = min([t['net_pnl'] for t in self.trades]) if self.trades else 0
        
        # Среднее количество баров в сделке
        avg_bars = sum(t['duration'] for t in self.trades) / total_trades if total_trades > 0 else 0
        
        # Расчет максимальной просадки
        equity_array = np.array([point['equity'] for point in self.equity_curve])
        max_equity = np.maximum.accumulate(equity_array)
        drawdown = (max_equity - equity_array) / max_equity
        max_drawdown_percent = np.max(drawdown) * 100
        max_drawdown = np.max(max_equity - equity_array)
        
        # Расчет коэффициентов Шарпа и Сортино (упрощенно)
        returns = np.diff(equity_array) / equity_array[:-1]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Для Sortino Ratio используем только отрицательные доходности
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.0001
        
        sortino_ratio = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,  # в процентах
            'profit_factor': profit_factor,
            'average_profit': average_profit,
            'average_loss': average_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_percent,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'avg_trade': avg_trade,
            'avg_bars': avg_bars,
            'max_win': max_win,
            'max_loss': max_loss
        }
    
    def _get_timeframe_minutes(self, timeframe):
        """
        Преобразование строкового представления таймфрейма в минуты.
        
        Args:
            timeframe (str): Таймфрейм ('1m', '5m', '1h', etc.)
            
        Returns:
            int: Количество минут в таймфрейме
        """
        # Словарь соответствия таймфреймов и минут
        timeframe_dict = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '8h': 480,
            '12h': 720,
            '1d': 1440,
            '3d': 4320,
            '1w': 10080,
        }
        
        return timeframe_dict.get(timeframe, 60)  # По умолчанию 1 час
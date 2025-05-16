"""
Модуль для управления рисками.
Отвечает за расчет размера позиции, контроль рисков и мониторинг общего риска портфеля.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('risk_manager')

class RiskManager:
    """Класс для управления рисками торговой системы"""
    
    def __init__(self, config):
        """
        Инициализация менеджера рисков.
        
        Args:
            config (dict): Конфигурация риск-менеджмента
        """
        self.max_position_size = config.get('max_position_size', 0.2)  # Макс. размер позиции (% от баланса)
        self.max_positions = config.get('max_positions', 3)  # Макс. количество одновременных позиций
        self.risk_per_trade = config.get('risk_per_trade', 0.005)  # Риск на сделку (% от баланса)
        self.default_leverage = config.get('default_leverage', 3)  # Стандартное плечо
        self.max_leverage = config.get('max_leverage', 5)  # Максимальное плечо
        self.max_daily_drawdown = config.get('max_daily_drawdown', 0.05)  # Макс. дневная просадка (% от баланса)
        
        # Текущие открытые позиции
        self.open_positions = {}
        
        # История торговли
        self.trade_history = []
        
        logger.info(f"Инициализирован риск-менеджер с параметрами: {config}")
    
    def calculate_position_size(self, signal, balance, current_price, volatility=None):
        """
        Расчет размера позиции на основе параметров сигнала и баланса.
        
        Args:
            signal (dict): Сигнал на вход
            balance (float): Текущий баланс
            current_price (float): Текущая цена
            volatility (float, optional): Показатель волатильности
            
        Returns:
            dict: Параметры позиции
        """
        # Извлечение параметров сигнала
        entry_price = signal.get('price', current_price)
        stop_loss = signal.get('stop_loss')
        leverage = signal.get('leverage', self.default_leverage)
        
        # Если стоп-лосс не указан, используем дефолтное значение
        if stop_loss is None:
            if 'atr' in signal and volatility is None:
                volatility = signal['atr']
            
            # Адаптивный стоп-лосс в зависимости от волатильности
            if volatility:
                sl_percent = max(0.5, min(2.0, volatility * 2))  # 0.5% - 2%
            else:
                sl_percent = 1.0  # Дефолтное значение
                
            if signal['type'] == 'buy':
                stop_loss = entry_price * (1 - sl_percent/100)
            else:  # sell
                stop_loss = entry_price * (1 + sl_percent/100)
        
        # Расчет риска в пунктах
        risk_points = abs(entry_price - stop_loss)
        
        # Расчет риска в процентах от цены входа
        risk_percent = risk_points / entry_price
        
        # Определение суммы риска в USD
        risk_amount = balance * self.risk_per_trade
        
        # Расчет размера позиции без учета плеча (в USD)
        position_size_usd = risk_amount / risk_percent
        
        # Расчет размера позиции с учетом плеча
        position_size_with_leverage = position_size_usd * leverage
        
        # Ограничение размера позиции
        max_position_size = balance * self.max_position_size * leverage
        position_size_with_leverage = min(position_size_with_leverage, max_position_size)
        
        # Расчет количества контрактов/монет
        quantity = position_size_with_leverage / entry_price
        
        # Округление для избежания ошибок с точностью
        quantity = self._round_quantity(quantity, entry_price)
        
        # Расчет тейк-профита (если не указан)
        take_profit = signal.get('take_profit')
        if take_profit is None:
            # Соотношение риск/прибыль
            if volatility and volatility > 2.0:  # Высокая волатильность
                risk_reward_ratio = 1.5  # Более консервативное для высокой волатильности
            else:
                risk_reward_ratio = 2.0  # Стандартное
                
            if signal['type'] == 'buy':
                take_profit = entry_price + risk_points * risk_reward_ratio
            else:  # sell
                take_profit = entry_price - risk_points * risk_reward_ratio
        
        return {
            'quantity': quantity,
            'leverage': leverage,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'position_size_usd': position_size_with_leverage
        }
    
    def _round_quantity(self, quantity, price):
        """
        Округление количества в соответствии с правилами биржи.
        
        Args:
            quantity (float): Количество
            price (float): Цена
            
        Returns:
            float: Округленное количество
        """
        # Определение шага в зависимости от цены (упрощенно)
        if price < 0.1:
            step = 1
        elif price < 1:
            step = 0.1
        elif price < 10:
            step = 0.01
        elif price < 100:
            step = 0.001
        elif price < 1000:
            step = 0.0001
        else:
            step = 0.00001
        
        # Округление до ближайшего шага
        return round(quantity / step) * step
    
    def can_open_position(self, signal, balance, current_time=None):
        """
        Проверка возможности открытия новой позиции.
        
        Args:
            signal (dict): Сигнал на вход
            balance (float): Текущий баланс
            current_time (datetime, optional): Текущее время
            
        Returns:
            bool: True, если можно открыть новую позицию, иначе False
        """
        # Проверка количества открытых позиций
        if len(self.open_positions) >= self.max_positions:
            logger.warning("Достигнуто максимальное количество одновременных позиций")
            return False
        
        # Проверка дневной просадки
        daily_drawdown = self.calculate_daily_drawdown(current_time)
        if daily_drawdown >= self.max_daily_drawdown * balance:
            logger.warning(f"Достигнута максимальная дневная просадка: {daily_drawdown:.2f} USD")
            return False
        
        # Проверка на противоположные сигналы
        symbol = signal.get('symbol')
        if symbol:
            for position_id, position in self.open_positions.items():
                if position['symbol'] == symbol and position['type'] != signal['type']:
                    logger.warning(f"Уже есть открытая позиция в противоположном направлении для {symbol}")
                    return False
        
        return True
    
    def calculate_daily_drawdown(self, current_time=None):
        """
        Расчет дневной просадки.
        
        Args:
            current_time (datetime, optional): Текущее время
            
        Returns:
            float: Сумма просадки за день
        """
        if not current_time:
            current_time = datetime.now()
            
        today = current_time.date()
        daily_trades = [trade for trade in self.trade_history 
                      if trade['close_time'].date() == today and trade['pnl'] < 0]
        
        return abs(sum(trade['pnl'] for trade in daily_trades))
    
    def register_position(self, position_id, position_data):
        """
        Регистрация новой открытой позиции.
        
        Args:
            position_id (str): Идентификатор позиции
            position_data (dict): Данные позиции
        """
        self.open_positions[position_id] = position_data
        logger.info(f"Зарегистрирована новая позиция: {position_id}")
    
    def close_position(self, position_id, close_price, close_time=None):
        """
        Закрытие позиции и запись результата в историю торговли.
        
        Args:
            position_id (str): Идентификатор позиции
            close_price (float): Цена закрытия
            close_time (datetime, optional): Время закрытия
            
        Returns:
            dict: Информация о закрытой позиции
        """
        if position_id not in self.open_positions:
            logger.warning(f"Позиция {position_id} не найдена")
            return None
        
        position = self.open_positions[position_id]
        close_time = close_time or datetime.now()
        
        # Расчет прибыли/убытка
        entry_price = position['entry_price']
        quantity = position['quantity']
        leverage = position['leverage']
        position_type = position['type']
        
        if position_type == 'buy':
            pnl = (close_price - entry_price) * quantity * leverage
        else:  # sell
            pnl = (entry_price - close_price) * quantity * leverage
        
        # Запись в историю торговли
        trade_result = {
            'position_id': position_id,
            'symbol': position['symbol'],
            'type': position_type,
            'entry_price': entry_price,
            'close_price': close_price,
            'quantity': quantity,
            'leverage': leverage,
            'open_time': position['open_time'],
            'close_time': close_time,
            'duration': (close_time - position['open_time']).total_seconds() / 60,  # в минутах
            'pnl': pnl,
            'pnl_percent': pnl / (entry_price * quantity / leverage) * 100,  # % от начального капитала
            'reason': position.get('close_reason', 'Manual close')
        }
        
        self.trade_history.append(trade_result)
        
        # Удаление из списка открытых позиций
        del self.open_positions[position_id]
        
        logger.info(f"Закрыта позиция {position_id}: PnL = {pnl:.2f} USD ({trade_result['pnl_percent']:.2f}%)")
        
        return trade_result
    
    def get_portfolio_stats(self, balance):
        """
        Получение статистики текущего портфеля.
        
        Args:
            balance (float): Текущий баланс
            
        Returns:
            dict: Статистика портфеля
        """
        total_margin_used = 0
        total_position_value = 0
        
        for position_id, position in self.open_positions.items():
            position_value = position['entry_price'] * position['quantity']
            margin_used = position_value / position['leverage']
            
            total_position_value += position_value
            total_margin_used += margin_used
        
        return {
            'positions_count': len(self.open_positions),
            'total_position_value': total_position_value,
            'total_margin_used': total_margin_used,
            'margin_usage_percent': (total_margin_used / balance) * 100 if balance > 0 else 0,
            'free_margin': balance - total_margin_used,
            'free_margin_percent': ((balance - total_margin_used) / balance) * 100 if balance > 0 else 0
        }
    
    def dynamic_leverage_adjustment(self, market_conditions):
        """
        Динамическая корректировка плеча в зависимости от рыночных условий.
        
        Args:
            market_conditions (dict): Информация о текущих рыночных условиях
            
        Returns:
            float: Рекомендуемое плечо
        """
        base_leverage = self.default_leverage
        
        # Корректировка на основе волатильности
        volatility = market_conditions.get('volatility', 1.0)  # Нормализованная волатильность (1.0 = средняя)
        leverage_volatility_factor = 1.0 / volatility if volatility > 0 else 1.0
        
        # Корректировка на основе тренда
        trend_strength = market_conditions.get('trend_strength', 0.5)  # 0-1, где 1 = сильный тренд
        leverage_trend_factor = 0.5 + trend_strength  # 0.5-1.5
        
        # Корректировка на основе объема
        volume_factor = market_conditions.get('volume_ratio', 1.0)  # Отношение текущего объема к среднему
        leverage_volume_factor = min(1.5, volume_factor) if volume_factor > 1.0 else 0.8
        
        # Расчет итогового плеча
        adjusted_leverage = base_leverage * leverage_volatility_factor * leverage_trend_factor * leverage_volume_factor
        
        # Ограничение максимальным и минимальным значениями
        adjusted_leverage = max(1.0, min(self.max_leverage, adjusted_leverage))
        
        return round(adjusted_leverage, 1)  # Округление до одного десятичного знака
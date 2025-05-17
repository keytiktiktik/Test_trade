# simple_fix_partial_close.py
"""
Простой скрипт для исправления проблемы с частичным закрытием позиций.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Настройка логирования
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/simple_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Импорт модулей системы
from config.settings import (
    EXCHANGE_CONFIG, BACKTEST_CONFIG, RISK_CONFIG,
    TREND_STRATEGY_CONFIG, BREAKOUT_STRATEGY_CONFIG
)
from data.data_provider import DataProvider
from risk_management.risk_manager import RiskManager
from backtest.backtest_engine import BacktestEngine
from strategies.trend_following import TrendFollowingStrategy

def fix_backtest_engine():
    """Вносит необходимые исправления в класс BacktestEngine"""
    # Сохраняем оригинальный метод инициализации
    original_init = BacktestEngine.__init__
    
    # Определяем новый метод инициализации с явными параметрами частичного закрытия
    def fixed_init(self, config, data_provider, risk_manager):
        # Вызываем оригинальный метод инициализации
        original_init(self, config, data_provider, risk_manager)
        
        # Явно добавляем параметры частичного закрытия
        self.partial_profit_target = config.get('partial_profit_target', 0.1)  # 10% по умолчанию
        self.partial_close_percent = config.get('partial_close_percent', 0.5)  # 50% по умолчанию
        self.move_stop_to_breakeven = config.get('move_stop_to_breakeven', True)
        
        logging.info(f"Инициализированы параметры частичного закрытия: "
                    f"partial_profit_target={self.partial_profit_target}, "
                    f"partial_close_percent={self.partial_close_percent}, "
                    f"move_stop_to_breakeven={self.move_stop_to_breakeven}")
    
    # Заменяем метод инициализации
    BacktestEngine.__init__ = fixed_init
    
    # Сохраняем оригинальный метод обработки открытых позиций
    original_process = BacktestEngine._process_open_positions
    
    # Определяем новый метод обработки открытых позиций
    def fixed_process_open_positions(self, current_time, current_data, strategies, symbol):
        """Исправленная версия метода обработки открытых позиций"""
        # Если нет открытых позиций, ничего не делаем
        if not self.open_positions:
            return
        
        # Получение текущей цены
        base_tf = min(current_data.keys(), key=lambda x: self._get_timeframe_minutes(x))
        current_price = current_data[base_tf].iloc[-1]['close']
        
        # Проверка всех открытых позиций
        for position_id, position in list(self.open_positions.items()):
            # Расчет текущей прибыли/убытка в процентах
            entry_price = position['entry_price']
            position_type = position['type']
            
            if position_type == 'buy':
                price_change_percent = (current_price - entry_price) / entry_price * 100
            else:  # sell
                price_change_percent = (entry_price - current_price) / entry_price * 100
                
            logging.debug(f"Позиция {position_id}: {position_type}, изменение={price_change_percent:.2f}%, "
                         f"порог={getattr(self, 'partial_profit_target', 0.1)*100:.1f}%, "
                         f"partial_close_done={position.get('partial_close_done', False)}")
            
            # Проверка на частичное закрытие позиции
            if hasattr(self, 'partial_profit_target') and hasattr(self, 'partial_close_percent'):
                if (not position.get('partial_close_done', False) and 
                    price_change_percent >= self.partial_profit_target * 100):
                    
                    logging.info(f"PARTIAL CLOSE TRIGGERED: Позиция {position_id}: "
                                f"тип={position_type}, "
                                f"изменение={price_change_percent:.2f}%, "
                                f"порог={self.partial_profit_target*100:.1f}%")
                    
                    # Вычисление размера для частичного закрытия
                    close_quantity = position['quantity'] * self.partial_close_percent
                    
                    # Частичное закрытие позиции
                    self._partial_close_position(position_id, close_quantity, current_price, current_time, 
                                               f"Partial Take Profit {self.partial_profit_target*100:.0f}%")
                    
                    # Обновление флага частичного закрытия
                    position['partial_close_done'] = True
                    
                    # Перемещение стопа в безубыток, если это настроено
                    if hasattr(self, 'move_stop_to_breakeven') and self.move_stop_to_breakeven:
                        if not position.get('breakeven_done', False):
                            position['stop_loss'] = position['entry_price']
                            position['breakeven_done'] = True
                            logging.info(f"Стоп-лосс для позиции {position_id} перемещен в безубыток ({position['entry_price']})")
            
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
                    if hasattr(strategy, 'check_exit_signal'):
                        exit_signal = strategy.check_exit_signal(data_with_indicators, position)
                        
                        if exit_signal:
                            self._close_position(position_id, current_price, current_time, exit_signal.get('reason', 'Strategy exit signal'))
    
    # Заменяем метод обработки открытых позиций
    BacktestEngine._process_open_positions = fixed_process_open_positions
    
    logging.info("Исправления успешно внесены в класс BacktestEngine")

def run_test_with_fixes():
    """Запускает бэктест с исправлениями"""
    # Применяем исправления
    fix_backtest_engine()
    
    # Настройка параметров
    symbol = 'BTCUSDT'
    timeframe = '1h'
    
    # Настройка параметров бэктеста для использования длинного исторического периода
    BACKTEST_CONFIG['start_date'] = '2020-01-01'  # Начало 2020 года
    BACKTEST_CONFIG['end_date'] = '2023-12-31'    # Конец 2023 года
    
    # Настройка частичного закрытия с более низким порогом
    BACKTEST_CONFIG['partial_profit_target'] = 0.03  # 3% для частичного закрытия
    BACKTEST_CONFIG['partial_close_percent'] = 0.5   # Закрытие 50% позиции
    BACKTEST_CONFIG['move_stop_to_breakeven'] = True
    
    print(f"Запуск бэктеста для {symbol} {timeframe} с исправлениями")
    print(f"Период: {BACKTEST_CONFIG['start_date']} - {BACKTEST_CONFIG['end_date']}")
    print(f"Порог частичного закрытия: {BACKTEST_CONFIG['partial_profit_target']*100}%")
    
    # Инициализация компонентов
    data_provider = DataProvider(EXCHANGE_CONFIG)
    risk_manager = RiskManager(RISK_CONFIG)
    backtest_engine = BacktestEngine(BACKTEST_CONFIG, data_provider, risk_manager)
    
    # Проверка, что параметры корректно инициализированы
    logging.info(f"Проверка инициализации параметров в backtest_engine:")
    logging.info(f"  partial_profit_target = {getattr(backtest_engine, 'partial_profit_target', 'NOT FOUND')}")
    logging.info(f"  partial_close_percent = {getattr(backtest_engine, 'partial_close_percent', 'NOT FOUND')}")
    logging.info(f"  move_stop_to_breakeven = {getattr(backtest_engine, 'move_stop_to_breakeven', 'NOT FOUND')}")
    
    # Создание стратегии
    trend_strategy = TrendFollowingStrategy(TREND_STRATEGY_CONFIG)
    active_strategies = [trend_strategy]
    
    # Запуск бэктеста
    tf_dict = {strategy.name: strategy.get_required_timeframes() for strategy in active_strategies}
    results = backtest_engine.run_backtest(active_strategies, symbol, tf_dict)
    
    # Проверка результатов
    print("\n=== Результаты бэктеста ===")
    print(f"Прибыль: {results['profit_percent']:.2f}%")
    print(f"Всего сделок: {results['stats']['total_trades']}")
    
    # Проверка частичных закрытий
    partial_trades = [t for t in results['trades'] if t.get('partial_close', False)]
    print(f"Частичных закрытий: {len(partial_trades)}")
    
    if partial_trades:
        print("\nДетали частичных закрытий:")
        for i, trade in enumerate(partial_trades):
            print(f"{i+1}. {trade['symbol']} {trade['type']} - "
                  f"Вход: {trade['entry_price']:.2f}, Выход: {trade['exit_price']:.2f}, "
                  f"PnL: {trade['net_pnl']:.2f} ({trade['pnl_percent']:.2f}%)")
    
    return results

if __name__ == "__main__":
    run_test_with_fixes()
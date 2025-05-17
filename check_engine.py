# test_fixed_engine.py
"""
Тест для проверки исправленного класса BacktestEngine.
"""

import logging
from config.settings import (
    EXCHANGE_CONFIG, BACKTEST_CONFIG, RISK_CONFIG,
    TREND_STRATEGY_CONFIG
)
from data.data_provider import DataProvider
from risk_management.risk_manager import RiskManager
from backtest.backtest_engine import BacktestEngine
from strategies.trend_following import TrendFollowingStrategy

# Настройка логирования
logging.basicConfig(level=logging.INFO)



def test_engine():
    """Тестирование работы исправленного класса BacktestEngine"""
    print("Тестирование BacktestEngine с частичным закрытием позиций")
    
    # Обновляем параметры
    BACKTEST_CONFIG['partial_profit_target'] = 0.02  # 3% для частичного закрытия
    BACKTEST_CONFIG['partial_close_percent'] = 0.5   # Закрытие 50% позиции
    BACKTEST_CONFIG['move_stop_to_breakeven'] = True
    
    # Настраиваем тестовый период
    BACKTEST_CONFIG['start_date'] = '2023-01-01'
    BACKTEST_CONFIG['end_date'] = '2023-12-31'
    
    
    print(f"Параметры частичного закрытия: порог={BACKTEST_CONFIG['partial_profit_target']*100}%, "
          f"размер закрытия={BACKTEST_CONFIG['partial_close_percent']*100}%")
    
    # Инициализация компонентов
    data_provider = DataProvider(EXCHANGE_CONFIG)
    risk_manager = RiskManager(RISK_CONFIG)
    backtest_engine = BacktestEngine(BACKTEST_CONFIG, data_provider, risk_manager)
    
    # Проверка инициализации
    print(f"Движок создан: {backtest_engine is not None}")
    print(f"Параметры частичного закрытия инициализированы:")
    print(f"  - partial_profit_target = {backtest_engine.partial_profit_target}")
    print(f"  - partial_close_percent = {backtest_engine.partial_close_percent}")
    print(f"  - move_stop_to_breakeven = {backtest_engine.move_stop_to_breakeven}")
    
    # Проверка методов
    print("\nМетоды класса BacktestEngine:")
    for method_name in ['run_backtest', '_process_open_positions', '_partial_close_position', '_close_position']:
        has_method = hasattr(backtest_engine, method_name)
        print(f"  - {method_name}: {'НАЙДЕН' if has_method else 'НЕ НАЙДЕН'}")
    
    # Запуск короткого теста
    try:
        print("\nЗапуск короткого бэктеста...")
        
        # Создание стратегии
        trend_strategy = TrendFollowingStrategy(TREND_STRATEGY_CONFIG)
        
        # Запуск бэктеста на коротком периоде
        BACKTEST_CONFIG['start_date'] = '2023-01-01'
        BACKTEST_CONFIG['end_date'] = '2023-01-31'  # Только январь 2023
        
        tf_dict = {trend_strategy.name: trend_strategy.get_required_timeframes()}
        results = backtest_engine.run_backtest([trend_strategy], 'BTCUSDT', tf_dict)
        
        # Проверка результатов
        print(f"\nРезультаты бэктеста:")
        print(f"  - Прибыль: {results['profit_percent']:.2f}%")
        print(f"  - Всего сделок: {results['stats']['total_trades']}")
        
        # Проверка частичных закрытий
        partial_trades = [t for t in results['trades'] if t.get('partial_close', False)]
        print(f"  - Частичных закрытий: {len(partial_trades)}")
        
        print("\nТестирование завершено успешно!")
        
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")

if __name__ == "__main__":
    test_engine()
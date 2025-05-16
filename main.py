"""
Основной файл для запуска торговой системы.
Позволяет запускать систему в режиме бэктестинга или реальной торговли.
"""

import os
import logging
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import sys

# Добавление текущей директории в путь для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорт модулей системы
from config.settings import (
    EXCHANGE_CONFIG, TRADING_CONFIG, BACKTEST_CONFIG, 
    RISK_CONFIG, TREND_STRATEGY_CONFIG, BREAKOUT_STRATEGY_CONFIG,
    HYBRID_STRATEGY_CONFIG
)
from data.data_provider import DataProvider
from risk_management.risk_manager import RiskManager
from backtest.backtest_engine import BacktestEngine
from visualization.visualizer import Visualizer
from strategies.trend_following import TrendFollowingStrategy
from strategies.breakout import BreakoutStrategy
from strategies.hybrid import MomentumMeanReversionStrategy

# Настройка логирования
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/trading_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('main')

def run_backtest(strategies_to_use, timeframes):
    """
    Запуск бэктестинга для выбранных стратегий и таймфреймов.
    
    Args:
        strategies_to_use (list): Список имен стратегий для использования
        timeframes (list): Список таймфреймов для тестирования
        
    Returns:
        dict: Результаты бэктестинга
    """
    results = {}
    
    # Тестирование на каждом таймфрейме
    for timeframe in timeframes:
        logger.info(f"Запуск бэктестинга для таймфрейма: {timeframe}")
        
        # Обновление конфигурации для текущего таймфрейма
        TRADING_CONFIG['timeframe'] = timeframe
        
        # Инициализация провайдера данных
        data_provider = DataProvider(EXCHANGE_CONFIG)
        
        # Инициализация риск-менеджера
        risk_manager = RiskManager(RISK_CONFIG)
        
        # Инициализация движка бэктестинга
        backtest_engine = BacktestEngine(BACKTEST_CONFIG, data_provider, risk_manager)
        
        # Создание экземпляров стратегий
        active_strategies = []
        
        if 'trend_following' in strategies_to_use:
            trend_strategy = TrendFollowingStrategy(TREND_STRATEGY_CONFIG)
            if trend_strategy:
                active_strategies.append(trend_strategy)
        
        if 'breakout' in strategies_to_use:
            breakout_strategy = BreakoutStrategy(BREAKOUT_STRATEGY_CONFIG)
            if breakout_strategy:
                active_strategies.append(breakout_strategy)
        
        if 'hybrid' in strategies_to_use:
            hybrid_strategy = MomentumMeanReversionStrategy(HYBRID_STRATEGY_CONFIG)
            if hybrid_strategy:
                active_strategies.append(hybrid_strategy)
        
        if not active_strategies:
            logger.error("Не удалось создать ни одной стратегии")
            continue
        
        # Запуск бэктестинга
        symbol = TRADING_CONFIG['symbol']
        tf_dict = {strategy.name: strategy.get_required_timeframes() for strategy in active_strategies}
        
        # Запуск бэктестинга для текущего таймфрейма
        result = backtest_engine.run_backtest(active_strategies, symbol, tf_dict)
        
        if result:
            results[timeframe] = result
            
            # Визуализация результатов
            visualize_results(result, timeframe)
    
    # Сравнение результатов по таймфреймам
    compare_timeframe_results(results)
    
    return results

def visualize_results(results, timeframe):
    """
    Визуализация результатов бэктестинга.
    
    Args:
        results (dict): Результаты бэктестинга
        timeframe (str): Таймфрейм
    """
    if not results:
        logger.error("Нет результатов для визуализации")
        return
    
    # Создание директории для сохранения графиков
    output_dir = f"results/{timeframe}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Инициализация визуализатора
    visualizer = Visualizer(save_path=output_dir)
    
    # Построение графиков
    equity_fig = visualizer.plot_equity_curve(results['equity_curve'], title=f"Equity Curve - {timeframe}")
    drawdown_fig = visualizer.plot_drawdown(results['equity_curve'], title=f"Drawdown - {timeframe}")
    stats_fig = visualizer.create_trade_statistics_table(results)
    
    # Сохранение результатов в JSON
    with open(f"{output_dir}/backtest_results.json", 'w') as f:
        # Преобразование datetime объектов в строки
        results_copy = results.copy()
        
        # Обработка equity_curve
        for point in results_copy['equity_curve']:
            if isinstance(point['timestamp'], datetime):
                point['timestamp'] = point['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Обработка trades
        for trade in results_copy['trades']:
            if isinstance(trade['open_time'], datetime):
                trade['open_time'] = trade['open_time'].strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(trade['close_time'], datetime):
                trade['close_time'] = trade['close_time'].strftime('%Y-%m-%d %H:%M:%S')
        
        json.dump(results_copy, f, indent=4)
    
    logger.info(f"Результаты для таймфрейма {timeframe} сохранены в директории: {output_dir}")

def compare_timeframe_results(results):
    """
    Сравнение результатов по разным таймфреймам.
    
    Args:
        results (dict): Словарь с результатами по таймфреймам
    """
    if not results:
        logger.error("Нет результатов для сравнения")
        return
    
    comparison = {}
    
    for timeframe, result in results.items():
        stats = result['stats']
        final_balance = result['final_balance']
        initial_balance = result['initial_balance']
        profit_percent = ((final_balance / initial_balance) - 1) * 100
        
        comparison[timeframe] = {
            'profit': profit_percent,
            'win_rate': stats['win_rate'],
            'sharpe': stats['sharpe_ratio'],
            'drawdown': stats['max_drawdown_percent'],
            'trades': stats['total_trades'],
            'final_balance': final_balance
        }
    
    # Вывод сравнительных результатов
    print("\n==== Сравнение результатов по таймфреймам ====")
    for tf, result in comparison.items():
        print(f"{tf}: Прибыль: {result['profit']:.2f}%, Win Rate: {result['win_rate']:.2f}%, Sharpe: {result['sharpe']:.2f}, Просадка: {result['drawdown']:.2f}%, Сделок: {result['trades']}")
    
    # Сохранение сравнительных результатов
    output_dir = f"results/comparison"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(f"{output_dir}/timeframe_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(comparison, f, indent=4)
    
    logger.info(f"Сравнительные результаты сохранены в директории: {output_dir}")

def main():
    """Основная функция для запуска системы"""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Торговая система для фьючерсов с плечом')
    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest',
                       help='Режим работы системы (backtest или live)')
    parser.add_argument('--strategies', nargs='+', 
                       choices=['trend_following', 'breakout', 'hybrid', 'all'], 
                       default=['all'],
                       help='Стратегии для использования')
    parser.add_argument('--timeframes', nargs='+',
                       choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d', 'all'],
                       default=['4h', '1d'],
                       help='Таймфреймы для тестирования')
    
    args = parser.parse_args()
    
    # Определение списка стратегий
    if 'all' in args.strategies:
        strategies_to_use = ['trend_following', 'breakout', 'hybrid']
    else:
        strategies_to_use = args.strategies
    
    # Определение списка таймфреймов
    if 'all' in args.timeframes:
        timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
    else:
        timeframes = args.timeframes
    
    if args.mode == 'backtest':
        logger.info(f"Запуск бэктестинга для стратегий: {strategies_to_use} на таймфреймах: {timeframes}")
        results = run_backtest(strategies_to_use, timeframes)
    else:
        logger.error("Режим live торговли пока не реализован")

if __name__ == "__main__":
    main()
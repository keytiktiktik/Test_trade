"""
Тестовый скрипт для демонстрации работы частичного закрытия позиций.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt # type: ignore

def run_backtest_with_params(params):
    # Распаковка параметров
    profit_target, close_percent, move_to_breakeven = params
    # Запуск бэктеста и возврат результатов
    return run_test_with_partial_close(profit_target, close_percent, move_to_breakeven)

param_combinations = [
    (0.99, 0, False),      # Стандартный
    (0.1, 0.5, True),      # 10% прибыль, 50% закрытие
    (0.05, 0.5, True),     # 5% прибыль, 50% закрытие
    (0.1, 0.75, True),     # 10% прибыль, 75% закрытие
]


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
        logging.FileHandler(f"{log_dir}/partial_close_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('partial_close_test')

def run_test_with_partial_close(partial_profit_target=0.1, partial_close_percent=0.5, move_stop_to_breakeven=True):
    """
    Запуск теста с частичным закрытием позиций.
    
    Args:
        partial_profit_target (float): Цель прибыли для частичного закрытия (в долях)
        partial_close_percent (float): Процент позиции для закрытия (в долях)
        move_stop_to_breakeven (bool): Перевод стопа в безубыток
        
    Returns:
        dict: Результаты бэктестинга
    """
    logger.info(f"Запуск теста с параметрами: partial_profit_target={partial_profit_target}, "
                f"partial_close_percent={partial_close_percent}, move_stop_to_breakeven={move_stop_to_breakeven}")
    
    # Обновляем настройки бэктестинга
    BACKTEST_CONFIG['partial_profit_target'] = partial_profit_target
    BACKTEST_CONFIG['partial_close_percent'] = partial_close_percent
    BACKTEST_CONFIG['move_stop_to_breakeven'] = move_stop_to_breakeven
    
    # Настраиваем время для теста
    BACKTEST_CONFIG['start_date'] = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    BACKTEST_CONFIG['end_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Инициализация провайдера данных
    data_provider = DataProvider(EXCHANGE_CONFIG)
    
    # Инициализация риск-менеджера
    risk_manager = RiskManager(RISK_CONFIG)
    
    # Инициализация движка бэктестинга
    backtest_engine = BacktestEngine(BACKTEST_CONFIG, data_provider, risk_manager)
    
    # Создание экземпляров стратегий
    active_strategies = []
    
    # Включаем все стратегии для более полного тестирования
    trend_strategy = TrendFollowingStrategy(TREND_STRATEGY_CONFIG)
    breakout_strategy = BreakoutStrategy(BREAKOUT_STRATEGY_CONFIG)
    hybrid_strategy = MomentumMeanReversionStrategy(HYBRID_STRATEGY_CONFIG)
    
    active_strategies.extend([trend_strategy, breakout_strategy, hybrid_strategy])
    
    # Запуск бэктестинга
    symbol = TRADING_CONFIG['symbol']
    tf_dict = {strategy.name: strategy.get_required_timeframes() for strategy in active_strategies}
    
    # Запуск бэктестинга
    results = backtest_engine.run_backtest(active_strategies, symbol, tf_dict)
    
    # Создание директории для сохранения графиков
    output_dir = f"results/partial_close_test/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Визуализация результатов
    visualizer = Visualizer(save_path=output_dir)
    
    # Получение данных для последнего таймфрейма для графика цены
    base_tf = '1h'  # Используем часовой таймфрейм для визуализации
    price_data = data_provider.get_complete_historical_data(symbol, base_tf, BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date'])
    
    # Построение всех графиков
    equity_fig, drawdown_fig, price_chart_fig, stats_fig = visualizer.plot_all_results(results, price_data)
    
    # Сохранение результатов в формате txt для быстрого просмотра
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write(f"Тест с параметрами:\n")
        f.write(f"- Цель прибыли для частичного закрытия: {partial_profit_target*100}%\n")
        f.write(f"- Процент частичного закрытия: {partial_close_percent*100}%\n")
        f.write(f"- Перевод стопа в безубыток: {'Да' if move_stop_to_breakeven else 'Нет'}\n\n")
        
        f.write(f"Результаты:\n")
        f.write(f"- Начальный баланс: ${results['initial_balance']:.2f}\n")
        f.write(f"- Конечный баланс: ${results['final_balance']:.2f}\n")
        f.write(f"- Прибыль: ${results['final_balance'] - results['initial_balance']:.2f} ({results['profit_percent']:.2f}%)\n")
        f.write(f"- Всего сделок: {results['stats']['total_trades']}\n")
        f.write(f"- Выигрышных сделок: {results['stats']['winning_trades']} ({results['stats']['win_rate']:.2f}%)\n")
        f.write(f"- Проигрышных сделок: {results['stats']['losing_trades']}\n")
        f.write(f"- Частичных закрытий: {results['stats'].get('partial_close_count', 0)}\n")
        f.write(f"- Максимальная просадка: {results['stats']['max_drawdown_percent']:.2f}%\n")
        f.write(f"- Sharpe Ratio: {results['stats']['sharpe_ratio']:.2f}\n")
        f.write(f"- Sortino Ratio: {results['stats']['sortino_ratio']:.2f}\n")
    
    logger.info(f"Результаты сохранены в директории: {output_dir}")
    return results, output_dir

def run_comparison_tests():
    """
    Запуск сравнительных тестов с разными параметрами.
    """
    logger.info("Запуск сравнительных тестов с разными параметрами частичного закрытия")
    
    # Тест 1: Без частичного закрытия (стандартный бэктест)
    results_standard, _ = run_test_with_partial_close(partial_profit_target=0.99, partial_close_percent=0, move_stop_to_breakeven=False)
    
    # Тест 2: Частичное закрытие 50% позиции при достижении 10% прибыли, перевод в безубыток
    results_partial_10, _ = run_test_with_partial_close(partial_profit_target=0.1, partial_close_percent=0.5, move_stop_to_breakeven=True)
    
    # Тест 3: Частичное закрытие 50% позиции при достижении 5% прибыли, перевод в безубыток
    results_partial_5, _ = run_test_with_partial_close(partial_profit_target=0.05, partial_close_percent=0.5, move_stop_to_breakeven=True)
    
    # Тест 4: Частичное закрытие 75% позиции при достижении 10% прибыли, перевод в безубыток
    results_partial_75, _ = run_test_with_partial_close(partial_profit_target=0.1, partial_close_percent=0.75, move_stop_to_breakeven=True)
    
    # Сравнение результатов
    compare_results = {
        'Стандартный (без част. закрытия)': {
            'profit_percent': results_standard['profit_percent'],
            'final_balance': results_standard['final_balance'],
            'max_drawdown': results_standard['stats']['max_drawdown_percent'],
            'win_rate': results_standard['stats']['win_rate'],
            'trades': results_standard['stats']['total_trades']
        },
        'Част. закрытие 50% при 10% прибыли': {
            'profit_percent': results_partial_10['profit_percent'],
            'final_balance': results_partial_10['final_balance'],
            'max_drawdown': results_partial_10['stats']['max_drawdown_percent'],
            'win_rate': results_partial_10['stats']['win_rate'],
            'trades': results_partial_10['stats']['total_trades'],
            'partial_closes': results_partial_10['stats'].get('partial_close_count', 0)
        },
        'Част. закрытие 50% при 5% прибыли': {
            'profit_percent': results_partial_5['profit_percent'],
            'final_balance': results_partial_5['final_balance'],
            'max_drawdown': results_partial_5['stats']['max_drawdown_percent'],
            'win_rate': results_partial_5['stats']['win_rate'],
            'trades': results_partial_5['stats']['total_trades'],
            'partial_closes': results_partial_5['stats'].get('partial_close_count', 0)
        },
        'Част. закрытие 75% при 10% прибыли': {
            'profit_percent': results_partial_75['profit_percent'],
            'final_balance': results_partial_75['final_balance'],
            'max_drawdown': results_partial_75['stats']['max_drawdown_percent'],
            'win_rate': results_partial_75['stats']['win_rate'],
            'trades': results_partial_75['stats']['total_trades'],
            'partial_closes': results_partial_75['stats'].get('partial_close_count', 0)
        }
    }
    
    # Создание директории для сохранения результатов сравнения
    output_dir = f"results/comparison/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Создание таблицы сравнения
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_title('Сравнение стратегий частичного закрытия позиций', fontsize=16)
    
    # Подготовка данных для таблицы
    table_data = []
    for name, stats in compare_results.items():
        row = [
            name,
            f"{stats['profit_percent']:.2f}%",
            f"${stats['final_balance']:.2f}",
            f"{stats['max_drawdown']:.2f}%",
            f"{stats['win_rate']:.2f}%",
            f"{stats['trades']}",
            f"{stats.get('partial_closes', '-')}"
        ]
        table_data.append(row)
    
    # Создание таблицы
    table = ax.table(
        cellText=table_data,
        colLabels=['Стратегия', 'Прибыль %', 'Конечный баланс', 'Макс. просадка', 'Win Rate', 'Всего сделок', 'Част. закрытия'],
        loc='center',
        cellLoc='center'
    )
    
    # Настройка таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Сохранение таблицы
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison.png", dpi=300)
    
    # Сохранение результатов в текстовом файле
    with open(f"{output_dir}/comparison.txt", 'w') as f:
        f.write("Сравнение стратегий частичного закрытия позиций\n\n")
        
        for name, stats in compare_results.items():
            f.write(f"{name}:\n")
            f.write(f"- Прибыль: {stats['profit_percent']:.2f}%\n")
            f.write(f"- Конечный баланс: ${stats['final_balance']:.2f}\n")
            f.write(f"- Макс. просадка: {stats['max_drawdown']:.2f}%\n")
            f.write(f"- Win Rate: {stats['win_rate']:.2f}%\n")
            f.write(f"- Всего сделок: {stats['trades']}\n")
            f.write(f"- Частичных закрытий: {stats.get('partial_closes', '-')}\n\n")
    
    logger.info(f"Сравнительные результаты сохранены в директории: {output_dir}")
    return compare_results

if __name__ == "__main__":
    with Pool(processes=min(len(param_combinations), cpu_count())) as pool:
        results = pool.map(run_backtest_with_params, param_combinations)
        
    run_comparison_tests()
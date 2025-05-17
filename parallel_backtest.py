"""
Модуль для параллельного запуска нескольких бэктестов на разных инструментах и таймфреймах.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Добавление текущей директории в путь для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорт модулей системы
from config.settings import (
    EXCHANGE_CONFIG, BACKTEST_CONFIG, RISK_CONFIG,
    TREND_STRATEGY_CONFIG, BREAKOUT_STRATEGY_CONFIG
)
from data.data_provider import DataProvider
from risk_management.risk_manager import RiskManager
from backtest.backtest_engine import BacktestEngine
from strategies.trend_following import TrendFollowingStrategy
from strategies.breakout import BreakoutStrategy

# Настройка логирования
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def setup_logger(name):
    """Настройка логгера для отдельного процесса"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Создаем обработчик для вывода в файл
    file_handler = logging.FileHandler(f"{log_dir}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler.setLevel(logging.INFO)
    
    # Создаем форматировщик
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Добавляем обработчик к логгеру
    logger.addHandler(file_handler)
    
    return logger

def run_single_backtest(params):
    """
    Запуск одного бэктеста с заданными параметрами.
    
    Args:
        params (dict): Параметры бэктеста
            - symbol (str): Символ торговой пары
            - timeframe (str): Таймфрейм
            - start_date (str): Дата начала бэктеста
            - end_date (str): Дата окончания бэктеста
            - strategy_configs (dict): Конфигурации стратегий
            - backtest_config (dict): Конфигурация бэктеста
            
    Returns:
        dict: Результаты бэктеста
    """
    # Настройка логгера
    process_id = multiprocessing.current_process().name
    logger = setup_logger(f"backtest_{params['symbol']}_{params['timeframe']}_{process_id}")
    
    logger.info(f"Запуск бэктеста для {params['symbol']} на таймфрейме {params['timeframe']}")
    
    # Инициализация провайдера данных
    data_provider = DataProvider(EXCHANGE_CONFIG)
    
    # Инициализация риск-менеджера
    risk_manager = RiskManager(RISK_CONFIG)
    
    # Настройка конфигурации бэктеста
    backtest_config = BACKTEST_CONFIG.copy()
    backtest_config.update(params.get('backtest_config', {}))
    backtest_config['start_date'] = params['start_date']
    backtest_config['end_date'] = params['end_date']
    
    # Инициализация движка бэктестинга
    backtest_engine = BacktestEngine(backtest_config, data_provider, risk_manager)
    
    # Создание экземпляров стратегий
    active_strategies = []
    
    # Тренд-следящая стратегия
    if params.get('strategy_configs', {}).get('trend_following', {}).get('enabled', True):
        trend_config = TREND_STRATEGY_CONFIG.copy()
        trend_config.update(params.get('strategy_configs', {}).get('trend_following', {}))
        trend_strategy = TrendFollowingStrategy(trend_config)
        active_strategies.append(trend_strategy)
    
    # Стратегия прорыва уровней
    if params.get('strategy_configs', {}).get('breakout', {}).get('enabled', True):
        breakout_config = BREAKOUT_STRATEGY_CONFIG.copy()
        breakout_config.update(params.get('strategy_configs', {}).get('breakout', {}))
        breakout_strategy = BreakoutStrategy(breakout_config)
        active_strategies.append(breakout_strategy)
    
    # Запуск бэктеста
    tf_dict = {strategy.name: strategy.get_required_timeframes() for strategy in active_strategies}
    
    try:
        results = backtest_engine.run_backtest(active_strategies, params['symbol'], tf_dict)
        
        # Добавляем информацию о параметрах
        results['params'] = params
        
        logger.info(f"Бэктест для {params['symbol']} на таймфрейме {params['timeframe']} завершен успешно")
        logger.info(f"Итоговый результат: {results['profit_percent']:.2f}%")
        
        return results
    except Exception as e:
        logger.error(f"Ошибка при выполнении бэктеста: {e}")
        return {
            'symbol': params['symbol'],
            'timeframe': params['timeframe'],
            'error': str(e),
            'params': params
        }

def run_parallel_backtests(backtest_params_list, max_workers=None):
    """
    Параллельный запуск нескольких бэктестов.
    
    Args:
        backtest_params_list (list): Список словарей с параметрами для каждого бэктеста
        max_workers (int, optional): Максимальное количество параллельных процессов
        
    Returns:
        list: Список результатов бэктестов
    """
    if max_workers is None:
        # Используем количество ядер CPU минус 1, чтобы не перегружать систему
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Запуск {len(backtest_params_list)} бэктестов на {max_workers} процессах")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем все бэктесты параллельно
        futures = [executor.submit(run_single_backtest, params) for params in backtest_params_list]
        
        # Получаем результаты по мере завершения
        # Собираем результаты по мере завершения
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"Завершено {i+1}/{len(backtest_params_list)}: "
                      f"{result.get('params', {}).get('symbol')} на таймфрейме "
                      f"{result.get('params', {}).get('timeframe')} - "
                      f"Результат: {result.get('profit_percent', 'ошибка')}%")
            except Exception as e:
                print(f"Ошибка при получении результата: {e}")
    
    return results

def save_backtest_results(results, output_dir='results/parallel'):
    """
    Сохранение результатов параллельных бэктестов.
    
    Args:
        results (list): Список результатов бэктестов
        output_dir (str): Директория для сохранения результатов
    """
    # Создаем директорию, если ее нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Текущая дата и время для имени файлов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Создаем сводную таблицу результатов
    summary_data = []
    
    for result in results:
        if 'error' in result:
            # Если была ошибка
            row = {
                'symbol': result.get('params', {}).get('symbol', ''),
                'timeframe': result.get('params', {}).get('timeframe', ''),
                'start_date': result.get('params', {}).get('start_date', ''),
                'end_date': result.get('params', {}).get('end_date', ''),
                'profit_percent': 'ERROR',
                'win_rate': 'ERROR',
                'sharpe_ratio': 'ERROR',
                'max_drawdown': 'ERROR',
                'total_trades': 'ERROR',
                'error': result.get('error', 'Unknown error')
            }
        else:
            # Если бэктест успешен
            row = {
                'symbol': result.get('params', {}).get('symbol', ''),
                'timeframe': result.get('params', {}).get('timeframe', ''),
                'start_date': result.get('params', {}).get('start_date', ''),
                'end_date': result.get('params', {}).get('end_date', ''),
                'profit_percent': result.get('profit_percent', 0),
                'win_rate': result.get('stats', {}).get('win_rate', 0),
                'sharpe_ratio': result.get('stats', {}).get('sharpe_ratio', 0),
                'max_drawdown': result.get('stats', {}).get('max_drawdown_percent', 0),
                'total_trades': result.get('stats', {}).get('total_trades', 0),
                'error': ''
            }
        
        summary_data.append(row)
    
    # Сортировка по прибыли (от большей к меньшей)
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by='profit_percent', ascending=False)
    
    # Сохраняем сводную таблицу
    summary_df.to_csv(f"{output_dir}/summary_{timestamp}.csv", index=False)
    
    # Сохраняем полные результаты каждого бэктеста
    for i, result in enumerate(results):
        if 'error' not in result:
            symbol = result.get('params', {}).get('symbol', '')
            timeframe = result.get('params', {}).get('timeframe', '')
            
            # Сохраняем детальные результаты по каждому бэктесту
            result_dir = f"{output_dir}/{symbol}_{timeframe}_{timestamp}"
            os.makedirs(result_dir, exist_ok=True)
            
            # Сохраняем статистику
            with open(f"{result_dir}/stats.txt", 'w') as f:
                f.write(f"Символ: {symbol}\n")
                f.write(f"Таймфрейм: {timeframe}\n")
                f.write(f"Период: {result.get('params', {}).get('start_date', '')} - {result.get('params', {}).get('end_date', '')}\n\n")
                f.write(f"Прибыль: {result.get('profit_percent', 0):.2f}%\n")
                f.write(f"Начальный баланс: ${result.get('initial_balance', 0):.2f}\n")
                f.write(f"Конечный баланс: ${result.get('final_balance', 0):.2f}\n\n")
                
                stats = result.get('stats', {})
                f.write(f"Общее количество сделок: {stats.get('total_trades', 0)}\n")
                f.write(f"Выигрышных сделок: {stats.get('winning_trades', 0)}\n")
                f.write(f"Проигрышных сделок: {stats.get('losing_trades', 0)}\n")
                f.write(f"Процент выигрышных: {stats.get('win_rate', 0):.2f}%\n")
                f.write(f"Profit Factor: {stats.get('profit_factor', 0):.2f}\n")
                f.write(f"Максимальная просадка: {stats.get('max_drawdown_percent', 0):.2f}%\n")
                f.write(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"Sortino Ratio: {stats.get('sortino_ratio', 0):.2f}\n")
            
            # Сохраняем данные о сделках
            trades_df = pd.DataFrame(result.get('trades', []))
            if not trades_df.empty:
                trades_df.to_csv(f"{result_dir}/trades.csv", index=False)
            
            # Сохраняем equity curve
            equity_df = pd.DataFrame(result.get('equity_curve', []))
            if not equity_df.empty:
                equity_df.to_csv(f"{result_dir}/equity_curve.csv", index=False)
    
    print(f"Результаты сохранены в директории: {output_dir}")
    return summary_df

def create_backtest_configs_for_multiple_symbols(symbols, timeframes, period_months):
    """
    Создание конфигураций для бэктестов на нескольких символах и таймфреймах.
    
    Args:
        symbols (list): Список символов для тестирования
        timeframes (list): Список таймфреймов
        period_months (int): Количество месяцев для бэктеста
        
    Returns:
        list: Список конфигураций для бэктестов
    """
    # Расчет дат для бэктеста
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(months=period_months)).strftime('%Y-%m-%d')
    
    configs = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            # Создаем конфигурацию для бэктеста
            config = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'strategy_configs': {
                    'trend_following': {
                        'enabled': True
                    },
                    'breakout': {
                        'enabled': True
                    }
                },
                'backtest_config': {
                    'partial_profit_target': 0.1,
                    'partial_close_percent': 0.5,
                    'move_stop_to_breakeven': True
                }
            }
            
            configs.append(config)
    
    return configs

if __name__ == "__main__":
    # Пример использования
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
    timeframes = ['1h', '4h', '1d']
    
    # Создаем конфигурации для бэктестов
    configs = create_backtest_configs_for_multiple_symbols(symbols, timeframes, period_months=6)
    
    # Запускаем параллельные бэктесты
    results = run_parallel_backtests(configs)
    
    # Сохраняем результаты
    summary = save_backtest_results(results)
    
    # Выводим сводную таблицу
    print("\n=== Сводная таблица результатов ===")
    print(summary[['symbol', 'timeframe', 'profit_percent', 'win_rate', 'sharpe_ratio', 'max_drawdown', 'total_trades']])
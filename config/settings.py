"""
Настройки и конфигурация торговой системы.
Содержит параметры для различных стратегий и общие настройки.
"""

from datetime import datetime, timedelta

# Настройки соединения с биржей
EXCHANGE_CONFIG = {
    'name': 'binance',
    'api_key': '',  # Оставить пустым для бэктестинга
    'api_secret': '',  # Оставить пустым для бэктестинга
    'testnet': False,  # Для бэктестинга это не имеет значения
}

# Основные настройки торговли
TRADING_CONFIG = {
    'symbol': 'BTCUSDT',    # Торговая пара
    'timeframe': '1h',      # Основной таймфрейм
    'mode': 'backtest',     # 'backtest' или 'live'
}

# Настройки для бэктестинга
BACKTEST_CONFIG = {
    'start_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),  # Последние 30 дней
    'end_date': datetime.now().strftime('%Y-%m-%d'),
    'initial_balance': 10000,
    'fee_rate': 0.0004,      # 0.04% комиссия
    'max_position_hold_time': 48,  # Максимальное время удержания позиции в часах (2 дня)
}

# Настройки риск-менеджмента
RISK_CONFIG = {
    'max_position_size': 0.1,       # Максимальный размер позиции (% от баланса)
    'max_positions': 3,             # Максимальное количество одновременных позиций
    'risk_per_trade': 0.01,         # Риск на сделку (% от баланса)
    'default_leverage': 3,          # Стандартное плечо
    'max_leverage': 5,              # Максимальное плечо
    'max_daily_drawdown': 0.05,     # Максимальная дневная просадка (% от баланса)
}

# Параметры стратегии тренда
TREND_STRATEGY_CONFIG = {
    'name': 'Trend Following',
    'enabled': True,
    'timeframes': {
        'trend': '1h',             # Таймфрейм для определения тренда
        'entry': '15m',            # Таймфрейм для входа (уменьшен для коротких сделок)
    },
    'params': {
        'ema_short': 8,            # Уменьшено для быстрой реакции
        'ema_long': 21,            # Уменьшено для быстрой реакции
        'rsi_period': 9,           # Уменьшено для большей чувствительности
        'rsi_oversold': 40,        
        'rsi_overbought': 60,      
        'adx_period': 10,          # Уменьшено для быстрой реакции
        'adx_threshold': 15,       
    }
}

# Параметры стратегии прорыва уровней
BREAKOUT_STRATEGY_CONFIG = {
    'name': 'Level Breakout',
    'enabled': True,
    'timeframes': {
        'levels': '1h',            # Таймфрейм для определения уровней (уменьшен)
        'entry': '15m',            # Таймфрейм для входа (уменьшен)
    },
    'params': {
        'lookback_periods': 30,     # Количество свечей для поиска уровней
        'significance_threshold': 1, # Минимальное количество касаний для уровня
        'volume_threshold': 1.2,     # Минимальное увеличение объема при пробое
        'bollinger_period': 20,
        'bollinger_std_dev': 1.8,
    }
}

# Параметры гибридной стратегии
HYBRID_STRATEGY_CONFIG = {
    'name': 'Momentum-MeanReversion',
    'enabled': True,
    'timeframes': {
        'analysis': '1h',          # Таймфрейм для анализа (уменьшен)
        'entry': '15m',            # Таймфрейм для входа (уменьшен)
    },
    'params': {
        'sma_short': 5,            # Период короткой SMA
        'sma_medium': 15,          # Период средней SMA (уменьшен)
        'sma_long': 30,            # Период длинной SMA (уменьшен)
        'rsi_period': 9,           # Период RSI (уменьшен)
        'stoch_period': 9,         # Период стохастика (уменьшен)
        'bollinger_period': 15,    # Период Боллинджера (уменьшен)
        'bb_std': 2.0,             # Стандартное отклонение Боллинджера
    }
}

# Веса для комбинирования стратегий
STRATEGY_WEIGHTS = {
    'Trend Following Strategy': 0.3,
    'Level Breakout Strategy': 0.3,
    'Momentum-MeanReversion': 0.4
}

# Пороги для генерации сигналов
SIGNAL_THRESHOLDS = {
    'buy': 0.3,    # Порог для сигнала на покупку
    'sell': -0.3,  # Порог для сигнала на продажу
}

# Настройки времени для фильтрации сделок
TIME_FILTERS = {
    'avoid_weekends': True,         # Избегать торговли в выходные
    'avoid_market_open_close': True, # Избегать открытия/закрытия основных рынков
    'market_open_hours': [9, 10],    # Часы открытия рынка UTC
    'market_close_hours': [21, 22],  # Часы закрытия рынка UTC
}
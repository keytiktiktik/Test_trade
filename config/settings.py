"""
Настройки и конфигурация торговой системы.
Содержит параметры для различных стратегий и общие настройки.
"""

# Настройки соединения с биржей
# Настройки биржи
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
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'initial_balance': 10000,
    'fee_rate': 0.0004,     # 0.04% комиссия
}

# Настройки риск-менеджмента
RISK_CONFIG = {
    'max_position_size': 0.1,       # Максимальный размер позиции (% от баланса)
    'max_positions': 5,             # Максимальное количество одновременных позиций
    'risk_per_trade': 0.005,        # Риск на сделку (% от баланса)
    'default_leverage': 5,          # Стандартное плечо
    'max_leverage': 20,              # Максимальное плечо
    'max_daily_drawdown': 0.05,     # Максимальная дневная просадка (% от баланса)
}

# Параметры стратегии тренда
TREND_STRATEGY_CONFIG = {
    'name': 'Trend Following',
    'enabled': True,
    'timeframes': {
        'trend': '1h',             # Таймфрейм для определения тренда
        'entry': '15m',            # Таймфрейм для входа
    },
    'params': {
        'ema_short': 8,            # Период короткой EMA
        'ema_long': 21,            # Период длинной EMA
        'rsi_period': 14,          # Период RSI
        'rsi_oversold': 40,        # Уровень перепроданности RSI
        'rsi_overbought': 60,      # Уровень перекупленности RSI
        'adx_period': 14,          # Период ADX
        'adx_threshold': 15,       # Порог силы тренда ADX
    }
}

# Параметры стратегии прорыва уровней
BREAKOUT_STRATEGY_CONFIG = {
    'name': 'Level Breakout',
    'enabled': True,
    'timeframes': {
        'levels': '4h',            # Таймфрейм для определения уровней
        'entry': '1h',             # Таймфрейм для входа
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
        'analysis': '4h',          # Таймфрейм для анализа
        'entry': '1h',             # Таймфрейм для входа
    },
    'params': {
        'sma_short': 5,            # Период короткой SMA
        'sma_medium': 20,          # Период средней SMA
        'sma_long': 50,            # Период длинной SMA
        'rsi_period': 14,          # Период RSI
        'stoch_period': 14,        # Период стохастика
        'bollinger_period': 20,    # Период Боллинджера
        'bb_std': 2.0,             # Стандартное отклонение Боллинджера
    }
}

# Веса для комбинирования стратегий
STRATEGY_WEIGHTS = {
    'Trend Following': 0.3,
    'Level Breakout': 0.3,
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
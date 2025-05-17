"""
Реализация стратегии, основанной на прорыве уровней поддержки/сопротивления.
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from indicators import technical
# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import Strategy
from indicators import technical

logger = logging.getLogger('breakout_strategy')

class BreakoutStrategy(Strategy):
    """
    Стратегия прорыва уровней.
    Использует уровни поддержки/сопротивления, Bollinger Bands и OBV для определения точек входа.
    """
    
    def __init__(self, config):
        """
        Инициализация стратегии прорыва уровней.
        
        Args:
            config (dict): Конфигурация стратегии
        """
        super().__init__("Level Breakout Strategy", config)
        
        # Получение параметров из конфигурации
        self.lookback = self.params.get('lookback_periods', 30)
        self.significance = self.params.get('significance_threshold', 1)
        self.volume_threshold = self.params.get('volume_threshold', 1.2)
        self.bollinger_period = self.params.get('bollinger_period', 20)
        self.bollinger_std_dev = self.params.get('bollinger_std_dev', 1.8)
        
        # Для хранения найденных уровней
        self.support_levels = []
        self.resistance_levels = []
        
        logger.info(f"Инициализирована стратегия прорыва уровней с параметрами: {self.params}")
    
    def calculate_indicators(self, df):
        """
        Расчет индикаторов для стратегии прорыва уровней.
        
        Args:
            df (pd.DataFrame): DataFrame с OHLCV данными
            
        Returns:
            pd.DataFrame: DataFrame с добавленными индикаторами
        """
        # Клонирование исходного DataFrame
        df = df.copy()
        
        # Расчет полос Боллинджера
        middle_band, upper_band, lower_band = technical.calculate_bollinger_bands(
            df['close'], period=self.bollinger_period, std_dev=self.bollinger_std_dev
        )
        df['bb_middle'] = middle_band
        df['bb_upper'] = upper_band
        df['bb_lower'] = lower_band
        
        # Расчет OBV (On-Balance Volume)
        df['obv'] = technical.calculate_obv(df)
        
        # Расчет ATR для стоп-лоссов
        df['atr'] = technical.calculate_atr(df, 14)
        
        # Поиск уровней поддержки и сопротивления
        if len(df) >= self.lookback:
            self.support_levels, self.resistance_levels = technical.find_support_resistance_levels(
                df.tail(self.lookback), significance_threshold=self.significance
            )
        
        return df
    
    def generate_signal(self, df):
        """
        Генерация сигналов для стратегии прорыва уровней с упрощенными условиями.
        
        Args:
            df (pd.DataFrame): DataFrame с OHLCV данными и индикаторами
            
        Returns:
            tuple: (сигнал, причина)
        """
        if len(df) < self.lookback:
            return 0, "Недостаточно данных"
        
        # Получение последних значений
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Проверка пробоя уровня сопротивления (упрощенные условия)
        for level in self.resistance_levels:
            if previous['close'] < level and current['close'] > level:
                # Проверка объема (упрощено)
                avg_volume = df['volume'].tail(10).mean()  # Было 20
                volume_increased = current['volume'] > avg_volume
                
                # Только базовая проверка объема
                if volume_increased:
                    return 1, f"Пробой сопротивления {level:.2f} с увеличенным объемом"
        
        # Проверка пробоя уровня поддержки (упрощенные условия)
        for level in self.support_levels:
            if previous['close'] > level and current['close'] < level:
                # Проверка объема (упрощено)
                avg_volume = df['volume'].tail(10).mean()  # Было 20
                volume_increased = current['volume'] > avg_volume
                
                # Только базовая проверка объема
                if volume_increased:
                    return -1, f"Пробой поддержки {level:.2f} с увеличенным объемом"
        
        return 0, "Нет сигнала"
"""
Реализация гибридной стратегии, сочетающей моментум и возврат к среднему.
"""

import pandas as pd
import numpy as np
import sys
import os
import logging

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import Strategy
from indicators import technical

logger = logging.getLogger('hybrid_strategy')

class MomentumMeanReversionStrategy(Strategy):
    """Гибридная стратегия, сочетающая моментум и возврат к среднему"""
    
    def __init__(self, config):
        """
        Инициализация гибридной стратегии.
        
        Args:
            config (dict): Конфигурация стратегии
        """
        super().__init__("Momentum-MeanReversion", config)
        
        # Получение параметров из конфигурации
        self.sma_short = self.params.get('sma_short', 5)
        self.sma_medium = self.params.get('sma_medium', 20)
        self.sma_long = self.params.get('sma_long', 50)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.stoch_period = self.params.get('stoch_period', 14)
        self.bollinger_period = self.params.get('bollinger_period', 20)
        self.bb_std = self.params.get('bb_std', 2.0)
        
        logger.info(f"Инициализирована гибридная стратегия с параметрами: {self.params}")
    
    def calculate_indicators(self, df):
        """
        Расчет индикаторов для гибридной стратегии.
        
        Args:
            df (pd.DataFrame): DataFrame с OHLCV данными
            
        Returns:
            pd.DataFrame: DataFrame с добавленными индикаторами
        """
        df = df.copy()
        
        # Простые скользящие средние
        df['sma_short'] = technical.calculate_sma(df['close'], self.sma_short)
        df['sma_medium'] = technical.calculate_sma(df['close'], self.sma_medium)
        df['sma_long'] = technical.calculate_sma(df['close'], self.sma_long)
        
        # RSI
        df['rsi'] = technical.calculate_rsi(df['close'], self.rsi_period)
        
        # Стохастик
        df['stoch_k'], df['stoch_d'] = technical.calculate_stochastic(df, self.stoch_period, 3)
        
        # Полосы Боллинджера
        df['bb_middle'], df['bb_upper'], df['bb_lower'] = technical.calculate_bollinger_bands(
            df['close'], self.bollinger_period, self.bb_std
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR для стоп-лоссов
        df['atr'] = technical.calculate_atr(df, 14)
        
        # Вычисление объемных показателей
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Добавляем показатель волатильности
        df['volatility'] = df['close'].pct_change().rolling(5).std() * 100
        
        return df
    
    def generate_signal(self, df):
        """
        Генерация сигналов на основе гибридной стратегии.
        
        Args:
            df (pd.DataFrame): DataFrame с OHLCV данными и индикаторами
            
        Returns:
            tuple: (сигнал, причина)
        """
        if len(df) < max(self.sma_long, self.bollinger_period) + 5:
            return 0, "Недостаточно данных"
        
        # Получаем последние данные
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Определение текущего рыночного режима
        is_trending = abs(current['sma_short'] - current['sma_long']) / current['sma_long'] > 0.015  # 1.5% разница
        is_volatile = current['volatility'] > df['volatility'].mean() * 1.2  # Волатильность выше среднего на 20%+
        
        # Проверка момента 1: Пересечение скользящих средних
        crossover_up = previous['sma_short'] <= previous['sma_medium'] and current['sma_short'] > current['sma_medium']
        crossover_down = previous['sma_short'] >= previous['sma_medium'] and current['sma_short'] < current['sma_medium']
        
        # Проверка момента 2: Стохастический RSI
        stoch_oversold = current['stoch_k'] < 20 and current['stoch_k'] > previous['stoch_k']
        stoch_overbought = current['stoch_k'] > 80 and current['stoch_k'] < previous['stoch_k']
        
        # Проверка среднереверсивных условий
        bb_lower_touch = current['close'] < current['bb_lower'] and previous['close'] >= previous['bb_lower']
        bb_upper_touch = current['close'] > current['bb_upper'] and previous['close'] <= previous['bb_upper']
        
        # Подтверждение объемом
        volume_confirms = current['volume_ratio'] > 1.1
        
        # ЛОГИКА ГЕНЕРАЦИИ СИГНАЛОВ
        
        # Для трендового рынка: используем моментум-стратегию
        if is_trending and not is_volatile:
            if crossover_up and volume_confirms and current['rsi'] > 45:
                return 1, f"Трендовый рынок, пересечение SMA вверх, RSI={current['rsi']:.1f}, объем x{current['volume_ratio']:.1f}"
                
            elif crossover_down and volume_confirms and current['rsi'] < 55:
                return -1, f"Трендовый рынок, пересечение SMA вниз, RSI={current['rsi']:.1f}, объем x{current['volume_ratio']:.1f}"
        
        # Для флэта и волатильного рынка: используем среднереверсивную стратегию
        else:
            if bb_lower_touch and stoch_oversold and volume_confirms:
                return 1, f"Возврат к среднему, касание нижней BB, Stoch={current['stoch_k']:.1f}, объем x{current['volume_ratio']:.1f}"
                
            elif bb_upper_touch and stoch_overbought and volume_confirms:
                return -1, f"Возврат к среднему, касание верхней BB, Stoch={current['stoch_k']:.1f}, объем x{current['volume_ratio']:.1f}"
        
        return 0, "Нет сигнала"
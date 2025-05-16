"""
Реализация стратегии, основанной на тренде.
Использует Supertrend, EMA, ADX и RSI для определения тренда и точек входа.
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

logger = logging.getLogger('trend_strategy')

class TrendFollowingStrategy(Strategy):
    """
    Стратегия следования за трендом.
    Использует Supertrend, EMA, ADX и RSI для определения тренда и точек входа.
    """
    
    def __init__(self, config):
        """
        Инициализация стратегии тренда.
        
        Args:
            config (dict): Конфигурация стратегии
        """
        super().__init__("Trend Following Strategy", config)
        
        # Получение параметров из конфигурации
        self.ema_short = self.params.get('ema_short', 8)
        self.ema_long = self.params.get('ema_long', 21)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_oversold = self.params.get('rsi_oversold', 40)
        self.rsi_overbought = self.params.get('rsi_overbought', 60)
        self.adx_period = self.params.get('adx_period', 14)
        self.adx_threshold = self.params.get('adx_threshold', 15)
        
        logger.info(f"Инициализирована стратегия тренда с параметрами: {self.params}")
    
    def calculate_indicators(self, df):
        """
        Расчет индикаторов для тренд-следящей стратегии.
        
        Args:
            df (pd.DataFrame): DataFrame с OHLCV данными
            
        Returns:
            pd.DataFrame: DataFrame с добавленными индикаторами
        """
        # Клонирование исходного DataFrame
        df = df.copy()
        
        # Расчет EMA
        df['sma20'] = technical.calculate_sma(df['close'], self.ema_short)
        df['sma50'] = technical.calculate_sma(df['close'], self.ema_long)
        
        # Расчет RSI
        df['rsi'] = technical.calculate_rsi(df['close'], self.rsi_period)
        
        # Расчет ADX
        df = technical.calculate_adx(df, self.adx_period)
        
        # Расчет ATR для стоп-лоссов
        df['atr'] = technical.calculate_atr(df, 14)
        
        return df
    
    def generate_signal(self, df):
        """
        Генерация сигналов для тренд-следящей стратегии с упрощенными фильтрами.
        
        Args:
            df (pd.DataFrame): DataFrame с OHLCV данными и индикаторами
            
        Returns:
            tuple: (сигнал, причина)
        """
        if len(df) < max(self.ema_short, self.ema_long, self.rsi_period) + 5:
            return 0, "Недостаточно данных"
        
        # Получение последних значений индикаторов
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Упрощенная проверка силы тренда через ADX
        strong_trend = current['adx'] > self.adx_threshold if 'adx' in current else True
        
        # УПРОЩЕННЫЕ ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ
        # Проверка направления тренда за последние N баров
        last_n_bars = 3
        uptrend_count = sum(1 for i in range(1, min(last_n_bars+1, len(df))) 
                         if df['close'].iloc[-i] > df['close'].iloc[-i-1])
        downtrend_count = sum(1 for i in range(1, min(last_n_bars+1, len(df))) 
                           if df['close'].iloc[-i] < df['close'].iloc[-i-1])
        
        uptrend_consistency = uptrend_count / last_n_bars if last_n_bars > 0 else 0
        downtrend_consistency = downtrend_count / last_n_bars if last_n_bars > 0 else 0
        
        # Проверка объема
        volume_increasing = current['volume'] > df['volume'].iloc[-3:].mean()
        
        # Проверка сигнала на покупку с упрощенными фильтрами
        if (current['sma20'] > current['sma50'] and 
            current['rsi'] > previous['rsi'] and  # Просто растущий RSI вместо выхода из перепроданности
            current['rsi'] < 70 and               # RSI не перекуплен
            strong_trend and
            uptrend_consistency >= 0.5):          # Порог снижен с 0.6 до 0.5
            return 1, f"EMA: восходящий тренд, RSI: растет ({current['rsi']:.2f}), ADX: {current['adx']:.2f}, Сила тренда: {uptrend_consistency:.2f}"
        
        # Проверка сигнала на продажу с упрощенными фильтрами
        elif (current['sma20'] < current['sma50'] and 
              current['rsi'] < previous['rsi'] and  # Просто падающий RSI вместо выхода из перекупленности
              current['rsi'] > 30 and               # RSI не перепродан
              strong_trend and
              downtrend_consistency >= 0.5):        # Порог снижен с 0.6 до 0.5
            return -1, f"EMA: нисходящий тренд, RSI: падает ({current['rsi']:.2f}), ADX: {current['adx']:.2f}, Сила тренда: {downtrend_consistency:.2f}"
        
        return 0, "Нет сигнала"
# optimized_indicators.py
"""
Оптимизированный модуль технических индикаторов с автоматическим выбором 
наиболее эффективного метода (pandas, JIT или GPU) в зависимости от размера данных.
"""

import numpy as np
import pandas as pd
from numba import jit, cuda
import math

# Проверка доступности CUDA
CUDA_AVAILABLE = cuda.is_available()

# Пороги для переключения между методами
GPU_THRESHOLD = 500000  # Ниже этого порога используется pandas, выше - GPU или JIT
JIT_THRESHOLD = 50000   # Ниже этого порога используется pandas, выше - JIT

#========================= JIT-оптимизированные функции =========================

@jit(nopython=True)
def _calculate_sma_jit(data, period):
    n = len(data)
    result = np.zeros(n)
    for i in range(n):
        if i < period - 1:
            result[i] = np.nan
        else:
            result[i] = np.mean(data[i-period+1:i+1])
    return result

@jit(nopython=True)
def _calculate_ema_jit(data, period):
    n = len(data)
    result = np.zeros(n)
    alpha = 2.0 / (period + 1)
    
    # Инициализация первого значения
    if n > 0:
        result[0] = data[0]
    
    # Расчет EMA
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    # Установка NaN для первых значений
    for i in range(min(period - 1, n)):
        result[i] = np.nan
        
    return result

@jit(nopython=True)
def _calculate_rsi_jit(data, period=14):
    n = len(data)
    result = np.zeros(n)
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    # Расчет изменений
    for i in range(1, n):
        change = data[i] - data[i-1]
        if change > 0:
            gains[i] = change
            losses[i] = 0
        else:
            gains[i] = 0
            losses[i] = -change
    
    # Расчет средних значений и RSI
    avg_gain = np.nan
    avg_loss = np.nan
    
    for i in range(period, n):
        if i == period:
            # Инициализация первых средних значений
            avg_gain = np.mean(gains[1:period+1])
            avg_loss = np.mean(losses[1:period+1])
        else:
            # Следующие значения по формуле Wilder
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
        if avg_loss == 0:
            result[i] = 100
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))
    
    return result

@jit(nopython=True)
def _calculate_bollinger_bands_jit(data, period=20, std_dev=2):
    n = len(data)
    middle_band = np.zeros(n)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        if i < period - 1:
            middle_band[i] = np.nan
            upper_band[i] = np.nan
            lower_band[i] = np.nan
        else:
            window = data[i-period+1:i+1]
            mean_val = np.mean(window)
            std_val = np.std(window)
            
            middle_band[i] = mean_val
            upper_band[i] = mean_val + std_dev * std_val
            lower_band[i] = mean_val - std_dev * std_val
    
    return middle_band, upper_band, lower_band

@jit(nopython=True)
def _calculate_atr_jit(high, low, close, period=14):
    n = len(high)
    tr = np.zeros(n)
    atr = np.zeros(n)
    
    # Расчет True Range
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Первое значение ATR
    if n > period:
        atr[period] = np.mean(tr[1:period+1])
        
        # Последующие значения ATR
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

#========================= GPU-оптимизированные функции =========================

if CUDA_AVAILABLE:
    @cuda.jit
    def _cuda_calculate_sma_kernel(data, period, result):
        i = cuda.grid(1)
        if i < len(data):
            if i < period - 1:
                result[i] = math.nan
            else:
                sum_val = 0.0
                for j in range(period):
                    sum_val += data[i - j]
                result[i] = sum_val / period
    
    @cuda.jit
    def _cuda_calculate_ema_kernel(data, alpha, result):
        i = cuda.grid(1)
        if i == 0 and i < len(data):
            result[i] = data[i]
        elif i < len(data):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    def _calculate_sma_gpu(data, period):
        """GPU-оптимизированная версия SMA"""
        # Проверяем, что данные - это np.array
        if not isinstance(data, np.ndarray):
            data_array = np.array(data, dtype=np.float32)
        else:
            data_array = data.astype(np.float32)
        
        # Создаем массив для результата
        result = np.zeros_like(data_array)
        
        # Расчет размера сетки и блоков
        threads_per_block = 256
        blocks_per_grid = (len(data_array) + threads_per_block - 1) // threads_per_block
        
        # Запуск CUDA ядра
        _cuda_calculate_sma_kernel[blocks_per_grid, threads_per_block](
            data_array, period, result)
        
        return result
    
    def _calculate_ema_gpu(data, period):
        """GPU-оптимизированная версия EMA"""
        # Проверяем, что данные - это np.array
        if not isinstance(data, np.ndarray):
            data_array = np.array(data, dtype=np.float32)
        else:
            data_array = data.astype(np.float32)
        
        # Создаем массив для результата
        result = np.zeros_like(data_array)
        
        # Расчет alpha
        alpha = 2.0 / (period + 1)
        
        # Расчет размера сетки и блоков
        threads_per_block = 256
        blocks_per_grid = (len(data_array) + threads_per_block - 1) // threads_per_block
        
        # Запуск CUDA ядра
        _cuda_calculate_ema_kernel[blocks_per_grid, threads_per_block](
            data_array, alpha, result)
        
        # Установка NaN для первых значений
        result[:period-1] = np.nan
        
        return result

#========================= Основные публичные функции =========================

def calculate_sma(data, period):
    """
    Рассчитывает простую скользящую среднюю (SMA) с автоматическим выбором оптимального метода.
    
    Args:
        data (pd.Series): Серия цен
        period (int): Период SMA
        
    Returns:
        pd.Series: Серия значений SMA
    """
    size = len(data)
    
    if size < GPU_THRESHOLD:
        # Маленькие данные - используем pandas
        result = data.rolling(window=period).mean()
    elif CUDA_AVAILABLE:
        try:
            # Большие данные и доступен GPU - используем GPU
            result = pd.Series(_calculate_sma_gpu(data.values, period), index=data.index)
        except Exception as e:
            # Если ошибка с GPU - используем JIT
            result = pd.Series(_calculate_sma_jit(data.values, period), index=data.index)
    else:
        # Большие данные, но GPU недоступен - используем JIT
        result = pd.Series(_calculate_sma_jit(data.values, period), index=data.index)
    
    return result

def calculate_ema(data, period):
    """
    Рассчитывает экспоненциальную скользящую среднюю (EMA) с автоматическим выбором оптимального метода.
    
    Args:
        data (pd.Series): Серия цен
        period (int): Период EMA
        
    Returns:
        pd.Series: Серия значений EMA
    """
    size = len(data)
    
    if size < GPU_THRESHOLD:
        # Маленькие данные - используем pandas
        result = data.ewm(span=period, adjust=False).mean()
    elif CUDA_AVAILABLE:
        try:
            # Большие данные и доступен GPU - используем GPU
            result = pd.Series(_calculate_ema_gpu(data.values, period), index=data.index)
        except Exception as e:
            # Если ошибка с GPU - используем JIT
            result = pd.Series(_calculate_ema_jit(data.values, period), index=data.index)
    else:
        # Большие данные, но GPU недоступен - используем JIT
        result = pd.Series(_calculate_ema_jit(data.values, period), index=data.index)
    
    return result

def calculate_rsi(data, period=14):
    """
    Рассчитывает индикатор относительной силы (RSI) с автоматическим выбором оптимального метода.
    
    Args:
        data (pd.Series): Серия цен закрытия
        period (int): Период RSI
        
    Returns:
        pd.Series: Серия значений RSI
    """
    size = len(data)
    
    if size < JIT_THRESHOLD:
        # Маленькие данные - используем pandas
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        result = rsi
    else:
        # Большие данные - используем JIT
        result = pd.Series(_calculate_rsi_jit(data.values, period), index=data.index)
    
    return result

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    Рассчитывает полосы Боллинджера с автоматическим выбором оптимального метода.
    
    Args:
        data (pd.Series): Серия цен закрытия
        period (int): Период для скользящей средней
        std_dev (float): Количество стандартных отклонений
        
    Returns:
        tuple: (middle_band, upper_band, lower_band)
    """
    size = len(data)
    
    if size < JIT_THRESHOLD:
        # Маленькие данные - используем pandas
        middle_band = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
    else:
        # Большие данные - используем JIT
        middle_band, upper_band, lower_band = _calculate_bollinger_bands_jit(data.values, period, std_dev)
        middle_band = pd.Series(middle_band, index=data.index)
        upper_band = pd.Series(upper_band, index=data.index)
        lower_band = pd.Series(lower_band, index=data.index)
    
    return middle_band, upper_band, lower_band

def calculate_atr(df, period=14):
    """
    Рассчитывает Average True Range (ATR) с автоматическим выбором оптимального метода.
    
    Args:
        df (pd.DataFrame): DataFrame с ценами high, low, close
        period (int): Период ATR
        
    Returns:
        pd.Series: Серия значений ATR
    """
    size = len(df)
    
    if size < JIT_THRESHOLD:
        # Маленькие данные - используем pandas
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
    else:
        # Большие данные - используем JIT
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        atr = _calculate_atr_jit(high, low, close, period)
        atr = pd.Series(atr, index=df.index)
    
    return atr

def calculate_supertrend(df, period=10, multiplier=3):
    """
    Рассчитывает индикатор SuperTrend.
    
    Args:
        df (pd.DataFrame): DataFrame с ценами high, low, close
        period (int): Период ATR
        multiplier (float): Множитель для ATR
        
    Returns:
        pd.DataFrame: DataFrame с добавленной колонкой supertrend
    """
    # Клонирование исходного DataFrame
    df = df.copy()
    
    # Вычисление ATR (используем оптимизированную функцию)
    df['atr'] = calculate_atr(df, period)
    
    # Вычисление базовых лент SuperTrend
    df['upperband'] = (df['high'] + df['low']) / 2 + multiplier * df['atr']
    df['lowerband'] = (df['high'] + df['low']) / 2 - multiplier * df['atr']
    
    # Инициализация тренда
    df['trend'] = 1  # По умолчанию восходящий тренд
    df['supertrend'] = df['lowerband']
    
    # Вычисление SuperTrend
    for i in range(period, len(df)):
        # Если текущая цена закрытия ниже суперстренда, тренд нисходящий
        if df['close'].iloc[i] <= df['supertrend'].iloc[i-1]:
            df['trend'].iloc[i] = -1
        # Если текущая цена закрытия выше суперстренда, тренд восходящий
        elif df['close'].iloc[i] >= df['supertrend'].iloc[i-1]:
            df['trend'].iloc[i] = 1
        
        # Если тренд восходящий
        if df['trend'].iloc[i] == 1:
            # Текущий суперстренд - максимум из текущей нижней ленты и предыдущего суперстренда
            df['supertrend'].iloc[i] = max(df['lowerband'].iloc[i], df['supertrend'].iloc[i-1])
        # Если тренд нисходящий
        else:
            # Текущий суперстренд - минимум из текущей верхней ленты и предыдущего суперстренда
            df['supertrend'].iloc[i] = min(df['upperband'].iloc[i], df['supertrend'].iloc[i-1])
    
    # Удаление временных колонок
    df = df.drop(['upperband', 'lowerband'], axis=1, errors='ignore')
    
    return df

def find_support_resistance_levels(df, lookback_periods=100, significance_threshold=3):
    """
    Находит уровни поддержки и сопротивления.
    
    Args:
        df (pd.DataFrame): DataFrame с ценами high, low, close
        lookback_periods (int): Количество свечей для анализа
        significance_threshold (int): Минимальное количество касаний для уровня
        
    Returns:
        tuple: (support_levels, resistance_levels)
    """
    # Ограничение данных для анализа
    df = df.tail(lookback_periods)
    
    # Находим локальные минимумы и максимумы
    local_min = []
    local_max = []
    
    for i in range(2, len(df) - 2):
        if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i-2] and \
           df['low'].iloc[i] < df['low'].iloc[i+1] and df['low'].iloc[i] < df['low'].iloc[i+2]:
            local_min.append(df['low'].iloc[i])
            
        if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i-2] and \
           df['high'].iloc[i] > df['high'].iloc[i+1] and df['high'].iloc[i] > df['high'].iloc[i+2]:
            local_max.append(df['high'].iloc[i])
    
    # Кластеризация уровней
    def cluster_levels(levels, threshold_percent=0.01):
        if not levels:
            return []
            
        levels.sort()
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[0]) / current_cluster[0] <= threshold_percent:
                current_cluster.append(level)
            else:
                if len(current_cluster) >= significance_threshold:
                    clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
                
        if len(current_cluster) >= significance_threshold:
            clustered.append(sum(current_cluster) / len(current_cluster))
            
        return clustered
    
    # Получение кластеризованных уровней
    support_levels = cluster_levels(local_min)
    resistance_levels = cluster_levels(local_max)
    
    return support_levels, resistance_levels

def calculate_obv(df):
    """
    Рассчитывает On-Balance Volume (OBV).
    
    Args:
        df (pd.DataFrame): DataFrame с ценами close и объемами volume
        
    Returns:
        pd.Series: Серия значений OBV
    """
    obv = np.zeros(len(df))
    obv[0] = df['volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['volume'].iloc[i]
        else:
            obv[i] = obv[i-1]
    
    return pd.Series(obv, index=df.index)
"""
Модуль с реализацией технических индикаторов для анализа рынка.
"""

import numpy as np
import pandas as pd

def calculate_sma(data, period):
    """
    Рассчитывает простую скользящую среднюю (SMA).
    
    Args:
        data (pd.Series): Серия цен закрытия
        period (int): Период SMA
        
    Returns:
        pd.Series: Серия значений SMA
    """
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    """
    Рассчитывает экспоненциальную скользящую среднюю (EMA).
    
    Args:
        data (pd.Series): Серия цен закрытия
        period (int): Период EMA
        
    Returns:
        pd.Series: Серия значений EMA
    """
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """
    Рассчитывает индикатор относительной силы (RSI).
    
    Args:
        data (pd.Series): Серия цен закрытия
        period (int): Период RSI
        
    Returns:
        pd.Series: Серия значений RSI
    """
    delta = data.diff()
    
    # Отдельные серии для положительных и отрицательных изменений
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Расчет среднего выигрыша и средней потери
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Расчет относительной силы
    rs = avg_gain / avg_loss
    
    # Расчет RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    Рассчитывает полосы Боллинджера.
    
    Args:
        data (pd.Series): Серия цен закрытия
        period (int): Период для скользящей средней
        std_dev (float): Количество стандартных отклонений
        
    Returns:
        tuple: (middle_band, upper_band, lower_band)
    """
    # Средняя линия (SMA)
    middle_band = calculate_sma(data, period)
    
    # Стандартное отклонение
    rolling_std = data.rolling(window=period).std()
    
    # Верхняя и нижняя полосы
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return middle_band, upper_band, lower_band

def calculate_adx(df, period=14):
    """
    Рассчитывает индекс направленного движения (ADX).
    
    Args:
        df (pd.DataFrame): DataFrame с ценами high, low, close
        period (int): Период ADX
        
    Returns:
        pd.DataFrame: DataFrame с добавленными колонками +DI, -DI, ADX
    """
    # Клонирование исходного DataFrame
    df = df.copy()
    
    # Вычисление True Range (TR)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    # Вычисление Directional Movement (DM)
    df['up_move'] = df['high'].diff()
    df['down_move'] = df['low'].diff()
    
    # Положительное и отрицательное направленное движение
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Сглаживание TR, +DM, -DM
    df['tr_smoothed'] = df['tr'].rolling(window=period).sum()
    df['plus_dm_smoothed'] = df['plus_dm'].rolling(window=period).sum()
    df['minus_dm_smoothed'] = df['minus_dm'].rolling(window=period).sum()
    
    # Вычисление +DI и -DI
    df['plus_di'] = 100 * (df['plus_dm_smoothed'] / df['tr_smoothed'])
    df['minus_di'] = 100 * (df['minus_dm_smoothed'] / df['tr_smoothed'])
    
    # Вычисление DX
    df['dx'] = 100 * (np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
    
    # Вычисление ADX (сглаженный DX)
    df['adx'] = df['dx'].rolling(window=period).mean()
    
    # Удаление временных колонок
    df_result = df.drop(['high_low', 'high_close', 'low_close', 'up_move', 'down_move', 
                    'plus_dm', 'minus_dm', 'tr_smoothed', 'plus_dm_smoothed', 
                    'minus_dm_smoothed', 'dx'], axis=1, errors='ignore')
    
    return df_result

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
    
    # Вычисление ATR
    df['tr'] = np.maximum(
        np.maximum(
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift(1))
        ),
        np.abs(df['low'] - df['close'].shift(1))
    )
    df['atr'] = df['tr'].rolling(window=period).mean()
    
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
    df = df.drop(['tr', 'upperband', 'lowerband'], axis=1, errors='ignore')
    
    return df

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

def calculate_stochastic(df, k_period=14, d_period=3):
    """
    Рассчитывает Стохастический осциллятор.
    
    Args:
        df (pd.DataFrame): DataFrame с ценами high, low, close
        k_period (int): Период для линии %K
        d_period (int): Период для линии %D
        
    Returns:
        tuple: (k_line, d_line)
    """
    # Расчет высоких и низких значений за период
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    # Расчет %K
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    
    # Расчет %D (SMA от %K)
    d = k.rolling(window=d_period).mean()
    
    return k, d

def calculate_atr(df, period=14):
    """
    Рассчитывает Average True Range (ATR).
    
    Args:
        df (pd.DataFrame): DataFrame с ценами high, low, close
        period (int): Период ATR
        
    Returns:
        pd.Series: Серия значений ATR
    """
    # Расчет True Range
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Сглаживание для получения ATR
    atr = tr.rolling(window=period).mean()
    
    return atr

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
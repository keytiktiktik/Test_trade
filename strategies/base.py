"""
Базовый класс для торговых стратегий.
Определяет общий интерфейс и функциональность для всех стратегий.
"""

from abc import ABC, abstractmethod
import pandas as pd
import logging

logger = logging.getLogger('strategy')

class Strategy(ABC):
    """Базовый абстрактный класс для торговых стратегий"""
    
    def __init__(self, name, config):
        """
        Инициализация базовой стратегии.
        
        Args:
            name (str): Название стратегии
            config (dict): Конфигурация стратегии
        """
        self.name = name
        self.config = config
        self.is_enabled = config.get('enabled', True)
        self.timeframes = config.get('timeframes', {})
        self.params = config.get('params', {})
        
        # Словарь для хранения последних данных
        self.latest_data = {}
        
        # Словарь для хранения открытых позиций
        self.open_positions = {}
        
        logger.info(f"Инициализирована стратегия: {self.name}")
    
    @abstractmethod
    def calculate_indicators(self, df):
        """
        Расчет индикаторов для стратегии.
        
        Args:
            df (pd.DataFrame): DataFrame с OHLCV данными
            
        Returns:
            pd.DataFrame: DataFrame с добавленными индикаторами
        """
        pass
    
    @abstractmethod
    def generate_signal(self, df):
        """
        Генерация торгового сигнала.
        
        Args:
            df (pd.DataFrame): DataFrame с OHLCV данными и индикаторами
            
        Returns:
            tuple: (сигнал, причина) где сигнал: 1 (покупка), -1 (продажа), 0 (нет сигнала)
        """
        pass
    
    def update_data(self, timeframe, data):
        """
        Обновление данных для стратегии.
        
        Args:
            timeframe (str): Таймфрейм данных
            data (pd.DataFrame): Новые данные
        """
        self.latest_data[timeframe] = data
    
    def get_required_timeframes(self):
        """
        Получение списка необходимых таймфреймов.
        
        Returns:
            list: Список необходимых таймфреймов
        """
        return list(self.timeframes.values())
    
    def check_time_filters(self, current_time, time_filters):
        """
        Проверка временных фильтров для улучшения входов.
        
        Args:
            current_time (datetime): Текущее время
            time_filters (dict): Настройки временных фильтров
            
        Returns:
            bool: True если можно торговать, False если нет
        """
        # Проверка на выходные дни
        if time_filters.get('avoid_weekends', True) and current_time.weekday() > 4:  # 5=Сб, 6=Вс
            return False
        
        # Проверка на часы открытия/закрытия рынка
        if time_filters.get('avoid_market_open_close', True):
            hour = current_time.hour
            open_hours = time_filters.get('market_open_hours', [9, 10])
            close_hours = time_filters.get('market_close_hours', [21, 22])
            
            if hour in open_hours or hour in close_hours:
                return False
        
        return True
    
    def check_exit_signal(self, data, position):
        """
        Проверка сигнала на выход из позиции.
        Базовая реализация проверки на выход, может быть переопределена в дочерних классах.
        
        Args:
            data (pd.DataFrame): DataFrame с OHLCV данными и индикаторами
            position (dict): Информация о текущей позиции
            
        Returns:
            dict or None: Сигнал на выход или None, если сигнала нет
        """
        if len(data) < 2:
            return None
        
        current = data.iloc[-1]
        previous = data.iloc[-2]
        position_type = position['type']
        
        # Быстрый выход при развороте тренда
        if position_type == 'buy' and current['close'] < previous['close'] * 0.995:  # 0.5% движение против позиции
            return {
                'type': 'exit',
                'price': current['close'],
                'time': current.name,
                'reason': "Краткосрочный разворот против лонг-позиции"
            }
        elif position_type == 'sell' and current['close'] > previous['close'] * 1.005:  # 0.5% движение против позиции
            return {
                'type': 'exit',
                'price': current['close'],
                'time': current.name,
                'reason': "Краткосрочный разворот против шорт-позиции"
            }
        
        # Проверка времени в позиции - закрываем если прошло более 1 дня (24 часа)
        if hasattr(current.name, 'to_pydatetime') and hasattr(position['open_time'], 'to_pydatetime'):
            position_duration = (current.name.to_pydatetime() - position['open_time'].to_pydatetime()).total_seconds() / 3600  # в часах
            if position_duration > 24:  # Прошло более 24 часов
                return {
                    'type': 'exit',
                    'price': current['close'],
                    'time': current.name,
                    'reason': f"Тайм-аут позиции: {position_duration:.1f} часов (>24ч)"
                }
        
        return None
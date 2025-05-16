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
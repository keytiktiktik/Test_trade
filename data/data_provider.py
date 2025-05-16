"""
Модуль для получения и обработки торговых данных.
Поддерживает получение исторических данных и реальных данных с биржи.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import sys
import os

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_provider')

class DataProvider:
    """Класс для получения и обработки торговых данных"""
    
    def __init__(self, exchange_config):
        """
        Инициализация провайдера данных.
        
        Args:
            exchange_config (dict): Конфигурация биржи
        """
        self.exchange_name = exchange_config['name']
        self.api_key = exchange_config['api_key']
        self.api_secret = exchange_config['api_secret']
        self.testnet = exchange_config.get('testnet', False)
        
        # Инициализация соединения с биржей
        self.exchange = self._initialize_exchange()
        
        # Словарь для кеширования данных
        self.data_cache = {}
        
    def _initialize_exchange(self):
        """Инициализация соединения с биржей"""
        try:
            # Создаем экземпляр класса биржи
            exchange_class = getattr(ccxt, self.exchange_name)
            
            # Настройки для биржи
            exchange_params = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Работаем с фьючерсами
                }
            }
            
            # Для тестнета добавляем дополнительные опции
            if self.testnet:
                if self.exchange_name == 'binance':
                    exchange_params['options']['defaultUrl'] = 'https://testnet.binancefuture.com'
                elif self.exchange_name == 'bybit':
                    exchange_params['urls'] = {'api': 'https://api-testnet.bybit.com'}
            
            exchange = exchange_class(exchange_params)
            
            # Проверка соединения
            if not (self.api_key and self.api_secret):
                        # Загружаем только рынки, без проверки балансов и т.д.
                        exchange.load_markets()
                        logger.info(f"Успешно подключено к бирже {self.exchange_name} в режиме без аутентификации (для бэктестинга)")
            else:
                        # Полная проверка соединения
                        exchange.load_markets()
                        logger.info(f"Успешно подключено к бирже {self.exchange_name} с аутентификацией")
                    
            return exchange
                    
        except Exception as e:
                    logger.error(f"Ошибка подключения к бирже: {e}")
                    # Проверяем, является ли ошибка связанной с аутентификацией
                    if "Invalid Api-Key" in str(e) or "AuthenticationError" in str(e):
                        logger.warning("Ошибка аутентификации. Переключаемся в режим без аутентификации для бэктестинга...")
                        # Пробуем подключиться без аутентификации
                        try:
                            exchange = exchange_class({
                                'enableRateLimit': True
                            })
                            exchange.load_markets()
                            logger.info(f"Успешно подключено к бирже {self.exchange_name} в режиме без аутентификации (fallback)")
                            return exchange
                        except Exception as fallback_error:
                            logger.error(f"Не удалось подключиться даже без аутентификации: {fallback_error}")
                    raise
    
    def get_historical_data(self, symbol, timeframe, since=None, limit=500):
        """
        Получение исторических OHLCV данных.
        
        Args:
            symbol (str): Символ торговой пары
            timeframe (str): Таймфрейм ('1m', '5m', '1h', etc.)
            since (int, optional): Timestamp начала в миллисекундах
            limit (int, optional): Максимальное количество свечей
            
        Returns:
            pd.DataFrame: DataFrame с OHLCV данными
        """
        try:
            # Преобразование даты в timestamp, если передана дата
            if isinstance(since, str):
                since = int(datetime.strptime(since, '%Y-%m-%d').timestamp() * 1000)
            
            # Проверяем, есть ли данные в кеше
            cache_key = f"{symbol}_{timeframe}_{since}_{limit}"
            if cache_key in self.data_cache:
                logger.info(f"Используем кешированные данные для {symbol} {timeframe}")
                return self.data_cache[cache_key]
            
            # Получение данных с биржи
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                
                # В случае ошибки или пустых данных, используем fallback
                if not ohlcv or len(ohlcv) == 0:
                    raise Exception("Нет данных от API")
                    
            except Exception as e:
                logger.warning(f"Не удалось получить данные через API: {e}. Используем симулированные данные для бэктестинга.")
                
                # Генерируем некоторые симулированные данные для тестирования
                # Это только для демонстрации, в реальности здесь лучше использовать локальные файлы с данными
                ohlcv = self._generate_test_data(symbol, timeframe, limit)
            
            # Преобразование в DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Сохраняем в кеш
            self.data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка получения исторических данных: {e}")
            return None
        
    def _generate_test_data(self, symbol, timeframe, limit=500):
        """
        Генерация тестовых данных для бэктестинга.
        
        Args:
            symbol (str): Символ торговой пары
            timeframe (str): Таймфрейм
            limit (int): Количество свечей
            
        Returns:
            list: Список с OHLCV данными
        """
        import random
        import numpy as np
        
        # Базовая цена для разных активов
        base_prices = {
            'BTCUSDT': 30000,
            'ETHUSDT': 2000,
            'BNBUSDT': 300,
            'DOGEUSDT': 0.1,
            'ADAUSDT': 0.5
        }
        
        # Используем известную базовую цену или генерируем
        base_price = base_prices.get(symbol, random.uniform(10, 1000))
        
        # Временной интервал между свечами
        if timeframe == '1m':
            interval = 60 * 1000
        elif timeframe == '5m':
            interval = 5 * 60 * 1000
        elif timeframe == '15m':
            interval = 15 * 60 * 1000
        elif timeframe == '30m':
            interval = 30 * 60 * 1000
        elif timeframe == '1h':
            interval = 60 * 60 * 1000
        elif timeframe == '4h':
            interval = 4 * 60 * 60 * 1000
        elif timeframe == '1d':
            interval = 24 * 60 * 60 * 1000
        else:
            interval = 60 * 60 * 1000  # По умолчанию 1 час
        
        # Начальная временная метка - сегодня минус (limit * interval)
        current_time = int(datetime.now().timestamp() * 1000) - (limit * interval)
        
        # Генерация шума для цены
        price_noise = np.random.normal(0, 1, limit)
        
        # Добавляем тренд и сезонность
        trend = np.linspace(0, 5, limit)  # Восходящий тренд
        seasonality = np.sin(np.linspace(0, 10, limit))  # Сезонный компонент
        
        # Комбинируем компоненты
        price_movement = price_noise + trend + seasonality
        
        # Нормализуем движение цены
        price_movement = (price_movement - price_movement.min()) / (price_movement.max() - price_movement.min()) * 2 - 1
        
        # Генерация OHLCV данных
        ohlcv_data = []
        for i in range(limit):
            # Временная метка
            timestamp = current_time + i * interval
            
            # Процентное изменение цены
            change_percent = price_movement[i] * 0.02  # Максимум 2% изменения
            
            # Цена открытия
            if i == 0:
                open_price = base_price
            else:
                open_price = ohlcv_data[i-1][4]  # Цена закрытия предыдущей свечи
            
            # Цена закрытия
            close_price = open_price * (1 + change_percent)
            
            # Максимум и минимум
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
            
            # Объем
            volume = base_price * random.uniform(10, 100)
            
            # Добавляем свечу в список
            ohlcv_data.append([timestamp, open_price, high_price, low_price, close_price, volume])
        
        return ohlcv_data
    
    def get_complete_historical_data(self, symbol, timeframe, start_date, end_date=None):
        """
        Получение полного набора исторических данных за период.
        
        Args:
            symbol (str): Символ торговой пары
            timeframe (str): Таймфрейм ('1m', '5m', '1h', etc.)
            start_date (str): Дата начала в формате 'YYYY-MM-DD'
            end_date (str, optional): Дата окончания в формате 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame с OHLCV данными
        """
        # Если конечная дата не указана, используем текущую
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Преобразование строк в datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Получение временной разницы в миллисекундах
        since = int(start_dt.timestamp() * 1000)
        until = int(end_dt.timestamp() * 1000)
        
        # Расчет количества интервалов для запроса данных
        timeframe_ms = self._get_timeframe_ms(timeframe)
        limit = 1000  # Максимальное количество свечей за запрос
        
        # Инициализация пустого DataFrame для сбора данных
        all_data = pd.DataFrame()
        
        # Постепенное получение данных
        current_since = since
        while current_since < until:
            # Получение данных за интервал
            df = self.get_historical_data(symbol, timeframe, current_since, limit)
            
            if df is None or len(df) == 0:
                break
                
            # Добавление данных к общему DataFrame
            all_data = pd.concat([all_data, df])
            
            # Удаление дубликатов
            all_data = all_data[~all_data.index.duplicated(keep='first')]
            
            # Вычисление следующей точки начала
            if len(df) < limit:
                break
                
            last_timestamp = df.index[-1].timestamp() * 1000
            current_since = int(last_timestamp + timeframe_ms)
            
            # Пауза для избежания превышения лимита запросов
            time.sleep(1)
        
        # Сортировка данных по времени
        all_data.sort_index(inplace=True)
        
        return all_data
    
    def _get_timeframe_ms(self, timeframe):
        """
        Преобразование таймфрейма в миллисекунды.
        
        Args:
            timeframe (str): Таймфрейм ('1m', '5m', '1h', etc.)
            
        Returns:
            int: Количество миллисекунд в таймфрейме
        """
        # Словарь соответствия таймфреймов и миллисекунд
        timeframe_dict = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        
        return timeframe_dict.get(timeframe, 60 * 60 * 1000)  # По умолчанию 1 час
    
    def get_current_data(self, symbol, limit=1):
        """
        Получение текущих данных для символа.
        
        Args:
            symbol (str): Символ торговой пары
            limit (int, optional): Количество последних элементов
            
        Returns:
            dict: Текущие данные символа
        """
        try:
            # Получение тикера
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Если нужно больше данных, можно использовать order book
            if limit > 1:
                orderbook = self.exchange.fetch_order_book(symbol)
                ticker['orderbook'] = orderbook
            
            return ticker
        
        except Exception as e:
            logger.error(f"Ошибка получения текущих данных: {e}")
            return None
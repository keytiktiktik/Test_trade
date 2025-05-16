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
import random

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
            
            # Проверяем режим работы - для бэктестинга не нужна аутентификация
            if self.api_key and self.api_secret:
                # Настройки для биржи с аутентификацией
                exchange_params = {
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',  # Работаем с фьючерсами
                    }
                }
            else:
                # Настройки для биржи без аутентификации - для бэктестинга
                exchange_params = {
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
            
            # Для бэктестинга загружаем только рынки, без запросов требующих аутентификацию
            if not (self.api_key and self.api_secret):
                # Загружаем только рынки, без проверки балансов и т.д.
                try:
                    exchange.load_markets()
                    logger.info(f"Успешно подключено к бирже {self.exchange_name} в режиме без аутентификации (для бэктестинга)")
                except Exception as e:
                    logger.warning(f"Ошибка загрузки рынков: {e}. Продолжаем без загрузки рынков.")
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
                    logger.info("Продолжаем работу с имитацией биржи для бэктестинга")
                    # Возвращаем пустой объект для бэктестинга
                    return None
            # Для остальных ошибок просто выводим сообщение
            logger.warning("Продолжаем работу без подключения к бирже (только для бэктестинга)")
            return None
    
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
                if self.exchange:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                    
                    # В случае ошибки или пустых данных, используем fallback
                    if not ohlcv or len(ohlcv) == 0:
                        raise Exception("Нет данных от API")
                else:
                    raise Exception("Нет соединения с биржей")
                    
            except Exception as e:
                logger.warning(f"Не удалось получить данные через API: {e}. Используем симулированные данные для бэктестинга.")
                
                # Генерируем некоторые симулированные данные для тестирования
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
        
        # Увеличим количество свечей для детального бэктестинга
        limit = 1000  # Максимальное количество свечей за запрос
        
        # Для коротких таймфреймов можем запросить больше данных
        if timeframe in ['1m', '5m', '15m', '30m']:
            limit = 2000  # Больше данных для короткого таймфрейма
        
        # Инициализация пустого DataFrame для сбора данных
        all_data = pd.DataFrame()
        
        try:
            # Проверяем, есть ли соединение с биржей
            if self.exchange:
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
                    time.sleep(0.5)
            
            # Если данных нет или не удалось получить через API, используем тестовые
            if len(all_data) == 0:
                logger.warning(f"Нет данных от API для {symbol} {timeframe}. Генерируем тестовые данные.")
                test_data = self._generate_test_data(symbol, timeframe, limit*3)
                all_data = pd.DataFrame(test_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                all_data['timestamp'] = pd.to_datetime(all_data['timestamp'], unit='ms')
                all_data.set_index('timestamp', inplace=True)
            
        except Exception as e:
            logger.error(f"Ошибка получения исторических данных: {e}")
            # Фолбэк на тестовые данные
            test_data = self._generate_test_data(symbol, timeframe, limit*3)
            all_data = pd.DataFrame(test_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            all_data['timestamp'] = pd.to_datetime(all_data['timestamp'], unit='ms')
            all_data.set_index('timestamp', inplace=True)
        
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
            # Проверяем, есть ли соединение с биржей
            if self.exchange:
                # Получение тикера
                ticker = self.exchange.fetch_ticker(symbol)
                
                # Если нужно больше данных, можно использовать order book
                if limit > 1:
                    orderbook = self.exchange.fetch_order_book(symbol)
                    ticker['orderbook'] = orderbook
                
                return ticker
            else:
                # Генерируем тестовые данные, если нет соединения
                last_data = self._generate_test_data(symbol, '1m', 1)[0]
                ticker = {
                    'symbol': symbol,
                    'timestamp': last_data[0],
                    'datetime': datetime.fromtimestamp(last_data[0]/1000).isoformat(),
                    'high': last_data[2],
                    'low': last_data[3],
                    'bid': last_data[4] * 0.999,
                    'ask': last_data[4] * 1.001,
                    'vwap': last_data[4],
                    'open': last_data[1],
                    'close': last_data[4],
                    'last': last_data[4],
                    'change': last_data[4] - last_data[1],
                    'percentage': ((last_data[4] / last_data[1]) - 1) * 100,
                    'baseVolume': last_data[5],
                    'quoteVolume': last_data[5] * last_data[4],
                }
                return ticker
        
        except Exception as e:
            logger.error(f"Ошибка получения текущих данных: {e}")
            # Генерируем тестовые данные при ошибке
            last_data = self._generate_test_data(symbol, '1m', 1)[0]
            ticker = {
                'symbol': symbol,
                'timestamp': last_data[0],
                'datetime': datetime.fromtimestamp(last_data[0]/1000).isoformat(),
                'high': last_data[2],
                'low': last_data[3],
                'bid': last_data[4] * 0.999,
                'ask': last_data[4] * 1.001,
                'vwap': last_data[4],
                'open': last_data[1],
                'close': last_data[4],
                'last': last_data[4],
                'change': last_data[4] - last_data[1],
                'percentage': ((last_data[4] / last_data[1]) - 1) * 100,
                'baseVolume': last_data[5],
                'quoteVolume': last_data[5] * last_data[4],
            }
            return ticker
    
    def _generate_test_data(self, symbol, timeframe, limit=500):
        """
        Генерация уникальных тестовых данных для каждого таймфрейма.
        
        Args:
            symbol (str): Символ торговой пары
            timeframe (str): Таймфрейм
            limit (int): Количество свечей
            
        Returns:
            list: Список с OHLCV данными
        """
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
        
        # Временной интервал между свечами и уникальный сид для каждого таймфрейма
        if timeframe == '1m':
            interval = 60 * 1000
            seed = 1001
            volatility_factor = 1.5  # Более высокая волатильность для младших таймфреймов
        elif timeframe == '5m':
            interval = 5 * 60 * 1000
            seed = 1002
            volatility_factor = 1.4
        elif timeframe == '15m':
            interval = 15 * 60 * 1000
            seed = 1003
            volatility_factor = 1.3
        elif timeframe == '30m':
            interval = 30 * 60 * 1000
            seed = 1004
            volatility_factor = 1.2
        elif timeframe == '1h':
            interval = 60 * 60 * 1000
            seed = 1005
            volatility_factor = 1.0
        elif timeframe == '4h':
            interval = 4 * 60 * 60 * 1000
            seed = 1006
            volatility_factor = 0.8
        elif timeframe == '1d':
            interval = 24 * 60 * 60 * 1000
            seed = 1007
            volatility_factor = 0.6  # Низкая волатильность для старших таймфреймов
        else:
            interval = 60 * 60 * 1000
            seed = 1008
            volatility_factor = 1.0
        
        # Установка сида для воспроизводимости, но уникальной для каждого таймфрейма
        np.random.seed(seed)
        random.seed(seed)
        
        # Начальная временная метка - сегодня минус (limit * interval)
        days_back = 30  # Данные за последний месяц (можно регулировать)
        current_time = int(datetime.now().timestamp() * 1000) - (days_back * 24 * 60 * 60 * 1000)
        
        # Генерация шума для цены с разной волатильностью для разных таймфреймов
        price_noise = np.random.normal(0, volatility_factor, limit)
        
        # Добавляем тренд и сезонность
        trend = np.linspace(0, 3, limit)  # Умеренный восходящий тренд
        
        # Добавим несколько циклов колебаний, особенно для младших таймфреймов
        cycles_per_period = {
            '1m': 20,
            '5m': 15,
            '15m': 10,
            '30m': 8,
            '1h': 6,
            '4h': 4,
            '1d': 2
        }
        cycles = cycles_per_period.get(timeframe, 6)
        seasonality = np.sin(np.linspace(0, cycles * np.pi, limit))
        
        # Добавим микрорендомные шоки для имитации рыночных событий
        shocks = np.zeros(limit)
        num_shocks = int(limit / 50)  # Примерно 1 шок на каждые 50 свечей
        shock_positions = np.random.choice(limit, num_shocks, replace=False)
        shock_magnitudes = np.random.normal(0, 3, num_shocks)  # Большие шоки для реалистичности
        
        for i, pos in enumerate(shock_positions):
            # Длительность шока (короче для младших таймфреймов)
            shock_duration = int(10 / (cycles_per_period.get(timeframe, 6) / 6))
            
            # Применяем шок и его затухание
            for j in range(min(shock_duration, limit - pos)):
                decay = 1 - (j / shock_duration)
                if pos + j < limit:
                    shocks[pos + j] = shock_magnitudes[i] * decay
        
        # Комбинируем компоненты
        price_movement = price_noise + trend + seasonality + shocks
        
        # Обеспечиваем что данные начинаются и заканчиваются примерно в том же диапазоне
        # для связности между разными таймфреймами
        initial_price = base_price
        final_adjustment = initial_price - (base_price * (1 + price_movement[-1] * 0.02))
        adjustment_array = np.linspace(0, final_adjustment, limit)
        
        # Генерация OHLCV данных
        ohlcv_data = []
        current_price = initial_price
        
        for i in range(limit):
            # Временная метка
            timestamp = current_time + i * interval
            
            # Процентное изменение цены, уникальное для каждого таймфрейма
            change_percent = price_movement[i] * 0.02
            
            # Цена открытия
            if i == 0:
                open_price = current_price
            else:
                open_price = ohlcv_data[i-1][4]  # Цена закрытия предыдущей свечи
            
            # Цена закрытия
            close_price = open_price * (1 + change_percent)
            
            # Максимум и минимум - более широкий диапазон для младших таймфреймов
            range_factor = volatility_factor * 0.01
            high_price = max(open_price, close_price) * (1 + random.uniform(0, range_factor))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, range_factor))
            
            # Объем - выше на младших таймфреймах
            volume_base = base_price * random.uniform(10, 100)
            volume_factor = 1.5 if timeframe in ['1m', '5m', '15m'] else 1.0
            volume = volume_base * volume_factor
            
            # Добавляем свечу в список
            ohlcv_data.append([timestamp, open_price, high_price, low_price, close_price, volume])
        
        return ohlcv_data
    
    def _load_historical_data_from_file(self, symbol, timeframe):
        """
        Загрузка исторических данных из локального файла.
        
        Args:
            symbol (str): Символ торговой пары
            timeframe (str): Таймфрейм
            
        Returns:
            list: Список с OHLCV данными
        """
        # Путь к файлу
        file_path = f"data/historical/{symbol}_{timeframe}.csv"
        
        try:
            if os.path.exists(file_path):
                # Загрузка данных из CSV
                df = pd.read_csv(file_path)
                
                # Преобразование в формат OHLCV
                ohlcv = []
                for _, row in df.iterrows():
                    timestamp = int(pd.Timestamp(row['timestamp']).timestamp() * 1000)
                    ohlcv.append([
                        timestamp,
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    ])
                
                return ohlcv
            else:
                logger.warning(f"Файл с историческими данными не найден: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Ошибка загрузки исторических данных из файла: {e}")
            return None
"""
Модуль для визуализации результатов бэктестинга и торговых сигналов.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import sys

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('visualizer')

class Visualizer:
    """Класс для визуализации результатов бэктестинга и торговых сигналов"""
    
    def __init__(self, save_path=None):
        """
        Инициализация визуализатора.
        
        Args:
            save_path (str, optional): Путь для сохранения графиков
        """
        self.save_path = save_path
        
        # Создание директории для сохранения, если ее нет
        if self.save_path and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def plot_equity_curve(self, equity_data, title="Equity Curve"):
        """
        Построение графика кривой капитала.
        
        Args:
            equity_data (list): Список словарей с данными equity curve
            title (str, optional): Заголовок графика
            
        Returns:
            fig: Объект графика
        """
        df = pd.DataFrame(equity_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df['timestamp'], df['equity'], label='Equity', color='blue', linewidth=2)
        ax.plot(df['timestamp'], df['balance'], label='Balance', color='green', linewidth=1.5, linestyle='--')
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('USD')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Форматирование оси X для отображения дат
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Поворот меток для лучшей читаемости
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(f"{self.save_path}/equity_curve.png", dpi=300)
        
        return fig
    
    def plot_drawdown(self, equity_data, title="Drawdown"):
        """
        Построение графика просадки.
        
        Args:
            equity_data (list): Список словарей с данными equity curve
            title (str, optional): Заголовок графика
            
        Returns:
            fig: Объект графика
        """
        df = pd.DataFrame(equity_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Расчет просадки
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['peak'] - df['equity']) / df['peak'] * 100  # в процентах
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(df['timestamp'], df['drawdown'], 0, color='red', alpha=0.3)
        ax.plot(df['timestamp'], df['drawdown'], color='red', linewidth=2)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.invert_yaxis()  # Инвертирование оси Y для более наглядного отображения
        ax.grid(True, alpha=0.3)
        
        # Форматирование оси X для отображения дат
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Форматирование оси Y для отображения процентов
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        
        # Поворот меток для лучшей читаемости
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(f"{self.save_path}/drawdown.png", dpi=300)
        
        return fig
    
    def plot_price_chart_with_trades(self, price_data, trades, indicators=None, title="Price Chart with Trades"):
        """
        Построение графика цены с отметками сделок и индикаторами.
        
        Args:
            price_data (pd.DataFrame): DataFrame с OHLCV данными
            trades (list): Список словарей с данными о сделках
            indicators (dict, optional): Словарь с индикаторами для отображения
            title (str, optional): Заголовок графика
            
        Returns:
            fig: Объект графика
        """
        # Matplotlib версия
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Построение OHLC графика
        width = 0.8
        width2 = 0.1
        
        up = price_data[price_data.close >= price_data.open]
        down = price_data[price_data.close < price_data.open]
        
        # Зеленые свечи (закрытие >= открытия)
        ax1.bar(up.index, up.close-up.open, width, bottom=up.open, color='green', alpha=0.5)
        ax1.bar(up.index, up.high-up.close, width2, bottom=up.close, color='green', alpha=0.5)
        ax1.bar(up.index, up.low-up.open, width2, bottom=up.open, color='green', alpha=0.5)
        
        # Красные свечи (закрытие < открытия)
        ax1.bar(down.index, down.open-down.close, width, bottom=down.close, color='red', alpha=0.5)
        ax1.bar(down.index, down.high-down.open, width2, bottom=down.open, color='red', alpha=0.5)
        ax1.bar(down.index, down.low-down.close, width2, bottom=down.close, color='red', alpha=0.5)
        
        # Добавление индикаторов, если они есть
        if indicators:
            for name, data in indicators.items():
                if isinstance(data, pd.Series):
                    ax1.plot(data.index, data.values, label=name)
                elif isinstance(data, tuple) and len(data) >= 2:
                    # Для полос Боллинджера или других подобных индикаторов с несколькими линиями
                    for i, line in enumerate(data):
                        ax1.plot(line.index, line.values, label=f"{name}_{i}")
        
        # Добавление точек входа и выхода
        for trade in trades:
            # Точка входа (зеленый треугольник вверх для покупки, красный треугольник вниз для продажи)
            if trade['type'] == 'buy':
                ax1.scatter(trade['open_time'], trade['entry_price'], 
                           marker='^', color='green', s=100, zorder=5)
            else:  # sell
                ax1.scatter(trade['open_time'], trade['entry_price'], 
                           marker='v', color='red', s=100, zorder=5)
            
            # Точка выхода (крестик)
            ax1.scatter(trade['close_time'], trade['exit_price'], 
                       marker='x', color='black', s=100, zorder=5)
        
        # Построение графика объема
        ax2.bar(price_data.index, price_data.volume, width, color='blue', alpha=0.5)
        
        # Настройка графиков
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        if indicators:
            ax1.legend()
        
        # Форматирование оси X для отображения дат
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Поворот меток для лучшей читаемости
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(f"{self.save_path}/price_chart_with_trades.png", dpi=300)
        
        return fig
    
    def create_trade_statistics_table(self, backtest_results):
        """
        Создание таблицы со статистикой торговли.
        
        Args:
            backtest_results (dict): Результаты бэктестинга
            
        Returns:
            fig: Объект графика с таблицей
        """
        stats = backtest_results['stats']
        trades = backtest_results['trades']
        
        # Общая статистика
        overall_stats = [
            ['Initial Balance', f"${backtest_results['initial_balance']:.2f}"],
            ['Final Balance', f"${backtest_results['final_balance']:.2f}"],
            ['Profit/Loss', f"${backtest_results['final_balance'] - backtest_results['initial_balance']:.2f}"],
            ['Profit %', f"{backtest_results['profit_percent']:.2f}%"],
            ['Total Trades', f"{stats['total_trades']}"],
            ['Win Rate', f"{stats['win_rate']:.2f}%"],
            ['Profit Factor', f"{stats['profit_factor']:.2f}"],
            ['Max Drawdown', f"{stats['max_drawdown_percent']:.2f}%"],
            ['Sharpe Ratio', f"{stats['sharpe_ratio']:.2f}"],
            ['Sortino Ratio', f"{stats['sortino_ratio']:.2f}"]
        ]
        
        # Статистика по стратегиям
        strategy_stats = {}
        for trade in trades:
            strategy = trade['strategy']
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit': 0,
                    'loss': 0
                }
            
            strategy_stats[strategy]['count'] += 1
            
            if trade['net_pnl'] > 0:
                strategy_stats[strategy]['wins'] += 1
                strategy_stats[strategy]['profit'] += trade['net_pnl']
            else:
                strategy_stats[strategy]['losses'] += 1
                strategy_stats[strategy]['loss'] += abs(trade['net_pnl'])
        
        # Создание таблицы с Matplotlib
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Таблица общей статистики
        ax1.axis('off')
        ax1.set_title('Trading System Performance Summary', fontsize=16)
        
        table1 = ax1.table(
            cellText=overall_stats,
            colLabels=['Metric', 'Value'],
            loc='center',
            cellLoc='center',
            colWidths=[0.5, 0.5]
        )
        
        table1.auto_set_font_size(False)
        table1.set_fontsize(12)
        table1.scale(1, 1.5)
        
        # Таблица статистики по стратегиям
        ax2.axis('off')
        ax2.set_title('Strategy Performance', fontsize=16)
        
        strategy_table_data = []
        for strategy, stats in strategy_stats.items():
            win_rate = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
            profit_factor = stats['profit'] / stats['loss'] if stats['loss'] > 0 else float('inf')
            
            strategy_table_data.append([
                strategy,
                f"{stats['count']}",
                f"{win_rate:.2f}%",
                f"${stats['profit'] - stats['loss']:.2f}",
                f"{profit_factor:.2f}"
            ])
        
        table2 = ax2.table(
            cellText=strategy_table_data,
            colLabels=['Strategy', 'Trades', 'Win Rate', 'Net Profit', 'Profit Factor'],
            loc='center',
            cellLoc='center',
            colWidths=[0.3, 0.15, 0.15, 0.2, 0.2]
        )
        
        table2.auto_set_font_size(False)
        table2.set_fontsize(12)
        table2.scale(1, 1.5)
        
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(f"{self.save_path}/trade_statistics.png", dpi=300)
        
        return fig
    
    def plot_all_results(self, backtest_results, price_data):
        """
        Создание всех графиков результатов бэктестинга.
        
        Args:
            backtest_results (dict): Результаты бэктестинга
            price_data (pd.DataFrame): DataFrame с OHLCV данными
            
        Returns:
            tuple: (equity_fig, drawdown_fig, price_chart_fig, stats_fig)
        """
        # График кривой капитала
        equity_fig = self.plot_equity_curve(backtest_results['equity_curve'], title="Equity Curve")
        
        # График просадки
        drawdown_fig = self.plot_drawdown(backtest_results['equity_curve'], title="Drawdown")
        
        # График цены с сделками
        price_chart_fig = self.plot_price_chart_with_trades(price_data, backtest_results['trades'], title="Price Chart with Trades")
        
        # Таблица статистики
        stats_fig = self.create_trade_statistics_table(backtest_results)
        
        return equity_fig, drawdown_fig, price_chart_fig, stats_fig
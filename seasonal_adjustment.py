import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

import panel_utils
from panel_utils import seasonal_adjust_panel_series

def seasonal_adjust_panel_wrapper(df, col_name):
    # ===== СЕЗОННАЯ КОРРЕКТИРОВКА ДЛЯ ПАНЕЛЬНЫХ ДАННЫХ =====
    
    # Заглушаем предупреждения
    warnings.filterwarnings('ignore')
    
    import json
    from pathlib import Path
    
    cfg = json.loads(Path("x13.json").read_text(encoding="utf-8"))
    x13_path = cfg["x13_path"]
    
    # Путь к бинарнику X-13
    X13_PATH = x13_path
    
    # Проверка, что бинарник доступен
    if not os.path.exists(X13_PATH):
        raise FileNotFoundError(f"X-13 binary not found at {X13_PATH}")
    
    # Если X13_PATH — полный путь к бинарнику, берем его директорию для x12path
    x13_dir = X13_PATH if os.path.isdir(X13_PATH) else os.path.dirname(X13_PATH)
    panel_utils.x13_dir = x13_dir
    
    # Создаем папку для логов
    os.makedirs("x13_logs", exist_ok=True)
    
    # ============ ОПРЕДЕЛЕНИЕ ПЕРЕМЕННЫХ ДЛЯ КОРРЕКТИРОВКИ ============
    
    # ФЕДЕРАЛЬНЫЕ переменные (одна серия для всех регионов)
    federal_variables = {
        'Exc_rate': 'Курс рубля', 
        'Inflation_Expectations': 'Инфляционные ожидания',
        'Bonds_Rate_Correct_5Y': 'Ставки на облигации скорректированные 5Y'
    }
    
    # РЕГИОНАЛЬНЫЕ переменные (серии по каждому региону)
    regional_variables = {
        'Int_Rate_FL': 'Ставка на кредиты ФЛ',
        'Int_Rate_FL_lag1' : 'Ставка на кредиты ФЛ лаг',
        'Int_Rate_Mort': 'Ставка на ипотеку',
        'Int_Rate_Mort_lag1': 'Ставка на ипотеку лаг',
        'Credit_impulse': 'Кредитный импульс',
        'Int_Rate_ConsCred': 'Ставка на потребительский кредит',
        'Int_Rate_ConsCred_lag1': 'Ставка на потребительский кредит лаг',
        'Cred_nagr': 'Кредитная нагрузка',
        'D_top5_rozn': 'Доля топ-5 банков в розничном портфеле региона',
        'Fin_Dostup':'Кол-во пунктов банковского обслуживания на 100к чел.',
        'Cred_structure':'Доля ипотечных жилищных кредитов в розничном портфеле',
        'Def_Zadolg_Fl':'Доля просроченной задолженности по кредитам физических лиц',
        'Def_Zadolg_Mort':'Доля просроченной задолженности по ипотечным жилищным кредитам',
        'Def_Zadolg_ConsCred':'Доля просроченной задолженности по потребительским кредитам'
        
    }
    
    for col in col_name:
        if col in federal_variables or col in regional_variables:
            continue
        if col in df.columns:
            if 'Date' in df.columns:
                unique_counts = df.groupby('Date')[col].nunique(dropna=True)
                if (unique_counts > 1).any():
                    regional_variables[col] = col
                else:
                    federal_variables[col] = col
            else:
                regional_variables[col] = col
        else:
            federal_variables[col] = col

    federal_variables_filtered = {}
    for col, name in federal_variables.items():
        if col in col_name:
            federal_variables_filtered[col] = name
    federal_variables = federal_variables_filtered
    
    regional_variables_filtered = {}
    for col, name in regional_variables.items():
        if col in col_name:
            regional_variables_filtered[col] = name
    regional_variables = regional_variables_filtered
    
    # ============ ФУНКЦИЯ ДЛЯ СЕЗОННОЙ КОРРЕКТИРОВКИ ОДНОЙ СЕРИИ ============
    
    
    # ============ ОБРАБОТКА ФЕДЕРАЛЬНЫХ ПЕРЕМЕННЫХ ============
    
    print("\n" + "="*70)
    print("СЕЗОННАЯ КОРРЕКТИРОВКА ФЕДЕРАЛЬНЫХ ПЕРЕМЕННЫХ")
    print("="*70)
    
    federal_adjusted = {}
    
    for col, name in federal_variables.items():
        if col not in df.columns:
            print(f"\n  {name} ({col}): столбец не найден. Пропуск.")
            continue
        
        print(f"\n{name} ({col}):")
        
        # Для федеральных переменных берем уникальные значения по датам
        fed_data = df[['Date', col]].copy().dropna()
        
        if len(fed_data) == 0:
            print(f"  ✗ Нет данных. Пропуск.")
            continue
        
        # Убедиться, что одно значение на дату
        fed_data = fed_data.drop_duplicates('Date')
        fed_data = fed_data.set_index('Date')[col].sort_index()
        
        # Сезонная корректировка
        result = seasonal_adjust_panel_series(fed_data, col, region_name=None)
        
        if result is not None:
            federal_adjusted[col] = result
            print(f"  ✓ Успешно скорректировано, снижение волатильности: {result['reduction']:.1f}%")
            
            # Добавляем результаты обратно в df
            for date, adj_val, trend_val, seas_val in zip(
                result['dates'], result['adj'], result['trend'], result['seasonal']
            ):
                mask = df['Date'] == date
                df.loc[mask, f'{col}_adj'] = adj_val
                df.loc[mask, f'{col}_trend'] = trend_val
                df.loc[mask, f'{col}_seasonal'] = seas_val
        else:
            print(f"  ✗ Не удалось скорректировать")
    
    # ============ ОБРАБОТКА РЕГИОНАЛЬНЫХ ПЕРЕМЕННЫХ ============
    
    print("\n" + "="*70)
    print("СЕЗОННАЯ КОРРЕКТИРОВКА РЕГИОНАЛЬНЫХ ПЕРЕМЕННЫХ")
    print("="*70)
    
    regional_adjusted = {}
    
    for col, name in regional_variables.items():
        if col not in df.columns:
            print(f"\n  {name} ({col}): столбец не найден. Пропуск.")
            continue
        
        print(f"\n{name} ({col}):")
        
        # Получаем список регионов
        regions = df['Region'].unique()
        print(f"  Обрабатываю {len(regions)} регионов...")
        
        regional_adjusted[col] = {}
        successful_regions = 0
        
        for region in regions:
            # Фильтруем данные по региону
            region_data = df[df['Region'] == region][['Date', col]].copy().dropna()
            
            if len(region_data) < 24:
                continue  # Пропускаем регионы с недостаточным количеством данных
            
            # Подготавливаем временной ряд
            region_series = region_data.set_index('Date')[col].sort_index()
            
            # Сезонная корректировка
            result = seasonal_adjust_panel_series(region_series, col, region_name=region, min_obs=24)
            
            if result is not None:
                regional_adjusted[col][region] = result
                successful_regions += 1
                
                # Добавляем результаты в df
                for date, adj_val, trend_val, seas_val in zip(
                    result['dates'], result['adj'], result['trend'], result['seasonal']
                ):
                    mask = (df['Region'] == region) & (df['Date'] == date)
                    if mask.any():
                        df.loc[mask, f'{col}_adj'] = adj_val
                        df.loc[mask, f'{col}_trend'] = trend_val
                        df.loc[mask, f'{col}_seasonal'] = seas_val
        
        print(f"  ✓ Успешно скорректировано {successful_regions}/{len(regions)} регионов")
    
    # ============ ПРОВЕРКА ДОБАВЛЕННЫХ СТОЛБЦОВ ============
    
    print("\n" + "="*70)
    print("ПРОВЕРКА ДОБАВЛЕННЫХ СТОЛБЦОВ")
    print("="*70)
    
    all_variables = list(regional_variables.keys()) + list(federal_variables.keys())
    added_columns = []
    
    for col in all_variables:
        for suffix in ['_adj', '_trend', '_seasonal']:
            new_col = f'{col}{suffix}'
            if new_col in df.columns:
                added_columns.append(new_col)
                non_na_count = df[new_col].notna().sum()
                print(f"✓ {new_col}: {non_na_count} не-NaN значений")
    
    print(f"\n✓ Всего добавлено столбцов: {len(added_columns)}")
    
    # ============ СВОДКА РЕЗУЛЬТАТОВ ============
    
    print("\n" + "="*70)
    print("СВОДКА РЕЗУЛЬТАТОВ СЕЗОННОЙ КОРРЕКТИРОВКИ")
    print("="*70)
    
    # Федеральные переменные
    print("\nФедеральные переменные:")
    print("-" * 50)
    print(f"{'Переменная':<30} {'Волатильность ↓':<15}")
    print("-" * 50)
    
    for col in federal_variables:
        adj_col = f'{col}_adj'
        if adj_col in df.columns and col in federal_adjusted:
            reduction = federal_adjusted[col]['reduction']
            print(f"{federal_variables[col]:<30} {reduction:>14.1f}%")
    
    # Региональные переменные (среднее по регионам)
    print("\n\nРегиональные переменные (среднее по регионам):")
    print("-" * 50)
    print(f"{'Переменная':<30} {'Волатильность ↓':<15}")
    print("-" * 50)
    
    for col in regional_variables:
        adj_col = f'{col}_adj'
        if adj_col in df.columns and col in regional_adjusted:
            # Рассчитываем среднее снижение волатильности
            reductions = []
            for region_data in regional_adjusted[col].values():
                reductions.append(region_data['reduction'])
            
            if reductions:
                avg_reduction = np.mean(reductions)
                print(f"{regional_variables[col]:<30} {avg_reduction:>14.1f}%")
    
    # ============ ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ============
    
    print("\n" + "="*70)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (ПРИМЕРЫ)")
    print("="*70)
    
    # Выбираем несколько регионов для визуализации
    sample_regions = df['Region'].unique()[:3]  # Первые 3 региона
    
    # Визуализация для нескольких переменных
    for col in list(regional_variables.keys())[:2]:  # Первые 2 региональные переменные
        adj_col = f'{col}_adj'
        
        if adj_col not in df.columns:
            continue
        
        for region in sample_regions:
            # Фильтруем данные для региона
            region_mask = (df['Region'] == region) & df[adj_col].notna()
            region_data = df[region_mask]
            
            if len(region_data) == 0:
                continue
            
            # Создаем график
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            
            # График 1: Исходный vs скорректированный
            ax = axes[0, 0]
            dates = region_data['Date']
            ax.plot(dates, region_data[col], label='Исходный', alpha=0.6, linewidth=1)
            ax.plot(dates, region_data[adj_col], label='Сезонно скорр.', linewidth=1.5)
            ax.set_title(f'{regional_variables[col]}\nРегион: {region}')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # График 2: Сезонная компонента
            ax = axes[0, 1]
            seasonal_col = f'{col}_seasonal'
            if seasonal_col in region_data.columns:
                ax.plot(dates, region_data[seasonal_col], color='red', linewidth=1.5)
                ax.axhline(0, color='black', linestyle='--', alpha=0.3)
                ax.set_title('Сезонная компонента')
                ax.grid(alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            # График 3: Тренд
            ax = axes[1, 0]
            trend_col = f'{col}_trend'
            if trend_col in region_data.columns:
                ax.plot(dates, region_data[trend_col], color='green', linewidth=1.5)
                ax.set_title('Тренд')
                ax.grid(alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            # График 4: Статистика
            ax = axes[1, 1]
            ax.axis('off')
            
            # Расчет статистик
            if len(region_data) > 0:
                original_std = region_data[col].std()
                adj_std = region_data[adj_col].std()
                reduction = 100 * (1 - adj_std / original_std) if original_std > 0 else 0
                
                stats_text = f"Статистика для {region}:\n\n"
                stats_text += f"Период: {region_data['Date'].min().strftime('%Y-%m')} - {region_data['Date'].max().strftime('%Y-%m')}\n"
                stats_text += f"Наблюдений: {len(region_data)}\n"
                stats_text += f"Снижение волатильности: {reduction:.1f}%\n"
                stats_text += f"Стд. откл. до: {original_std:.4f}\n"
                stats_text += f"Стд. откл. после: {adj_std:.4f}"
                
                ax.text(0.1, 0.5, stats_text, fontsize=10, va='center', linespacing=1.5)
            
            plt.tight_layout()
            
            # Сохраняем график
            region_safe = region.replace('/', '_').replace('\\', '_').replace(':', '_')
            filename = f'x13_logs/panel_{col}_{region_safe}_seasonal.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()
    
    # Визуализация для федеральных переменных
    print("\nГрафики федеральных переменных...")
    
    for col, name in federal_variables.items():
        adj_col = f'{col}_adj'
        
        if adj_col not in df.columns:
            continue
        
        # Фильтруем данные
        fed_mask = df[adj_col].notna()
        fed_data = df[fed_mask].drop_duplicates('Date')
        
        if len(fed_data) == 0:
            continue
        
        # Создаем график
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # График 1: Исходный vs скорректированный
        ax = axes[0, 0]
        dates = fed_data['Date']
        ax.plot(dates, fed_data[col], label='Исходный', alpha=0.6, linewidth=1)
        ax.plot(dates, fed_data[adj_col], label='Сезонно скорр.', linewidth=1.5)
        ax.set_title(f'{name}\nФедеральная переменная')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # График 2: Сезонная компонента
        ax = axes[0, 1]
        seasonal_col = f'{col}_seasonal'
        if seasonal_col in fed_data.columns:
            ax.plot(dates, fed_data[seasonal_col], color='red', linewidth=1.5)
            ax.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax.set_title('Сезонная компонента')
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # График 3: Тренд
        ax = axes[1, 0]
        trend_col = f'{col}_trend'
        if trend_col in fed_data.columns:
            ax.plot(dates, fed_data[trend_col], color='green', linewidth=1.5)
            ax.set_title('Тренд')
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # График 4: Статистика
        ax = axes[1, 1]
        ax.axis('off')
        
        # Расчет статистик
        if len(fed_data) > 0 and col in federal_adjusted:
            reduction = federal_adjusted[col]['reduction']
            original_std = fed_data[col].std()
            adj_std = fed_data[adj_col].std()
            
            stats_text = f"Статистика для {name}:\n\n"
            stats_text += f"Период: {fed_data['Date'].min().strftime('%Y-%m')} - {fed_data['Date'].max().strftime('%Y-%m')}\n"
            stats_text += f"Наблюдений: {len(fed_data)}\n"
            stats_text += f"Снижение волатильности: {reduction:.1f}%\n"
            stats_text += f"Стд. откл. до: {original_std:.4f}\n"
            stats_text += f"Стд. откл. после: {adj_std:.4f}"
            
            ax.text(0.1, 0.5, stats_text, fontsize=10, va='center', linespacing=1.5)
        
        plt.tight_layout()
        
        # Сохраняем график
        filename = f'x13_logs/federal_{col}_seasonal.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    
    # ============ ФИНАЛЬНАЯ СТАТИСТИКА ============
    
    print("\n" + "="*70)
    print("ФИНАЛЬНАЯ СТАТИСТИКА")
    print("="*70)
    
    print(f"\nОбработано федеральных переменных: {len(federal_adjusted)}/{len(federal_variables)}")
    print(f"Обработано региональных переменных: {len(regional_adjusted)}/{len(regional_variables)}")
    print(f"Общее количество добавленных столбцов: {len(added_columns)}")
    print(f"Форма df: {df.shape}")
    
    # Проверяем наличие столбцов с суффиксами
    adj_columns = [col for col in df.columns if col.endswith('_adj')]
    trend_columns = [col for col in df.columns if col.endswith('_trend')]
    seasonal_columns = [col for col in df.columns if col.endswith('_seasonal')]
    
    print(f"\nСтолбцов _adj: {len(adj_columns)}")
    print(f"Столбцов _trend: {len(trend_columns)}")
    print(f"Столбцов _seasonal: {len(seasonal_columns)}")
    
    print("\n✓ Скорректированные ряды добавлены в df")
    print("✓ Столбцы с суффиксами: _adj (скорр.), _trend (тренд), _seasonal (сезон)")
    
    # Сортируем по дате и региону
    df = df.sort_values(['Region', 'Date']).reset_index(drop=True)
    
    return df


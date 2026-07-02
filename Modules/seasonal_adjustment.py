import json
import os
import tempfile
import uuid
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.x13 import x13_arima_analysis

x13_dir = None


def friedman_seasonality_test(df, date_col='Date', value_col='Variable', 
                              test_name=None):

    df_test = df.copy()
    df_test['Month'] = df_test[date_col].dt.month
    df_test['Year'] = df_test[date_col].dt.year
    
    # Перевод в таблицу: года × месяцы
    pivot_data = df_test.pivot(index='Year', columns='Month', values=value_col)
    
    # Удаляем строки с NaN (неполные годы)
    pivot_data = pivot_data.dropna()
    
    # Проверяем достаточность данных
    if len(pivot_data) < 3:
        print(f"{test_name}: Недостаточно лет ({len(pivot_data)}), минимум 3")
        return {
            'variable': test_name if test_name else value_col,
            'chi_squared': np.nan,
            'p_value': np.nan,
            'has_seasonality': np.nan,
            'n_years': len(pivot_data),
            'n_months': len(pivot_data.columns),
            'error': 'Недостаточно данных'
        }
    
    try:
        # Тест Фридмана
        stat, p_value = friedmanchisquare(*[pivot_data[m].values for m in pivot_data.columns])
        
        # Определяем наличие сезонности
        has_seasonality = p_value < 0.05
        
        return {
            'variable': test_name if test_name else value_col,
            'chi_squared': stat,
            'p_value': p_value,
            'has_seasonality': has_seasonality,
            'n_years': len(pivot_data),
            'n_months': len(pivot_data.columns),
            'error': None
        }
    
    except Exception as e:
        print(f"Ошибка для {test_name}: {str(e)}")
        return {
            'variable': test_name if test_name else value_col,
            'chi_squared': np.nan,
            'p_value': np.nan,
            'has_seasonality': np.nan,
            'n_years': len(pivot_data),
            'n_months': len(pivot_data.columns),
            'error': str(e)
        }


def run_friedman_seasonality_tests(df, date_col='Date'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if date_col in numeric_cols:
        numeric_cols.remove(date_col)

    # Применяем тест
    friedman_results = []

    for var_col in numeric_cols:
        if var_col in df.columns:
            result = friedman_seasonality_test(df, date_col=date_col, 
                                              value_col=var_col, 
                                              test_name=var_col)
            if result:
                friedman_results.append(result)

    # Сохраняем результаты Фридмана
    friedman_summary = pd.DataFrame([
        {
            'Variable': r['variable'],
            'Chi_Squared': r['chi_squared'],
            'P_Value': r['p_value'],
            'Has_Seasonality': r['has_seasonality'],
            'N_Years': r['n_years'],
            'N_Months': r['n_months']
        } 
        for r in friedman_results
    ])

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТОВ ФРИДМАНА")
    print("="*80)
    print(friedman_summary.to_string(index=False))

    return friedman_summary


def run_seasonal_diagnostics(df, date_col, value_col, shock_year=None):
    print("\n" + "=" * 80)
    print(f"Diagnostics for {value_col}")
    print("=" * 80)

    tmp_full = df[[date_col, value_col]].dropna().copy()
    tmp_full[date_col] = pd.to_datetime(tmp_full[date_col])
    tmp_full = tmp_full.sort_values(date_col)

    shock_year_int = None
    if shock_year is not None:
        try:
            shock_year_int = int(str(shock_year))
        except Exception:
            shock_year_int = None

    tmp_stats = tmp_full
    if shock_year_int is not None:
        tmp_stats = tmp_full[tmp_full[date_col].dt.year != shock_year_int].copy()
        years_left = tmp_stats[date_col].dt.year.nunique()
        print(f"Shock filter applied: {shock_year_int}; rows={len(tmp_stats)}, years={years_left}")
    else:
        years_full = tmp_full[date_col].dt.year.nunique()
        if shock_year is None:
            print(f"Shock filter not applied: rows={len(tmp_full)}, years={years_full}")
        else:
            print(f"Shock filter not applied (invalid): rows={len(tmp_full)}, years={years_full}")

    if tmp_stats.empty:
        print("No data after filtering; diagnostics skipped.")
        return

    try:
        friedman_res = friedman_seasonality_test(
            tmp_stats,
            date_col=date_col,
            value_col=value_col,
            test_name=value_col
        )
        print("Friedman test result:", friedman_res)
    except Exception as e:
        print("Friedman test error:", e)

    tmp_stats["Month"] = tmp_stats[date_col].dt.month
    month_stats = tmp_stats.groupby("Month")[value_col].agg(["mean", "median", "var", "count"])
    print("\nMonth-of-year mean/median:")
    print(month_stats[["mean", "median"]].to_string())

    var_min = month_stats["var"].min()
    var_max = month_stats["var"].max()
    if pd.isna(var_min) or pd.isna(var_max) or var_min == 0:
        var_ratio = np.nan
    else:
        var_ratio = var_max / var_min
    print(f"\nStability indicator (month variance ratio max/min): {var_ratio}")

    pivot = tmp_stats.pivot_table(
        index=tmp_stats[date_col].dt.year,
        columns="Month",
        values=value_col,
        aggfunc="mean"
    )
    plt.figure(figsize=(12, 5))
    for m in pivot.columns:
        plt.plot(pivot.index, pivot[m], marker="o", linewidth=1, label=str(m))
    plt.title(f"{value_col} seasonal subseries (by month)")
    plt.xlabel("Year")
    plt.ylabel(value_col)
    plt.legend(ncol=6, fontsize=7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.boxplot(x="Month", y=value_col, data=tmp_stats)
    plt.title(f"{value_col} month boxplot")
    plt.tight_layout()
    plt.show()

    def _run_stl_acf_block(block_df, label=None):
        s = block_df.set_index(date_col)[value_col].sort_index()
        s = s.asfreq("MS")
        if s.isna().any():
            if label:
                print(f"{label} STL/ACF skipped: gappy series")
            else:
                print("STL/ACF skipped: gappy series")
            return
        s = s.dropna()
        if len(s) < 24:
            if label:
                print(f"{label} STL/ACF skipped: not enough observations")
            else:
                print("STL/ACF skipped: not enough observations")
            return

        label_suffix = f" ({label})" if label else ""
        try:
            stl = STL(s, period=12, robust=True)
            res = stl.fit()
            res.plot()
            plt.suptitle(f"{value_col} STL decomposition{label_suffix}")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            if label:
                print(f"{label} STL error:", e)
            else:
                print("STL error:", e)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        try:
            plot_acf(s.dropna(), lags=36, ax=axes[0])
            axes[0].set_title(f"{value_col} ACF{label_suffix}")
        except Exception as e:
            axes[0].set_title(f"{value_col} ACF error{label_suffix}")
            if label:
                print(f"{label} ACF error:", e)
            else:
                print("ACF error:", e)

        try:
            s_seasonal = s.diff(12).dropna()
            plot_acf(s_seasonal, lags=36, ax=axes[1])
            axes[1].set_title(f"{value_col} seasonal ACF (diff 12){label_suffix}")
        except Exception as e:
            axes[1].set_title(f"{value_col} seasonal ACF error{label_suffix}")
            if label:
                print(f"{label} Seasonal ACF error:", e)
            else:
                print("Seasonal ACF error:", e)

        plt.tight_layout()
        plt.show()

    if shock_year_int is None:
        _run_stl_acf_block(tmp_full)
    else:
        pre_cut = pd.Timestamp(shock_year_int, 1, 1)
        post_cut = pd.Timestamp(shock_year_int + 1, 1, 1)
        pre_block = tmp_full[tmp_full[date_col] < pre_cut]
        post_block = tmp_full[tmp_full[date_col] >= post_cut]
        _run_stl_acf_block(pre_block, "pre-shock")
        _run_stl_acf_block(post_block, "post-shock")


def seasonal_adjust_panel_series(series_data, variable_name, region_name=None, min_obs=24):
    """
    Выполнить X-13 сезонную корректировку для одной серии (панельные данные)
    
    Parameters:
    -----------
    series_data : pd.Series
        Временной ряд с DatetimeIndex
    variable_name : str
        Название переменной
    region_name : str, optional
        Название региона (для региональных переменных)
    min_obs : int
        Minimum number of observations
    
    Returns:
    --------
    dict с результатами или None при ошибке
    """
    try:
        # Проверка данных
        if len(series_data) < min_obs:
            return None
        
        # Убедиться, что индекс - DatetimeIndex
        if not isinstance(series_data.index, pd.DatetimeIndex):
            series_data.index = pd.to_datetime(series_data.index)
        
        # Установить частоту на месячную
        ts_regular = series_data.asfreq('MS').dropna()
        
        if len(ts_regular) < min_obs:
            return None
        
        # Информация о данных
        start_date = ts_regular.index[0]
        end_date = ts_regular.index[-1]
        
        # Создаем уникальную временную папку для каждого вызова X-13
        # чтобы избежать конфликтов файлов
        temp_dir = tempfile.mkdtemp(prefix=f"x13_{uuid.uuid4().hex[:8]}_")
        
        # Запуск X-13 с указанием уникальной временной директории
        try:
            res = x13_arima_analysis(
                endog=ts_regular,
                x12path=x13_dir,
                prefer_x13=True,
                outlier=True,
                trading=False,
                print_stdout=False
            )
            
            # Извлечение компонент
            adj = res.seasadj
            trend = res.trend
            seas = ts_regular - adj
            
        except Exception as e:
            # Если X-13 не сработал, попробуем без outlier
            try:
                res = x13_arima_analysis(
                    endog=ts_regular,
                    x12path=x13_dir,
                    prefer_x13=True,
                    outlier=False,  # Без определения выбросов
                    trading=False,
                    print_stdout=False
                )
                
                adj = res.seasadj
                trend = res.trend
                seas = ts_regular - adj
                
            except Exception as e2:
                # Если все еще не работает, пропускаем
                return None
        
        # Расчёт эффективности
        original_std = ts_regular.std()
        adj_std = adj.std()
        reduction = 100 * (1 - adj_std / original_std) if original_std > 0 else 0
        
        # Удаляем временную директорию
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        return {
            'dates': ts_regular.index,
            'adj': adj.values,
            'trend': trend.values,
            'seasonal': seas.values,
            'reduction': reduction
        }
        
    except Exception as e:
        # Подавляем вывод ошибок
        return None


def seasonal_adjust_panel_wrapper(df, col_name):
    # ===== СЕЗОННАЯ КОРРЕКТИРОВКА ДЛЯ ПАНЕЛЬНЫХ ДАННЫХ =====
    
    # Заглушаем предупреждения
    warnings.filterwarnings('ignore')
    
    
    cfg = json.loads(Path("x13.json").read_text(encoding="utf-8"))
    x13_path = cfg["x13_path"]
    
    # Путь к бинарнику X-13
    X13_PATH = x13_path
    
    # Проверка, что бинарник доступен
    if not os.path.exists(X13_PATH):
        raise FileNotFoundError(f"X-13 binary not found at {X13_PATH}")
    
    # Если X13_PATH — полный путь к бинарнику, берем его директорию для x12path
    global x13_dir
    x13_dir = X13_PATH if os.path.isdir(X13_PATH) else os.path.dirname(X13_PATH)
    
    # Создаем папку для логов
    os.makedirs("x13_logs", exist_ok=True)
    
    # ============ ОПРЕДЕЛЕНИЕ ПЕРЕМЕННЫХ ДЛЯ КОРРЕКТИРОВКИ ============
    
    # ФЕДЕРАЛЬНЫЕ переменные (одна серия для всех регионов)
    federal_variables = {
        'Exc_rate': 'Курс рубля', 
        'Inflation_Expectations': 'Инфляционные ожидания',
        'Bonds_Rate_Correct_5Y': 'Ставки на облигации скорректированные 5Y',
        'IBC_constr_fed' : "ИБК строительства",
        'IBC_torg_fed' : "ИБК торговля",
        'IBC_auto_fed' : "ИБК авто",
        'd_IBC_constr_fed' : "ИБК строительства",
        'd_IBC_torg_fed' : "ИБК торговля",
        'd_IBC_auto_fed' : "ИБК авто",
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
        'Def_Zadolg_ConsCred':'Доля просроченной задолженности по потребительским кредитам',
        'IBC_constr':'ИБК строительства',
        'IBC_torg':'ИБК торговля',
        'IBC_auto':'ИБК авто',
        'd_IBC_constr':'ИБК строительства',
        'd_IBC_torg':'ИБК торговля',
        'd_IBC_auto':'ИБК авто',
        
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


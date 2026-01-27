import os
import tempfile
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import friedmanchisquare
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.x13 import x13_arima_analysis


def detect_outlier_regions_iqr(df, key_vars, min_obs=60, max_na_pct=0.3):
    """
    Определение выбросов по IQR (межквартильный размах)
    """
    region_stats = []
    
    for region in df['Region'].unique():
        reg_data = df[df['Region'] == region][key_vars].dropna()
        
        # 1. Минимум наблюдений
        n_obs = len(reg_data)
        if n_obs < min_obs:
            region_stats.append({
                'Region': region,
                'N_obs': n_obs,
                'Reason': f'Мало данных (<{min_obs})',
                'Status': 'EXCLUDE'
            })
            continue
        
        # 2. % пропусков
        total_rows = len(df[df['Region'] == region])
        na_pct = 1 - (n_obs / total_rows)
        if na_pct > max_na_pct:
            region_stats.append({
                'Region': region,
                'N_obs': n_obs,
                'NA_pct': f'{na_pct:.1%}',
                'Reason': f'Много пропусков (>{max_na_pct*100}%)',
                'Status': 'EXCLUDE'
            })
            continue
        
        # 3. IQR по переменным
        iqr_outliers = []
        for col in key_vars:
            if col in reg_data.columns:
                data = reg_data[col].dropna()
                if len(data) > 0:
                    Q1 = data.quantile(0.15)
                    Q3 = data.quantile(0.85)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                    iqr_outliers.append(outliers)
        
        iqr_total = sum(iqr_outliers)
        iqr_pct = iqr_total / n_obs if n_obs > 0 else 0

        
        # РЕШЕНИЕ
        status = 'KEEP'
        reasons = []
        
        if iqr_pct > 0.2:  # >20% выбросов по IQR
            reasons.append(f'Много выбросов по IQR ({iqr_pct:.1%})')
            status = 'EXCLUDE'
        
        region_stats.append({
            'Region': region,
            'N_obs': n_obs,
            'NA_pct': f'{na_pct:.1%}',
            'IQR_outliers': iqr_total,
            'IQR_pct': f'{iqr_pct:.1%}',
            'Reasons': '; '.join(reasons),
            'Status': status
        })
    
    return pd.DataFrame(region_stats)

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

def _kpss_level_stat(y):
    """KPSS статистика для уровня стационарности (без тренда), гомоскедастичная долгосрочная дисперсия."""
    y = np.asarray(y, float)
    T = len(y)
    u = y - y.mean()
    S = np.cumsum(u)
    sigma2 = np.mean(u**2)
    return np.sum(S**2) / (T**2 * sigma2)

def hadri_panel_test(df, id_col='Region', time_col='Date', y_col='y', trend=True):
    """
    Упрощённый тест панельной стационарности Hadri (2000):
    - Строит KPSS-подобные статистики по каждому юниту и усредняет их.
    - Возвращает среднюю статистику и эмпирическое p-значение по бутстрепу.
    H0: все панели стационарны.
    """
    d = df[[id_col, time_col, y_col]].dropna().copy()
    d = d.sort_values([id_col, time_col])

    # Требуется сбалансированная панель для этой реализации
    counts = d.groupby(id_col)[y_col].size().values
    if not np.all(counts == counts[0]):
        raise ValueError("Тест Hadri требует сбалансированную панель")

    kpss_vals = []

    for _, g in d.groupby(id_col):
        y = g[y_col].values.astype(float)

        if trend:
            # Очистка от тренда: константа + линейное время
            t = np.arange(len(y))
            X = sm.add_constant(t)
            res = sm.OLS(y, X).fit()
            u = res.resid
            S = np.cumsum(u)
            sigma2 = np.mean(u**2)
            T = len(y)
            stat = np.sum(S**2) / (T**2 * sigma2)
        else:
            stat = _kpss_level_stat(y)

        kpss_vals.append(stat)

    kpss_vals = np.asarray(kpss_vals, float)
    N = len(kpss_vals)

    # Средняя KPSS-статистика (Hadri LM)
    lm_mean = float(kpss_vals.mean())

    # Эмпирическое p-значение через бутстреп перестановок по юнитам
    B = 1000
    if N > 1:
        rng = np.random.default_rng(12345)
        lm_boot = []
        for _ in range(B):
            idx = rng.integers(0, N, size=N)
            lm_boot.append(kpss_vals[idx].mean())
        lm_boot = np.asarray(lm_boot)
        pvalue = float((lm_boot >= lm_mean).mean())
    else:
        pvalue = np.nan

    return {
        "statistic": lm_mean,
        "pvalue": pvalue,
        "kpss_mean": lm_mean,
        "kpss_std": float(kpss_vals.std(ddof=1)) if N > 1 else np.nan,
        "N": int(N)
    }

def levin_lin_chu_test(df, id_col='Region', time_col='Date', y_col='y',
                       lags=1, trend=True):
    """
    Упрощённая реализация Levin–Lin–Chu:
    - выравнивание по общей временной сетке,
    - отбрасывание несбалансированных юнитов после лагов.
    H0: присутствует единичный корень (панели НЕ стационарны).
    """
    d = df[[id_col, time_col, y_col]].copy()
    d = d.dropna(subset=[id_col, time_col, y_col])

    # Общая временная сетка
    all_times = np.sort(d[time_col].unique())

    dy_list = []
    ylag_list = []
    lagdiff_list = []
    trend_list = []
    fe_ids = []

    unit_idx = 0

    for _, g in d.groupby(id_col):
        # Выравниваем по общей сетке времени
        g = g.set_index(time_col).reindex(all_times).sort_index()
        y = pd.to_numeric(g[y_col], errors='coerce').values

        # Если все пропуски — пропускаем юнит
        if np.isnan(y).all():
            continue

        # Убираем ведущие и хвостовые NaN блоком
        mask_valid = ~np.isnan(y)
        if mask_valid.sum() <= (lags + 2):
            continue
        first = np.argmax(mask_valid)
        last = len(mask_valid) - np.argmax(mask_valid[::-1])
        y = y[first:last]

        # После обрезки снова проверка длины
        if len(y) <= (lags + 2):
            continue

        # Строим Δy и лаги
        dy = np.diff(y)
        ylag = y[:-1]

        # Лаги разностей
        lag_diffs = []
        for k in range(1, lags + 1):
            lag_k = np.r_[ [np.nan]*k, np.diff(y)[:-k] ]
            lag_diffs.append(lag_k[1:])
        if lags > 0:
            lag_diffs = np.column_stack(lag_diffs)
        else:
            lag_diffs = np.empty((len(dy), 0))

        T_eff = len(dy)
        ylag = ylag[1:]
        if len(ylag) != T_eff:
            T_eff = min(T_eff, len(ylag))
        dy = dy[:T_eff]
        ylag = ylag[:T_eff]
        lag_diffs = lag_diffs[:T_eff, :]

        if T_eff <= 0:
            continue

        dy_list.append(dy)
        ylag_list.append(ylag)
        lagdiff_list.append(lag_diffs)

        if trend:
            t = np.arange(1, T_eff + 1)
            trend_list.append(t)

        fe_ids.append(np.repeat(unit_idx, T_eff))
        unit_idx += 1

    # Если после отсева юнитов ничего не осталось
    if len(dy_list) == 0:
        raise ValueError("LLC: нет полезных панелей после балансировки; проверьте пропуски и лаги")

    dy_all = np.concatenate(dy_list)
    ylag_all = np.concatenate(ylag_list)
    if lags > 0:
        lagdiff_all = np.vstack(lagdiff_list)
    else:
        lagdiff_all = np.empty((len(dy_all), 0))

    ids_all = np.concatenate(fe_ids)
    fe_dummies = pd.get_dummies(ids_all, drop_first=True).values

    if trend:
        trend_all = np.concatenate(trend_list)[:, None]
    else:
        trend_all = np.empty((len(dy_all), 0))

    # Финальная проверка на одинаковую длину
    n_obs = len(dy_all)
    ylag_all = ylag_all[:n_obs]
    lagdiff_all = lagdiff_all[:n_obs, :]
    fe_dummies = fe_dummies[:n_obs, :]
    trend_all = trend_all[:n_obs, :]

    X = np.column_stack([ylag_all[:, None], lagdiff_all, fe_dummies, trend_all])
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(dy_all, X)
    res = model.fit()

    rho_idx = 1  # const на позиции 0, rho на 1
    rho_hat = res.params[rho_idx]
    t_stat = res.tvalues[rho_idx]
    pvalue = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "rho_hat": float(rho_hat),
        "t_stat": float(t_stat),
        "pvalue": float(pvalue),
        "N": int(unit_idx)
    }

def run_panel_unit_root_tests(df_reg_analys: pd.DataFrame,
                                     panel_vars: list = None) -> pd.DataFrame:
    """
    Применяет тесты на единичные корни к переменным из списка.

    Parameters
    ----------
    df_reg_analys : pd.DataFrame
        Исходный датасет с колонками 'Region', 'Date' и тестируемыми переменными.

    panel_vars : list, optional
        Список переменных для тестирования.
        Если None, тестируются все переменные с суффиксом '_adj'.
    """

    # === ПОЛУЧЕНИЕ ПЕРЕМЕННЫХ ДЛЯ ТЕСТИРОВАНИЯ ===
    if panel_vars is None:
        panel_vars = [
            col for col in df_reg_analys.columns
            if col.endswith('_adj') and col not in ['Date', 'Region']
        ]
    else:
        missing_vars = [v for v in panel_vars if v not in df_reg_analys.columns]
        if missing_vars:
            print(f" Предупреждение: следующие переменные не найдены в датасете: {missing_vars}")
            panel_vars = [v for v in panel_vars if v in df_reg_analys.columns]

        if not panel_vars:
            raise ValueError("Нет переменных для тестирования")

    print(f"Тестирование {len(panel_vars)} переменных на единичные корни")
    print("=" * 70)
    print(f"Переменные для тестирования:")
    for i, var in enumerate(panel_vars, 1):
        print(f"  {i}. {var}")
    print("=" * 70)

    results_rows = []
    pdf = df_reg_analys.copy()

    for var in panel_vars:
        try:
            x = pdf[var].values
            x_clean = x[~pd.isna(x)]
            if len(x_clean) <= 10:
                print(f"⚠ {var}: недостаточно данных (< 10 наблюдений)")
                continue

            # === Hadri ===
            hadri_stat = np.nan
            hadri_pval = np.nan
            try:
                hadri_res = hadri_panel_test(
                    df_reg_analys[['Region', 'Date', var]].rename(columns={var: 'y'}),
                    id_col='Region',
                    time_col='Date',
                    y_col='y',
                    trend=True
                )
                hadri_stat = float(hadri_res["statistic"])
                hadri_pval = float(hadri_res["pvalue"])
            except Exception as e:
                print(f"Ошибка теста Hadri для {var}: {e}")

            # === Levin–Lin–Chu ===
            llc_stat = np.nan
            llc_pval = np.nan
            try:
                llc_res = levin_lin_chu_test(
                    df_reg_analys[['Region', 'Date', var]].rename(columns={var: 'y'}),
                    id_col='Region',
                    time_col='Date',
                    y_col='y',
                    lags=1,
                    trend=True
                )
                llc_stat = float(llc_res["t_stat"])
                llc_pval = float(llc_res["pvalue"])
            except Exception as e:
                print(f"Ошибка теста Levin–Lin–Chu для {var}: {e}")

            # === PP и DF-GLS по регионам ===
            pp_stats = []
            pp_pvalues = []
            pp_rejections = 0

            dfgls_stats = []
            dfgls_rejections = 0

            regions = pdf['Region'].dropna().unique()

            for region in regions:
                region_series = pdf.loc[pdf['Region'] == region, var]
                region_data = pd.to_numeric(region_series, errors='coerce').dropna().values

                if len(region_data) > 5:
                    # PP (приблизительно через ADF)
                    try:
                        adf_res = sm.tsa.stattools.adfuller(region_data, regression='c', autolag='AIC')
                        pp_stat = adf_res[0]
                        pp_pval = adf_res[1]
                        pp_stats.append(pp_stat)
                        pp_pvalues.append(pp_pval)
                        if pp_pval < 0.05:
                            pp_rejections += 1
                    except Exception:
                        pass

                    # DF-GLS (через GLS-детрендинг + ADF)
                    try:
                        T = len(region_data)
                        y = region_data
                        alpha = 1 - 7 / T
                        y_tilde = y[1:] - alpha * y[:-1]
                        X = np.ones(T)
                        X_tilde = X[1:] - alpha * X[:-1]
                        beta = np.linalg.lstsq(X_tilde[:, None], y_tilde, rcond=None)[0][0]
                        u = y - beta * X
                        du = np.diff(u)
                        u_lag = u[:-1]
                        adf_gls = sm.OLS(du, sm.add_constant(u_lag)).fit()
                        stat = adf_gls.tvalues[1]
                        dfgls_stats.append(stat)
                        if stat < -2.86:
                            dfgls_rejections += 1
                    except Exception:
                        pass

            # Агрегация PP
            pp_mean = np.nan
            pp_median = np.nan
            pp_rejection_rate = np.nan
            if len(pp_stats) > 0:
                pp_mean = float(np.nanmean(pp_stats))
                pp_median = float(np.nanmedian(pp_stats))
                pp_rejection_rate = float(pp_rejections / len(pp_stats) * 100.0)

            # Агрегация DF-GLS
            dfgls_mean = np.nan
            dfgls_median = np.nan
            dfgls_rejection_rate = np.nan
            if len(dfgls_stats) > 0:
                dfgls_mean = float(np.nanmean(dfgls_stats))
                dfgls_median = float(np.nanmedian(dfgls_stats))
                dfgls_rejection_rate = float(dfgls_rejections / len(dfgls_stats) * 100.0)

            results_rows.append({
                "Переменная": var,
                "Hadri_Stat": hadri_stat,
                "Hadri_PValue": hadri_pval,
                "LLC_Stat": llc_stat,
                "LLC_PValue": llc_pval,
                "PP_Mean": pp_mean,
                "PP_Median": pp_median,
                "PP_Rejections_%": pp_rejection_rate,
                "DFGLS_Mean": dfgls_mean,
                "DFGLS_Median": dfgls_median,
                "DFGLS_Rejections_%": dfgls_rejection_rate
            })

        except Exception as e:
            print(f"Общая ошибка для {var}: {e}")

    results = pd.DataFrame(results_rows)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ТЕСТОВ НА ЕДИНИЧНЫЕ КОРНИ")
    print("=" * 70)

    # === ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ ===
    results['Hadri_Stationarity'] = results['Hadri_PValue'] >= 0.05
    results['LLC_Stationarity'] = results['LLC_PValue'] < 0.05
    results['PP_Stationarity'] = results['PP_Rejections_%'] >= 50
    results['DFGLS_Stationarity'] = results['DFGLS_Rejections_%'] >= 50

    # Старый строгий критерий
    strict_stationarity = (
        (results['Hadri_Stationarity'] & results['LLC_Stationarity']) &
        (results['PP_Stationarity'] | results['DFGLS_Stationarity'])
    )

    # Новый критерий: минимум 3 из 4 тестов "за стационарность"
    stationarity_votes = (
        results[['Hadri_Stationarity',
                 'LLC_Stationarity',
                 'PP_Stationarity',
                 'DFGLS_Stationarity']]
        .sum(axis=1)
    )

    results['Overall_Stationarity'] = strict_stationarity | (stationarity_votes >= 3)


    print(results.to_string(index=False))

    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 70)
    print(f"Всего переменных протестировано: {len(results)}")
    print(f"Тест Hadri (H0: стационарность): {results['Hadri_Stationarity'].sum()} стационарны")
    print(f"Тест Levin–Lin–Chu (отвергаем H0: единичный корень): {results['LLC_Stationarity'].sum()} стационарны")
    print(f"Тест PP: {results['PP_Stationarity'].sum()} стационарны (≥50% регионов)")
    print(f"Тест DF-GLS: {results['DFGLS_Stationarity'].sum()} стационарны (≥50% регионов)")
    print(f"\n✓ Общий вывод (Hadri И LLC + минимум 1 из PP/DF-GLS): {results['Overall_Stationarity'].sum()} стационарны")

    # Вывод нестационарных переменных
    non_stationary_vars = results[~results['Overall_Stationarity']]['Переменная'].tolist()
    if non_stationary_vars:
        print(f"\n  Нестационарные переменные: {len(non_stationary_vars)}")
        for var in non_stationary_vars:
            print(f"  - {var}")

    return results

class ModelResultsAggregator:
    '''
    Агрегирует результаты панельных регрессий (POOL, FE, RE)
    из разных спецификаций и подвыборок в единую таблицу.
    '''

    def __init__(self):
        self.results_storage = []
        self.dependent_vars = []
        self.subsamples = []

    def add_model_results(self,
                          model_results,
                          dependent_variable: str,
                          subsample_name: str,
                          model_type: str,
                          specification_name: str,
                          se_type: str | None = None):
        """
        Добавляет результаты одной модели в хранилище.

        specification_name — уникальное имя столбца
        (например 'Модель_1', 'Модель_2', ...).
        se_type — строка с типом стандартных ошибок
        (например 'robust', 'clustered (region)'...)
        """
        coefficients = model_results.params
        pvalues = model_results.pvalues

        # R^2
        try:
            r_squared = model_results.rsquared
        except Exception:
            r_squared = None

        # N
        try:
            nobs = int(model_results.nobs)
        except Exception:
            nobs = None

        # если тип SE не передан, попытаемся вытащить из атрибута модели (если есть)
        if se_type is None:
            se_type = getattr(model_results, 'cov_type', None)

        for var_name in coefficients.index:
            coef = coefficients[var_name]
            pval = pvalues[var_name]
            stars = self._get_significance_stars(pval)

            self.results_storage.append({
                'dependent_var': dependent_variable,
                'subsample': subsample_name,
                'model_type': model_type,
                'specification': specification_name,
                'variable': var_name,
                'coefficient': coef,
                'pvalue': pval,
                'stars': stars,
                'r_squared': r_squared,
                'nobs': nobs,
                'se_type': se_type
            })

        if dependent_variable not in self.dependent_vars:
            self.dependent_vars.append(dependent_variable)
        if subsample_name not in self.subsamples:
            self.subsamples.append(subsample_name)

    def _get_significance_stars(self, pvalue):
        if pd.isna(pvalue):
            return ''
        if pvalue < 0.01:
            return '***'
        elif pvalue < 0.05:
            return '**'
        elif pvalue < 0.10:
            return '*'
        else:
            return ''

    def build_table(self,
                    dependent_var_filter: str = None,
                    specification_filter: list | None = None,
                    include_pvalues: bool = True,
                    decimals: int = 3) -> pd.DataFrame:
        """
        Итоговая таблица:
        1-я строка: 'Зависимая переменная'
        2-я–...: коэффициенты и p-value по регрессорам
        строка под R^2: 'Тип стандартных ошибок'
        следующая строка: 'n-obs'
        предпоследняя строка: 'Метод оценки' (POOL/FE/RE)
        последняя строка: 'Выборка'.
        """
        df = pd.DataFrame(self.results_storage)

        # фильтры
        if dependent_var_filter:
            df = df[df['dependent_var'] == dependent_var_filter]
        if specification_filter is not None:
            df = df[df['specification'].isin(specification_filter)]

        # порядок столбцов по спецификациям
        col_order = list(dict.fromkeys(df['specification']))

        # тело таблицы
        body_dict = {}
        for spec in col_order:
            subset = df[df['specification'] == spec]
            col_data = {}
            for _, row in subset.iterrows():
                var = row['variable']
                coef = row['coefficient']
                stars = row['stars']
                pval = row['pvalue']

                if include_pvalues and pd.notna(pval):
                    cell = f"{coef:.{decimals}f}{stars} ({pval:.{decimals}f})"
                else:
                    cell = f"{coef:.{decimals}f}{stars}"

                col_data[var] = cell
            body_dict[spec] = col_data

        body_df = pd.DataFrame(body_dict).reindex(columns=col_order).fillna('')

        # Зависимая переменная (верхняя строка)
        dep_names = (df.groupby('specification')['dependent_var']
                       .first()
                       .reindex(col_order))
        dep_row = pd.DataFrame([dep_names], index=['Зависимая переменная'])

        # R^2
        r2_vals = (df.groupby('specification')['r_squared']
                     .max()
                     .round(decimals)
                     .reindex(col_order))
        r2_row = pd.DataFrame([r2_vals], index=['R^2'])

        # Тип стандартных ошибок (под R^2)
        se_vals = (df.groupby('specification')['se_type']
                     .first()
                     .reindex(col_order))
        se_row = pd.DataFrame([se_vals], index=['Тип стандартных ошибок'])

        # n-obs (число наблюдений)
        nobs_vals = (df.groupby('specification')['nobs']
                       .max()
                       .reindex(col_order))
        nobs_row = pd.DataFrame([nobs_vals], index=['n-obs'])

        # Метод оценки (предпоследняя строка)
        model_names = (df.groupby('specification')['model_type']
                         .first()
                         .reindex(col_order))
        model_row = pd.DataFrame([model_names], index=['Метод оценки'])

        # Выборка (последняя строка)
        subsample_vals = (df.groupby('specification')['subsample']
                            .first()
                            .reindex(col_order))
        subsample_row = pd.DataFrame([subsample_vals], index=['Выборка'])

        # Итог: Зависимая переменная -> тело -> R^2 -> Тип SE -> n-obs -> Метод -> Выборка
        output_df = pd.concat(
            [dep_row, body_df, r2_row, se_row, nobs_row, model_row, subsample_row],
            axis=0
        )
        return output_df

    def build_summary_table(self,
                            dependent_var_filter: str = None,
                            decimals: int = 3) -> pd.DataFrame:
        """
        Сводная таблица по спецификациям (как дополнение к основной).
        Внизу остаётся строка с N (как раньше).
        """
        df = pd.DataFrame(self.results_storage)

        if dependent_var_filter:
            df = df[df['dependent_var'] == dependent_var_filter]

        df['col_name'] = df['specification']

        df['formatted_coef'] = (df['coefficient'].round(decimals).astype(str) +
                                df['stars'])
        df['formatted_pval'] = '(' + df['pvalue'].round(decimals).astype(str) + ')'

        result = df.groupby(['variable', 'col_name']).apply(
            lambda x: ' '.join([x.iloc[0]['formatted_coef'],
                                x.iloc[0]['formatted_pval']])
        ).unstack(fill_value='')

        # строка с N внизу (можно оставить как N, чтобы не ломать старый код)
        nobs_table = (df.groupby('col_name')['nobs']
                        .max()
                        .rename('N')
                        .to_frame()
                        .T)
        nobs_table = nobs_table.astype('Int64').astype(str)

        result = pd.concat([result, nobs_table])
        return result


def run_panel_regressions(df_input, df_reg, dependent_var, exog_vars_initial, cov_type, cluster_entity=None, df_name=None, intercept=True):
    from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS

    df_label_map = {
        'df_reg': 'ОБЩАЯ ВЫБОРКА',
        'df_reg_clus_one': 'КЛАСТЕР 1',
        'df_reg_clus_two': 'КЛАСТЕР 2',
        'df_reg_clus_three': 'КЛАСТЕР 3'
    }
    if df_name is None:
        if df_input is df_reg:
            df_name = 'df_reg'
    df_label = df_label_map.get(df_name, df_name if df_name else 'ДАННЫЕ')
    roisfix_suffix = ''
    if any('ROISFIX' in var for var in exog_vars_initial):
        if df_name == 'df_reg':
            roisfix_suffix = ' - ROISFIX'
        else:
            df_label = f"{df_label} ROISFIX"

    print("="*70)
    print(f"ПАНЕЛЬНАЯ РЕГРЕССИЯ: POOL + FE + RE ({df_label}){roisfix_suffix}")
    print(f"Зависимая переменная: {dependent_var}")
    print("="*70)

    # Создаем копию df_reg для работы
    df_clean = df_input.copy()  # логика без изменений

    # Приводим df_reg к формату с Region и Date как столбцами
    if isinstance(df_reg.index, pd.MultiIndex):
        df_reg_temp = df_reg.reset_index()
    else:
        df_reg_temp = df_reg.copy()

    # Проверяем наличие зависимой переменной
    if dependent_var in df_reg_temp.columns:
        # Присоединяем зависимую переменную по Region и Date
        df_clean = df_clean.merge(
            df_reg_temp[['Region', 'Date', dependent_var]],
            on=['Region', 'Date'],
            how='left',
            suffixes=('', '_from_reg')
        )
    else:
        print(f"ERROR - {dependent_var} отсутствует в df_reg_temp")

    # Проверяем независимые переменные
    exog_vars_for_regression = []
    for var in exog_vars_initial:
        if var in df_clean.columns:
            exog_vars_for_regression.append(var)
        else:
            print(f"  WARNING: {var} исключена")

    # Удаляем NaN
    cols_to_check = [dependent_var] + exog_vars_for_regression
    df_clean = df_clean.dropna(subset=cols_to_check)

    # ===== Установка панельного индекса =====
    df_clean = df_clean.set_index(['Region', 'Date']).sort_index()

    # ===== Подготовка Y и X =====
    y = df_clean[[dependent_var]]
    X = df_clean[exog_vars_for_regression]
    X_pooled = X
    X_re = X
    if intercept:
        X_pooled = sm.add_constant(X_pooled, has_constant="add")
        X_re = sm.add_constant(X_re, has_constant="add")

    # ===== POOLED OLS =====
    print("\n" + "="*70)
    print("МОДЕЛЬ 1: POOLED OLS")
    print("="*70)

    try:
        pooled_mod = PooledOLS(y, X_pooled)
        pooled_res = pooled_mod.fit(cov_type=cov_type, cluster_entity=cluster_entity)
        print(pooled_res.summary)
        pooled_success = True
    except Exception as e:
        print(f"ERROR: {e}")
        pooled_res = None
        pooled_success = False

    # ===== FIXED EFFECTS =====
    print("\n" + "="*70)
    print("МОДЕЛЬ 2: FIXED EFFECTS (WITHIN)")
    print("="*70)

    try:
        fe_mod = PanelOLS(y, X, entity_effects=True, drop_absorbed=True)
        if cluster_entity is None:
            fe_res = fe_mod.fit(cov_type=cov_type,)
        else:
            fe_res = fe_mod.fit(cov_type=cov_type, cluster_entity=cluster_entity)
        print(fe_res.summary)
        fe_success = True
    except Exception as e:
        print(f"ERROR: {e}")
        fe_res = None
        fe_success = False

    # ===== RANDOM EFFECTS =====
    print("\n" + "="*70)
    print("МОДЕЛЬ 3: RANDOM EFFECTS")
    print("="*70)

    try:
        re_mod = RandomEffects(y, X_re)
        re_res = re_mod.fit(cov_type=cov_type, cluster_entity=cluster_entity)
        print(re_res.summary)
        re_success = True
    except Exception as e:
        print(f"ERROR: {e}")
        re_res = None
        re_success = False

    return y, X, pooled_res, fe_res, re_res, pooled_success, fe_success, re_success


def run_panel_regressions_trend(df_input, df_reg, dependent_var, exog_vars_initial, cov_type, cluster_entity=None, df_name=None, trend=True, intercept=True):
    from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS

    df_label_map = {
        'df_reg': 'ОБЩАЯ ВЫБОРКА',
        'df_reg_clus_one': 'КЛАСТЕР 1',
        'df_reg_clus_two': 'КЛАСТЕР 2',
        'df_reg_clus_three': 'КЛАСТЕР 3'
    }
    if df_name is None:
        if df_input is df_reg:
            df_name = 'df_reg'
    df_label = df_label_map.get(df_name, df_name if df_name else 'ДАННЫЕ')
    roisfix_suffix = ''
    if any('ROISFIX' in var for var in exog_vars_initial):
        if df_name == 'df_reg':
            roisfix_suffix = ' - ROISFIX'
        else:
            df_label = f"{df_label} ROISFIX"

    print("="*70)
    print(f"ПАНЕЛЬНАЯ РЕГРЕССИЯ: POOL + FE + RE ({df_label}){roisfix_suffix}")
    print(f"Зависимая переменная: {dependent_var}")
    print("="*70)

    # Создаем копию df_reg для работы
    df_clean = df_input.copy()  # логика без изменений

    # Приводим df_reg к формату с Region и Date как столбцами
    if isinstance(df_reg.index, pd.MultiIndex):
        df_reg_temp = df_reg.reset_index()
    else:
        df_reg_temp = df_reg.copy()

    # Проверяем наличие зависимой переменной
    if dependent_var in df_reg_temp.columns:
        # Присоединяем зависимую переменную по Region и Date
        df_clean = df_clean.merge(
            df_reg_temp[['Region', 'Date', dependent_var]],
            on=['Region', 'Date'],
            how='left',
            suffixes=('', '_from_reg')
        )
    else:
        print(f"ERROR - {dependent_var} отсутствует в df_reg_temp")

    # Проверяем независимые переменные
    exog_vars_for_regression = []
    for var in exog_vars_initial:
        if var in df_clean.columns:
            exog_vars_for_regression.append(var)
        else:
            print(f"  WARNING: {var} исключена")

    # Удаляем NaN
    cols_to_check = [dependent_var] + exog_vars_for_regression
    df_clean = df_clean.dropna(subset=cols_to_check)

    # Общий тренд по времени (одинаковый для всех регионов)
    if trend:
        unique_dates = pd.Series(df_clean['Date'].unique()).sort_values()
        date_to_trend = {date: i + 1 for i, date in enumerate(unique_dates)}
        df_clean['trend'] = df_clean['Date'].map(date_to_trend)
        if 'trend' not in exog_vars_for_regression:
            exog_vars_for_regression.append('trend')

    # ===== Установка панельного индекса =====
    df_clean = df_clean.set_index(['Region', 'Date']).sort_index()

    # ===== Подготовка Y и X =====
    y = df_clean[[dependent_var]]
    X = df_clean[exog_vars_for_regression]
    if trend:
        exog_vars_re = [var for var in exog_vars_for_regression if var != 'trend']
        X_re = df_clean[exog_vars_re]
    else:
        X_re = X
    X_pooled = X
    X_re_fit = X_re
    if intercept:
        X_pooled = sm.add_constant(X_pooled, has_constant="add")
        X_re_fit = sm.add_constant(X_re_fit, has_constant="add")

    # ===== POOLED OLS =====
    print("\n" + "="*70)
    print("МОДЕЛЬ 1: POOLED OLS")
    print("="*70)

    try:
        pooled_mod = PooledOLS(y, X_pooled)
        pooled_res = pooled_mod.fit(cov_type=cov_type)
        print(pooled_res.summary)
        pooled_success = True
    except Exception as e:
        print(f"ERROR: {e}")
        pooled_res = None
        pooled_success = False

    # ===== FIXED EFFECTS =====
    print("\n" + "="*70)
    print("МОДЕЛЬ 2: FIXED EFFECTS (WITHIN)")
    print("="*70)

    try:
        fe_mod = PanelOLS(y, X, entity_effects=True)
        if cluster_entity is None:
            fe_res = fe_mod.fit(cov_type=cov_type)
        else:
            fe_res = fe_mod.fit(cov_type=cov_type, cluster_entity=cluster_entity)
        print(fe_res.summary)
        fe_success = True
    except Exception as e:
        print(f"ERROR: {e}")
        fe_res = None
        fe_success = False

    # ===== RANDOM EFFECTS =====
    print("\n" + "="*70)
    print("МОДЕЛЬ 3: RANDOM EFFECTS")
    print("="*70)

    try:
        re_mod = RandomEffects(y, X_re_fit)
        re_res = re_mod.fit(cov_type=cov_type)
        print(re_res.summary)
        re_success = True
    except Exception as e:
        print(f"ERROR: {e}")
        re_res = None
        re_success = False

    return y, X, pooled_res, fe_res, re_res, pooled_success, fe_success, re_success


def run_spec_tests(y, X, pooled_res, fe_res, re_res, pooled_success, fe_success, re_success):
    from scipy.stats import chi2, f

    hausman_stat = None
    hausman_pval = None
    bp_lm_stat = None
    bp_lm_pval = None
    f_stat = None
    f_pval = None

    df_clean = X
    exog_vars_for_regression = list(X.columns)

    def _align_exog_to_params(X_df, params):
        if not isinstance(X_df, pd.DataFrame):
            return X_df
        if params is None:
            return X_df
        names = getattr(params, "index", None)
        X_use = X_df
        if names is not None:
            if "const" in names and "const" not in X_use.columns:
                X_use = sm.add_constant(X_use, has_constant="add")
            missing = [name for name in names if name not in X_use.columns]
            if not missing:
                return X_use.loc[:, names]
        if X_use.shape[1] != len(params):
            X_try = sm.add_constant(X_use, has_constant="add")
            if X_try.shape[1] == len(params):
                return X_try
        return X_use

    print("\n" + "="*70)
    print("ТЕСТЫ СПЕЦИФИКАЦИИ")
    print("="*70)

    # ТЕСТ ХАУСМАНА
    print("\n ТЕСТ ХАУСМАНА (FE vs RE)")
    print("-"*70)

    if fe_success and re_success:
        try:
            coef_diff = (fe_res.params - re_res.params).dropna()

            var_fe = fe_res.cov.loc[coef_diff.index, coef_diff.index]
            var_re = re_res.cov.loc[coef_diff.index, coef_diff.index]
            var_diff = var_fe - var_re

            try:
                inv_var_diff = np.linalg.inv(var_diff.values)
            except np.linalg.LinAlgError:
                print("Матрица сингулярна, используется pseudo-inverse")
                inv_var_diff = np.linalg.pinv(var_diff.values)

            H = coef_diff.values @ inv_var_diff @ coef_diff.values
            p_val = 1 - chi2.cdf(H, df=len(coef_diff))

            print(f"H-статистика: {H:.4f}")
            print(f"p-значение: {p_val:.6f}")
            print(f"df: {len(coef_diff)}")

            hausman_stat = float(H)
            hausman_pval = float(p_val)

            if p_val < 0.05:
                print("\nВывод: p < 0.05 - Используйте FIXED EFFECTS")
            else:
                print("\nВывод: p >= 0.05 - Используйте RANDOM EFFECTS")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print("Невозможно провести (требуются обе модели)")

    # ТЕСТ БРЕУША-ПАГАНА
    print("\n ТЕСТ БРЕУША-ПАГАНА (RE vs Pooled)")
    print("-"*70)

    if re_success and pooled_success:
        try:
            X_pooled = _align_exog_to_params(X, pooled_res.params)
            u_pooled = (y.values - X_pooled.values @ pooled_res.params.values.reshape(-1, 1)).flatten()
            sigma_u_sq = (u_pooled**2).sum() / len(u_pooled)

            regions = df_clean.index.get_level_values('Region').unique()
            N = len(regions)
            T = len(df_clean) // N

            sum_mean_u_sq = 0
            for region in regions:
                u_region = u_pooled[df_clean.index.get_level_values('Region') == region]
                mean_u = u_region.mean()
                sum_mean_u_sq += mean_u**2

            LM = (N * T**2) / (2 * (T - 1)) * (sum_mean_u_sq / (sigma_u_sq * N) - 1)**2
            p_val_bp = 1 - chi2.cdf(LM, df=1)

            print(f"LM статистика: {LM:.4f}")
            print(f"p-значение: {p_val_bp:.6f}")
            print(f"N (регионов): {N}, T (периодов): {T}")

            bp_lm_stat = float(LM)
            bp_lm_pval = float(p_val_bp)

            if p_val_bp < 0.05:
                print("\nВывод: p < 0.05 - Есть региональные эффекты (используй RE или FE)")
            else:
                print("\nВывод: p >= 0.05 - Нет эффектов (адекватна Pooled)")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print("Невозможно провести")

    # F-ТЕСТ
    print("\n F-ТЕСТ (FE vs Pooled)")
    print("-"*70)

    if fe_success and pooled_success:
        try:
            N = df_clean.index.get_level_values('Region').nunique()
            T = df_clean.index.get_level_values('Date').nunique()
            k = len(exog_vars_for_regression)

            X_pooled = _align_exog_to_params(X, pooled_res.params)
            X_fe = _align_exog_to_params(X, fe_res.params)
            u_pooled = (y.values - X_pooled.values @ pooled_res.params.values.reshape(-1, 1)).flatten()
            u_fe = (y.values - X_fe.values @ fe_res.params.values.reshape(-1, 1)).flatten()

            SSR_pooled = (u_pooled**2).sum()
            SSR_fe = (u_fe**2).sum()

            F_stat = ((SSR_pooled - SSR_fe) / (N - 1)) / (SSR_fe / (N*T - N - k))
            p_val_f = 1 - f.cdf(F_stat, N-1, N*T - N - k)

            print(f"F-статистика: {F_stat:.4f}")
            print(f"p-значение: {p_val_f:.6f}")
            print(f"df: ({N-1}, {N*T - N - k})")

            f_stat = float(F_stat)
            f_pval = float(p_val_f)

            if p_val_f < 0.05:
                print("\nВывод: p < 0.05 - FE значимо лучше чем Pooled")
            else:
                print("\nВывод: p >= 0.05 - Pooled адекватна")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print("Невозможно провести")

    return hausman_stat, hausman_pval, bp_lm_stat, bp_lm_pval, f_stat, f_pval


def run_panel_model_diagnostics(
    y,
    X,
    pooled_res=None,
    fe_res=None,
    re_res=None,
    pooled_success=True,
    fe_success=True,
    re_success=True,
    model_label=None,
    pval_threshold=0.05,
    shock_vars=None,
    max_shapiro_n=5000,
    hetero_add_constant=True,
):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    def _as_series(obj):
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            if obj.shape[1] == 1:
                return obj.iloc[:, 0]
            return obj.squeeze()
        if isinstance(obj, pd.Series):
            return obj
        return pd.Series(np.asarray(obj).flatten())

    def _get_residuals(res):
        if res is None:
            return None
        for attr in ("resids", "resid"):
            if hasattr(res, attr):
                out = _as_series(getattr(res, attr))
                if out is not None:
                    return out
        return None

    def _align(y_in, X_in, resids_in=None):
        y_s = _as_series(y_in)
        X_df = X_in.copy() if isinstance(X_in, pd.DataFrame) else None
        resids_s = _as_series(resids_in)

        idx = None
        for obj in (y_s, X_df, resids_s):
            if obj is not None and hasattr(obj, "index"):
                idx = obj.index if idx is None else idx.intersection(obj.index)
        if idx is not None:
            if y_s is not None:
                y_s = y_s.loc[idx]
            if X_df is not None:
                X_df = X_df.loc[idx]
            if resids_s is not None:
                resids_s = resids_s.loc[idx]
        return y_s, X_df, resids_s

    def _format_pvalue(pval):
        if pval is None or (isinstance(pval, float) and np.isnan(pval)):
            return "nan"
        return f"{pval:.6f}"

    def _get_model_exog(res, resids_index=None):
        if res is None:
            print("Heteroskedasticity tests skipped: model exog unavailable.")
            return None
        model = getattr(res, "model", None)
        exog = getattr(model, "exog", None) if model is not None else None
        if exog is None:
            print("Heteroskedasticity tests skipped: model exog unavailable.")
            return None
        exog_names = getattr(model, "exog_names", None)
        exog_df = None
        if isinstance(exog, pd.DataFrame):
            exog_df = exog.copy()
        elif hasattr(exog, "dataframe"):
            try:
                exog_df = exog.dataframe.copy()
            except Exception:
                exog_df = None
        elif hasattr(exog, "to_pandas"):
            try:
                exog_df = exog.to_pandas().copy()
            except Exception:
                exog_df = None
        else:
            try:
                exog_df = pd.DataFrame(np.asarray(exog))
            except Exception:
                exog_df = None
        if exog_df is None:
            print("Heteroskedasticity tests skipped: model exog unavailable.")
            return None
        if exog_names is not None and len(exog_names) == exog_df.shape[1]:
            exog_df.columns = list(exog_names)
        if resids_index is not None:
            try:
                exog_df = exog_df.reindex(resids_index)
            except Exception:
                pass
        return exog_df

    def _is_constant(col):
        col = pd.Series(col).dropna()
        if col.empty:
            return False
        try:
            vals = col.astype(float)
            return np.nanstd(vals.values) < 1e-12
        except Exception:
            return col.nunique(dropna=True) <= 1

    def _has_constant(exog):
        for col in exog.columns:
            if _is_constant(exog[col]):
                return True
        return False

    def _modified_wald_test(resids):
        if not isinstance(resids.index, pd.MultiIndex) or len(resids.index.names) < 2:
            print("Modified Wald test skipped: needs panel index.")
            return
        idx_names = list(resids.index.names)
        if "Region" in idx_names and "Date" in idx_names:
            entity_level = idx_names.index("Region")
        else:
            entity_level = 0
        counts = resids.groupby(level=entity_level).size()
        if counts.empty or counts.nunique() != 1:
            print("Modified Wald test skipped: unbalanced panel.")
            return
        resid_sq = resids**2
        s_i2 = resid_sq.groupby(level=entity_level).mean()
        s_bar2 = resid_sq.mean()
        if s_bar2 == 0 or np.isnan(s_bar2):
            print("Modified Wald test skipped: zero variance.")
            return
        T = int(counts.iloc[0])
        N = int(len(counts))
        W = (T / (2 * s_bar2**2)) * ((s_i2 - s_bar2) ** 2).sum()
        p_val = 1 - stats.chi2.cdf(W, df=N)
        print(f"Modified Wald: stat={W:.4f}, p={_format_pvalue(p_val)}")

    def _hetero_feasible(exog):
        const_cols = []
        nonconst_cols = []
        for col in exog.columns:
            if _is_constant(exog[col]):
                const_cols.append(col)
            else:
                nonconst_cols.append(col)
        if not const_cols or not nonconst_cols:
            return False, "insufficient columns"
        try:
            rank = np.linalg.matrix_rank(exog.values.astype(float))
        except Exception:
            return False, "rank-deficient exog"
        if rank < exog.shape[1]:
            return False, "rank-deficient exog"
        return True, None

    def _normality_tests(resids):
        if len(resids) < 3:
            return {
                "jb_stat": np.nan,
                "jb_p": np.nan,
                "jb_skew": np.nan,
                "jb_kurt": np.nan,
                "shapiro_stat": np.nan,
                "shapiro_p": np.nan,
                "shapiro_n": int(len(resids)),
            }
        jb_stat, jb_p, skew, kurt = jarque_bera(resids)
        if len(resids) > max_shapiro_n:
            shapiro_sample = resids.sample(max_shapiro_n, random_state=42)
            shapiro_n = max_shapiro_n
        else:
            shapiro_sample = resids
            shapiro_n = len(resids)
        shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample)
        return {
            "jb_stat": float(jb_stat),
            "jb_p": float(jb_p),
            "jb_skew": float(skew),
            "jb_kurt": float(kurt),
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "shapiro_n": int(shapiro_n),
        }

    def _wooldridge_test(resids):
        if not isinstance(resids.index, pd.MultiIndex) or len(resids.index.names) < 2:
            return None
        idx_names = list(resids.index.names)
        if "Region" in idx_names and "Date" in idx_names:
            entity = "Region"
            time = "Date"
        else:
            entity = idx_names[0]
            time = idx_names[1]
        df = resids.rename("resid").reset_index()
        df = df.sort_values([entity, time])
        df["resid_lag"] = df.groupby(entity)["resid"].shift(1)
        df["diff"] = df["resid"] - df["resid_lag"]
        df = df.dropna(subset=["resid_lag", "diff"])
        if df.empty or df["resid_lag"].var() == 0:
            return None
        ols_res = sm.OLS(df["diff"].values, df["resid_lag"].values).fit()
        beta = float(ols_res.params[0])
        se = float(ols_res.bse[0])
        if se == 0 or np.isnan(se):
            return None
        t_stat = (beta + 0.5) / se
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=ols_res.df_resid))
        return {
            "beta": beta,
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "nobs": int(ols_res.nobs),
        }

    def _pesaran_cd(resids):
        if not isinstance(resids.index, pd.MultiIndex) or len(resids.index.names) < 2:
            return None
        idx_names = list(resids.index.names)
        if "Region" in idx_names:
            entity_level = idx_names.index("Region")
        else:
            entity_level = 0
        wide = resids.unstack(level=entity_level)
        if wide.shape[1] < 2:
            return None
        corr = wide.corr()
        upper_mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        upper = corr.where(upper_mask)
        rho_sum = upper.stack().sum()
        n_pairs = int(upper.count().sum())
        if n_pairs == 0:
            return None
        valid = ~wide.isna()
        overlap = valid.T @ valid
        upper_overlap = overlap.where(upper_mask)
        t_bar = float(upper_overlap.stack().mean())
        n_entities = wide.shape[1]
        cd_stat = np.sqrt(2 * t_bar / (n_entities * (n_entities - 1))) * rho_sum
        p_val = 2 * (1 - stats.norm.cdf(abs(cd_stat)))
        return {
            "cd_stat": float(cd_stat),
            "p_val": float(p_val),
            "n_entities": int(n_entities),
            "t_bar": float(t_bar),
        }

    def _dwh_tests(y_s, X_df):
        y_s = _as_series(y_s)
        if y_s is None or X_df is None or X_df.empty:
            return None
        results = []
        for var in X_df.columns:
            X_exog = X_df.drop(columns=[var])
            if X_exog.shape[1] == 0:
                continue
            z = sm.add_constant(X_exog, has_constant="add")
            try:
                stage1 = sm.OLS(X_df[var].astype(float), z).fit()
                v_hat = stage1.resid
            except Exception:
                continue
            X_aug = pd.concat([X_df, v_hat.rename(f"{var}_resid")], axis=1)
            X_aug = sm.add_constant(X_aug, has_constant="add")
            try:
                stage2 = sm.OLS(y_s.astype(float), X_aug).fit()
                coef = float(stage2.params[f"{var}_resid"])
                se = float(stage2.bse[f"{var}_resid"])
                if se == 0 or np.isnan(se):
                    continue
                t_stat = coef / se
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=stage2.df_resid))
                results.append({
                    "variable": var,
                    "t_stat": float(t_stat),
                    "p_val": float(p_val),
                })
            except Exception:
                continue
        if not results:
            return None
        return pd.DataFrame(results).sort_values("p_val")

    def _shock_wald(res, shock_vars_in):
        if res is None:
            return None
        shock_vars_in = shock_vars_in or []
        params = getattr(res, "params", None)
        cov = getattr(res, "cov", None)
        if params is None or cov is None:
            return None
        present = [v for v in shock_vars_in if v in params.index]
        if not present:
            return None
        b = params.loc[present].values
        v = cov.loc[present, present].values
        try:
            inv_v = np.linalg.inv(v)
        except np.linalg.LinAlgError:
            inv_v = np.linalg.pinv(v)
        stat = float(b.T @ inv_v @ b)
        p_val = float(1 - stats.chi2.cdf(stat, df=len(present)))
        return {"stat": stat, "p_val": p_val, "vars": present}

    shock_vars = shock_vars or ["Covid_dum", "Sank_dum"]
    model_name = model_label
    if model_name is None and isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        model_name = y.columns[0]

    print("\n" + "=" * 80)
    print("MODEL DIAGNOSTICS")
    if model_name:
        print(f"Target: {model_name}")
    print("=" * 80)
    print(f"p-value threshold: {pval_threshold}")

    y_aligned, X_aligned, _ = _align(y, X, None)

    dwh_df = _dwh_tests(y_aligned, X_aligned)
    if dwh_df is None:
        print("\nEndogeneity (Durbin-Wu-Hausman, control-function): skipped")
    else:
        print("\nEndogeneity (Durbin-Wu-Hausman, control-function)")
        print("H0: regressor is exogenous (p < threshold suggests endogeneity).")
        print(dwh_df.to_string(index=False, formatters={"p_val": _format_pvalue}))

    def _run_for_model(label, res, success, is_fe=False):
        print("\n" + "-" * 70)
        print(f"MODEL: {label}")
        print("-" * 70)
        if not success or res is None:
            print("Model failed or missing; diagnostics skipped.")
            return
        resids = _get_residuals(res)
        if resids is None:
            print("Residuals not available; diagnostics skipped.")
            return
        y_m, X_m, resids = _align(y, X, resids)
        if X_m is None or resids is None:
            print("Data alignment failed; diagnostics skipped.")
            return
        resids = resids.dropna()
        if resids.empty:
            print("No residuals after alignment; diagnostics skipped.")
            return

        print("\nHeteroskedasticity tests")
        if is_fe:
            _modified_wald_test(resids)
        else:
            exog = _get_model_exog(res, resids.index)
            if exog is not None:
                try:
                    exog = exog.loc[resids.index]
                except Exception:
                    print("Heteroskedasticity tests skipped: exog misaligned with residuals.")
                else:
                    exog = exog.dropna()
                    if exog.empty:
                        print("Heteroskedasticity tests skipped: exog empty after alignment.")
                    else:
                        resids_het = resids.loc[exog.index]
                        if hetero_add_constant and not _has_constant(exog):
                            exog = sm.add_constant(exog, has_constant="add")
                        feasible, reason = _hetero_feasible(exog)
                        if not feasible:
                            print(f"Heteroskedasticity tests skipped: {reason}.")
                        else:
                            try:
                                bp_stat, bp_p, _, _ = het_breuschpagan(resids_het, exog)
                                print(f"Breusch-Pagan: stat={bp_stat:.4f}, p={_format_pvalue(bp_p)}")
                            except Exception as e:
                                print(f"Breusch-Pagan error: {e}")
                            try:
                                w_stat, w_p, _, _ = het_white(resids_het, exog)
                                print(f"White:         stat={w_stat:.4f}, p={_format_pvalue(w_p)}")
                            except Exception as e:
                                msg = str(e).lower()
                                if "rank" in msg or "singular" in msg:
                                    print("White test skipped: rank-deficient design.")
                                else:
                                    print(f"White test error: {e}")

        try:
            normal = _normality_tests(resids)
            print("\nNormality tests")
            print(f"Jarque-Bera: stat={normal['jb_stat']:.4f}, p={_format_pvalue(normal['jb_p'])}, "
                  f"skew={normal['jb_skew']:.4f}, kurt={normal['jb_kurt']:.4f}")
            print(f"Shapiro-Wilk (n={normal['shapiro_n']}): stat={normal['shapiro_stat']:.4f}, "
                  f"p={_format_pvalue(normal['shapiro_p'])}")
        except Exception as e:
            print(f"\nNormality tests error: {e}")

        try:
            dw = durbin_watson(resids.values)
            print("\nAutocorrelation tests")
            print(f"Durbin-Watson: {dw:.4f}")
            wool = _wooldridge_test(resids)
            if wool is None:
                print("Wooldridge test: skipped (needs panel index).")
            else:
                print(f"Wooldridge AR(1): beta={wool['beta']:.4f}, "
                      f"t={wool['t_stat']:.4f}, p={_format_pvalue(wool['p_val'])}, n={wool['nobs']}")
        except Exception as e:
            print(f"\nAutocorrelation tests error: {e}")

        try:
            cd = _pesaran_cd(resids)
            if cd is None:
                print("\nCross-sectional dependence: skipped (needs panel data).")
            else:
                print("\nCross-sectional dependence (Pesaran CD)")
                print(f"CD stat={cd['cd_stat']:.4f}, p={_format_pvalue(cd['p_val'])}, "
                      f"N={cd['n_entities']}, T_avg={cd['t_bar']:.2f}")
        except Exception as e:
            print(f"\nCross-sectional dependence error: {e}")

        try:
            shock = _shock_wald(res, shock_vars)
            if shock is not None:
                print("\nShock dummy joint test")
                vars_str = ", ".join(shock["vars"])
                print(f"Wald({vars_str}) stat={shock['stat']:.4f}, p={_format_pvalue(shock['p_val'])}")
        except Exception as e:
            print(f"\nShock dummy test error: {e}")

    _run_for_model("Pooled OLS", pooled_res, pooled_success)
    _run_for_model("Fixed Effects", fe_res, fe_success, is_fe=True)
    _run_for_model("Random Effects", re_res, re_success)

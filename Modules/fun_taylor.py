import numpy as np

import arch.unitroot as au
from scipy.stats import friedmanchisquare
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss


def calculate_expected_inflation_12m(df, date_col='Date', 
                                     inflation_col='Inflation_Expectations'):
    """
    Расчет ожидаемой инфляции на 12 месяцев вперед
    
    Если данные на следующий год отсутствуют (последние 12 месяцев),
    используются значения текущего года в качестве прокси
    """
    df_calc = df.copy()

    df_calc['Month'] = df_calc[date_col].dt.month
    df_calc['Year'] = df_calc[date_col].dt.year

    # i_1 = количество месяцев до конца текущего года
    df_calc['i_1'] = 12 - df_calc['Month']
    
    # i_2 = количество месяцев до конца следующего года
    df_calc['i_2'] = 12 + df_calc['i_1']
    
    print(df_calc[['Date', 'Month', 'i_1', 'i_2']].head(10).to_string())

    # Ожидаемая инфляция на конец текущего года
    df_calc['pi_current_year'] = df_calc[inflation_col]
    
    # Ожидаемая инфляция на конец следующего года
    df_calc['pi_next_year'] = df_calc[inflation_col].shift(-12)
    
    # Если нет данных на следующий год, используем значение текущего года в качестве прокси
    df_calc['pi_next_year'] = df_calc['pi_next_year'].fillna(df_calc['pi_current_year'])
    
    print(df_calc[[date_col, 'pi_current_year', 'pi_next_year']].head(15).to_string())

    # Расчет ожидаемой инфляции на 12 месяцев
    df_calc['Expected_Inflation_12m'] = (
        (df_calc['i_1'] * df_calc['pi_current_year'] +
         (12 - df_calc['i_1']) * df_calc['pi_next_year']) / 12
    )
    
    return df_calc


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


def get_series_for_test(df, col_name):
    """
    Проверяет наличие скорректированного ряда (_adj).
    Если есть → возвращает его, иначе возвращает исходный.
    
    Returns:
    --------
    tuple: (series, source_type)
        series : pd.Series - временной ряд
        source_type : str - "Сезонно-скорректированный" или "Исходный"
    """
    
    adj_col = f'{col_name}_adj'
    
    # Проверяем наличие adj столбца и его наполненность
    if adj_col in df.columns:
        adj_series = df[adj_col].dropna()
        if len(adj_series) > 0:
            return adj_series, "Сезонно-скорректированный"
    
    # Если adj не существует или пуст - берем исходный
    original_series = df[col_name].dropna()
    return original_series, "Исходный"


def optimized_ljung_box_test(series, name, source_type, max_lags=None):
    """
    Тест Льюинга-Бокса с автоматическим подбором лагов
    
    Parameters:
    -----------
    series : pd.Series или array-like
        Временной ряд для тестирования
    name : str
        Название переменной
    source_type : str
        "Сезонно-скорректированный" или "Исходный"
    max_lags : int, optional
        Максимальное количество лагов
    
    Returns:
    --------
    dict : результаты теста
    """
    
    if len(series) < 30:
        lags = min(10, len(series) - 1)
    else:
        # Автоматический подбор лагов: sqrt(n) + 10 для месячных данных
        lags = min(int(np.sqrt(len(series))) + 10, len(series) - 1)
    
    if max_lags:
        lags = min(lags, max_lags)
    
    # Минимальное количество лагов для теста
    lags = max(5, lags)
    
    try:
        # Основной тест на выбранных лагах
        result = acorr_ljungbox(series, lags=[lags], return_df=True)
        lb_stat = result['lb_stat'].iloc[0]
        lb_pvalue = result['lb_pvalue'].iloc[0]
        
        # Дополнительно проверяем несколько ключевых лагов
        key_lags = [1, 3, 6, 12]
        key_results = []
        
        for lag in key_lags:
            if lag <= lags:
                res = acorr_ljungbox(series, lags=[lag], return_df=True)
                key_results.append({
                    'lag': lag,
                    'pvalue': res['lb_pvalue'].iloc[0]
                })
        
        return {
            'name': name,
            'source_type': source_type,
            'n_obs': len(series),
            'test_lags': lags,
            'lb_statistic': lb_stat,
            'p_value': lb_pvalue,
            'is_white_noise': lb_pvalue > 0.05,
            'key_lags_results': key_results,
            'series': series  # для визуализации
        }
    
    except Exception as e:
        print(f"Ошибка в тесте для {name} ({source_type}): {e}")
        return None


def stationarity_tests(series, name):
    series_clean = series.dropna()
    #ADF
    try:
        adf_result = adfuller(series_clean, autolag = 'BIC')
        adf_stat = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_critical_one = adf_result[4]['1%']
        adf_critical_five = adf_result[4]['5%']
        adf_critical_ten = adf_result[4]['10%']
        adf_stationary = adf_pvalue < 0.05
    except Exception as e:
        adf_stat, adf_pvalue, adf_critical_one, adf_critical_five, adf_critical_ten, adf_stationary = (
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        )

    #KPSS
    try:
        kpss_result = kpss(series_clean, regression='c', nlags='auto')
        kpss_stat = kpss_result[0]
        kpss_pvalue = kpss_result[1]
        kpss_critical_one = kpss_result[3]['1%']
        kpss_critical_five = kpss_result[3]['5%']
        kpss_critical_ten = kpss_result[3]['10%']
        kpss_stationary = kpss_pvalue > 0.05  # H0: stationary
    except:
        kpss_stat, kpss_pvalue, kpss_critical_one, kpss_critical_five, kpss_critical_ten, kpss_stationary = (
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        )
  #DF-GLS
    try:
        DF_GLS_result = au.DFGLS(series_clean, trend='c', max_lags = 2, method = 'bic')
        DF_GLS_stat = DF_GLS_result.stat
        DF_GLS_pvalue = DF_GLS_result.pvalue
        DF_GLS_critical_one =  DF_GLS_result.critical_values['1%']
        DF_GLS_critical_five = DF_GLS_result.critical_values['5%']
        DF_GLS_critical_ten = DF_GLS_result.critical_values['10%']
        DF_GLS_stationary = DF_GLS_pvalue < 0.05  
    except:
        DF_GLS_stat, DF_GLS_pvalue, DF_GLS_critical_one, DF_GLS_critical_five, DF_GLS_critical_ten, DF_GLS_stationary = (
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        )  
    return {
        'Variable': name,
        #ADF
        'ADF_Statistic': adf_stat,
        'ADF_p_value': adf_pvalue,
        'ADF_Critical_1%': adf_critical_one,
        'ADF_Critical_5%': adf_critical_five,
        'ADF_Critical_10%': adf_critical_ten,
        'ADF_Stationary': adf_stationary,
        #KPSS
        'KPSS_Statistic': kpss_stat,
        'KPSS_p_value': kpss_pvalue,
        'KPSS_Critical_1%': kpss_critical_one,
        'KPSS_Critical_5%': kpss_critical_five,
        'KPSS_Critical_10%': kpss_critical_ten,
        'KPSS_Stationary': kpss_stationary,
        # DF-GLS
        'DF_GLS_Statistic': DF_GLS_stat,
        'DF_GLS_p_value': DF_GLS_pvalue,
        'DF_GLS_Critical_1%': DF_GLS_critical_one,
        'DF_GLS_Critical_5%': DF_GLS_critical_five,
        'DF_GLS_Critical_10%': DF_GLS_critical_ten,
        'DF_GLS_Stationary': DF_GLS_stationary
    }


def get_series_for_test(df, col_name):
    """
    Проверяет наличие скорректированного ряда (_adj).
    Если есть → возвращает его, иначе возвращает исходный.
    
    Returns:
    --------
    tuple: (series, source_type)
        series : pd.Series - временной ряд
        source_type : str - "Сезонно-скорректированный" или "Исходный"
    """
    
    adj_col = f'{col_name}_adj'
    
    # Проверяем наличие adj столбца и его наполненность
    if adj_col in df.columns:
        adj_series = df[adj_col].dropna()
        if len(adj_series) > 0:
            return adj_series, "Сезонно-скорректированный"
    
    # Если adj не существует или пуст - берем исходный
    original_series = df[col_name].dropna()
    return original_series, "Исходный"


def optimized_ljung_box_test(series, name, source_type, max_lags=None):
    """
    Тест Льюинга-Бокса с автоматическим подбором лагов
    
    Parameters:
    -----------
    series : pd.Series или array-like
        Временной ряд для тестирования
    name : str
        Название переменной
    source_type : str
        "Сезонно-скорректированный" или "Исходный"
    max_lags : int, optional
        Максимальное количество лагов
    
    Returns:
    --------
    dict : результаты теста
    """
    
    if len(series) < 30:
        lags = min(10, len(series) - 1)
    else:
        # Автоматический подбор лагов: sqrt(n) + 10 для месячных данных
        lags = min(int(np.sqrt(len(series))) + 10, len(series) - 1)
    
    if max_lags:
        lags = min(lags, max_lags)
    
    # Минимальное количество лагов для теста
    lags = max(5, lags)
    
    try:
        # Основной тест на выбранных лагах
        result = acorr_ljungbox(series, lags=[lags], return_df=True)
        lb_stat = result['lb_stat'].iloc[0]
        lb_pvalue = result['lb_pvalue'].iloc[0]
        
        # Дополнительно проверяем несколько ключевых лагов
        key_lags = [1, 3, 6, 12]
        key_results = []
        
        for lag in key_lags:
            if lag <= lags:
                res = acorr_ljungbox(series, lags=[lag], return_df=True)
                key_results.append({
                    'lag': lag,
                    'pvalue': res['lb_pvalue'].iloc[0]
                })
        
        return {
            'name': name,
            'source_type': source_type,
            'n_obs': len(series),
            'test_lags': lags,
            'lb_statistic': lb_stat,
            'p_value': lb_pvalue,
            'is_white_noise': lb_pvalue > 0.05,
            'key_lags_results': key_results,
            'series': series  # для визуализации
        }
    
    except Exception as e:
        print(f"Ошибка в тесте для {name} ({source_type}): {e}")
        return None


def extract_ardl_coefficients(result, y_name, print_results=True):
    """
    Извлекает ключевые коэффициенты из ARDL модели:
    - α (alpha): реакция на инфляцию
    - μ (mu): реакция на уверенность
    - γ (gamma): коэффициент инерции (лаг зависимой переменной)
    """
    params = result.params
    pvalues = result.pvalues
    
    coeffs = {
        'alpha': None,
        'mu': None,
        'gamma': None,
        'alpha_pval': None,
        'mu_pval': None,
        'gamma_pval': None
    }
    
    # Извлечение альфа (инфляция)
    if 'D_Inflation_gap' in params.index:
        coeffs['alpha'] = params['D_Inflation_gap']
        coeffs['alpha_pval'] = pvalues['D_Inflation_gap']
    
    # Извлечение мю (уверенность)
    if 'D_Ent_conf_ind_total' in params.index:
        coeffs['mu'] = params['D_Ent_conf_ind_total']
        coeffs['mu_pval'] = pvalues['D_Ent_conf_ind_total']
    
    # Извлечение гамма (лаг зависимой переменной)
    # Ищем по паттерну: y_lag.1, L.y, D_log_ROISFIX.L1, etc.
    lag_patterns = [
    f'{y_name}.L1',
    f'L.{y_name}',
    f'{y_name}_lag',
    f'{y_name}_lag.1',
    f'{y_name}_L1',
    f'L1_{y_name}',
    ]
    
    for pattern in lag_patterns:
        if pattern in params.index:
            coeffs['gamma'] = params[pattern]
            coeffs['gamma_pval'] = pvalues[pattern]
            coeffs['gamma_name'] = pattern
            break
    
    # Вывод результатов
    if print_results:
        print("\n" + "="*60)
        print("ИНТЕРПРЕТАЦИЯ КЛЮЧЕВЫХ КОЭФФИЦИЕНТОВ ARDL")
        print("="*60)
        
        # α - реакция на инфляцию
        if coeffs['alpha'] is not None:
            alpha_sr = coeffs['alpha']
            alpha_pval = coeffs['alpha_pval']
            sig_alpha = "***" if alpha_pval < 0.01 else ("**" if alpha_pval < 0.05 else "*" if alpha_pval < 0.1 else "")
            print(f"\nα (АЛЬФА) - краткосрочная реакция на инфляцию:")
            print(f"  Значение: {alpha_sr:.6f} {sig_alpha}")
            print(f"  P-value: {alpha_pval:.6f}")
            print(f"  Интерпретация: На 1% рост инфляции, ставка {'растет' if alpha_sr > 0 else 'падает'} на {abs(alpha_sr):.4f}%")
            if alpha_pval < 0.05:
                print(f"  Статистическая значимость: Значима (p < 0.05)")
            else:
                print(f"  Статистическая значимость: Незначима (p > 0.05)")
        else:
            print("\n α (альфа) не найдена в модели")
        
        # μ - реакция на уверенность
        if coeffs['mu'] is not None:
            mu_sr = coeffs['mu']
            mu_pval = coeffs['mu_pval']
            sig_mu = "***" if mu_pval < 0.01 else ("**" if mu_pval < 0.05 else "*" if mu_pval < 0.1 else "")
            print(f"\nμ (МЮ) - краткосрочная реакция на разрыв выпуска:")
            print(f"  Значение: {mu_sr:.6f} {sig_mu}")
            print(f"  P-value: {mu_pval:.6f}")
            print(f"  Интерпретация: На 1 пункт рост уверенности (прокси разрыва выпуска), ставка {'растет' if mu_sr > 0 else 'падает'} на {abs(mu_sr):.4f}%")
            if mu_pval < 0.05:
                print(f"  Статистическая значимость: Значима (p < 0.05)")
            else:
                print(f"  Статистическая значимость: Незначима (p > 0.05)")
        else:
            print("\n μ (мю) не найдена в модели")
        
        # γ - инерция (лагированная ставка)
        if coeffs['gamma'] is not None:
            gamma = coeffs['gamma']
            gamma_pval = coeffs['gamma_pval']
            speed_adj = 1 - gamma
            sig_gamma = "***" if gamma_pval < 0.01 else ("**" if gamma_pval < 0.05 else "*" if gamma_pval < 0.1 else "")
            
            print(f"\nγ (ГАММА) - коэффициент инерции (лаг зависимой переменной):")
            print(f"  Переменная: {coeffs.get('gamma_name', 'неизвестно')}")
            print(f"  Значение: {gamma:.6f} {sig_gamma}")
            print(f"  P-value: {gamma_pval:.6f}")
            
            print(f"\n  Интерпретация:")
            print(f"    - Инерция системы: {gamma:.4f} (доля предыдущего периода)")
            print(f"    - Скорость адаптации к равновесию: {speed_adj:.4f}")
            print(f"    - На каждый период корректируется: {speed_adj*100:.2f}% отклонения")
            
            if 0 < gamma < 1:
                print(f"    - Модель СТАБИЛЬНА (динамическое равновесие)")
            elif gamma >= 1:
                print(f"    -  Модель НЕСТАБИЛЬНА (взрывной процесс)")
            elif gamma < 0:
                print(f"    - ОТРИЦАТЕЛЬНАЯ инерция (колебания вокруг равновесия)")
        else:
            print("\n γ (гамма) не найдена в модели")
    
    return coeffs

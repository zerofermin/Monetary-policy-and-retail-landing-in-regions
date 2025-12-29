import os
import pandas as pd


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


def ensure_results_dir(path='Results'):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def add_model_set(aggregator, spec):
    spec_name = spec.get('spec_name')
    dependent_var = spec.get('dependent_var')
    subsample = spec.get('subsample')
    results = spec.get('results', {})
    se_type = spec.get('se_type', None)

    pooled_res = results.get('pooled')
    if pooled_res is not None:
        aggregator.add_model_results(
            pooled_res,
            dependent_var,
            subsample,
            'POOL',
            spec_name,
            se_type=se_type
        )

    fe_res = results.get('fe')
    if fe_res is not None:
        aggregator.add_model_results(
            fe_res,
            dependent_var,
            subsample,
            'FE',
            spec_name,
            se_type=se_type
        )

    re_res = results.get('re')
    if re_res is not None:
        aggregator.add_model_results(
            re_res,
            dependent_var,
            subsample,
            'RE',
            spec_name,
            se_type=se_type
        )


def build_and_export(aggregator, out_path, include_pvalues=True, decimals=3):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        ensure_results_dir(out_dir)
    full_table = aggregator.build_table(
        include_pvalues=include_pvalues,
        decimals=decimals
    )
    full_table.to_excel(out_path)
    return full_table

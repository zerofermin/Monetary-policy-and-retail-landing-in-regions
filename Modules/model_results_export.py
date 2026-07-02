import os
import re
import pandas as pd

SHOCK_PREFIXES = ('d_Mon_Shock', 'd_ROISFIX', 'd_MIACR')


def normalize_shock_var_name(var: str) -> str:
    """
    Убирает суффикс _lagN только у шоковых переменных.
    Базовые контролы (d_Int_Rate_ConsCred_lag1 и т.п.) остаются без изменений.
    """
    if any(var.startswith(p) for p in SHOCK_PREFIXES):
        return re.sub(r'_lag\d+$', '', var)
    return var


def select_estimator(hausman_pval) -> str:
    """
    Тест Хаусмана: p > 0.05 → RE (случайные эффекты состоятельны), иначе → FE.
    При None / NaN возвращает 'FE' (консервативный вариант).
    """
    try:
        p = float(hausman_pval)
        return 'RE' if p > 0.05 else 'FE'
    except (TypeError, ValueError):
        return 'FE'


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
                    decimals: int = 3,
                    var_rename_fn=None) -> pd.DataFrame:
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
        spec_order = list(dict.fromkeys(df['specification']))
        model_order = ['POOL', 'FE', 'RE']
        col_pairs = []
        for spec in spec_order:
            spec_models = df.loc[df['specification'] == spec, 'model_type'].unique().tolist()
            for model_type in model_order:
                if model_type in spec_models:
                    col_pairs.append((spec, model_type))

        col_index = pd.MultiIndex.from_tuples(col_pairs, names=['specification', 'model_type'])
        col_names = [f"{spec} ({model_type})" for spec, model_type in col_pairs]

        # тело таблицы
        body_dict = {}
        for spec, model_type in col_pairs:
            subset = df[(df['specification'] == spec) & (df['model_type'] == model_type)]
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

                display_var = var_rename_fn(var) if var_rename_fn else var
                col_data[display_var] = cell
            body_dict[f"{spec} ({model_type})"] = col_data

        body_df = pd.DataFrame(body_dict).reindex(columns=col_names).fillna('')

        # Зависимая переменная (верхняя строка)
        dep_names = (df.groupby(['specification', 'model_type'])['dependent_var']
                       .first()
                       .reindex(col_index))
        dep_names.index = col_names
        dep_row = pd.DataFrame([dep_names], index=['Зависимая переменная'])

        # R^2
        r2_vals = (df.groupby(['specification', 'model_type'])['r_squared']
                     .max()
                     .round(decimals)
                     .reindex(col_index))
        r2_vals.index = col_names
        r2_row = pd.DataFrame([r2_vals], index=['R^2'])

        # Тип стандартных ошибок (под R^2)
        se_vals = (df.groupby(['specification', 'model_type'])['se_type']
                     .first()
                     .reindex(col_index))
        se_vals.index = col_names
        se_row = pd.DataFrame([se_vals], index=['Тип стандартных ошибок'])

        # n-obs (число наблюдений)
        nobs_vals = (df.groupby(['specification', 'model_type'])['nobs']
                       .max()
                       .reindex(col_index))
        nobs_vals.index = col_names
        nobs_row = pd.DataFrame([nobs_vals], index=['n-obs'])

        # Метод оценки (предпоследняя строка)
        model_names = (df.groupby(['specification', 'model_type'])['model_type']
                         .first()
                         .reindex(col_index))
        model_names.index = col_names
        model_row = pd.DataFrame([model_names], index=['Метод оценки'])

        # Выборка (последняя строка)
        subsample_vals = (df.groupby(['specification', 'model_type'])['subsample']
                            .first()
                            .reindex(col_index))
        subsample_vals.index = col_names
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


def build_and_export_by_shock_category_with_tests(
    aggregator,
    shock_spec_meta,
    tests_by_column,
    out_path,
    include_pvalues=True,
    decimals=3,
    test_decimals=6,
    shock_order=None,
    model_types=None,
    interaction_col_suffix='*',
    interaction_suffixes=None,
    hausman_key='Hausman (FE vs RE) p-value (robust)',
):
    """
    Записывает один Excel-лист на тип шока. Внутри каждого листа:
    - Четыре стекованных блока: Рекомендованная модель, POOL, FE, RE.
    - Колонки: Lag 0 … Lag 6 (простые), разделитель, Lag 0* … Lag 6* (event),
      разделитель, Lag 0** … Lag 6** (cluster).
    - Строки шоков нормализованы (суффикс лага убран).
    - Раздел ТЕСТЫ внизу с предрассчитанной строкой 'Выбор RE или FE'.

    shock_spec_meta — список словарей {spec_name, shock_group, lag_num, is_interaction}.
      is_interaction может быть bool (True/False) или строкой типа ('event'/'cluster'/None).
    interaction_suffixes — словарь {тип: суффикс}, напр. {'event': '*', 'cluster': '**'}.
    hausman_key — ключ теста Хаусмана в tests_by_column для выбора RE/FE.
    """
    if shock_order is None:
        shock_order = ['Mon_Shock', 'ROISFIX', 'MIACR']
    if model_types is None:
        model_types = ['POOL', 'FE', 'RE']
    if interaction_suffixes is None:
        interaction_suffixes = {'event': interaction_col_suffix, 'cluster': '**'}

    out_dir = os.path.dirname(out_path)
    if out_dir:
        ensure_results_dir(out_dir)

    # Карта: оригинальное имя колонки → метка лага ("Lag N", "Lag N*", "Lag N**" и т.д.)
    col_to_lag_label = {}
    for m in shock_spec_meta:
        for mt in model_types:
            original = f"{m['spec_name']} ({mt})"
            itype = m['is_interaction']
            if itype:
                suffix = interaction_suffixes.get(itype, interaction_col_suffix)
            else:
                suffix = ''
            col_to_lag_label[original] = f"Lag {m['lag_num']}{suffix}"

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for shock_type in shock_order:
            simple_specs = [m['spec_name'] for m in shock_spec_meta
                            if m['shock_group'] == shock_type and not m['is_interaction']]

            # Группируем interaction-спеки по типу, сохраняя порядок встречи
            inter_specs_by_type = {}
            for m in shock_spec_meta:
                if m['shock_group'] == shock_type and m['is_interaction']:
                    itype = m['is_interaction']
                    inter_specs_by_type.setdefault(itype, [])
                    if m['spec_name'] not in inter_specs_by_type[itype]:
                        inter_specs_by_type[itype].append(m['spec_name'])

            if not simple_specs and not inter_specs_by_type:
                continue

            all_inter_specs = [s for specs in inter_specs_by_type.values() for s in specs]

            # Все оригинальные имена колонок для этого типа шока
            all_orig_cols = []
            for spec_name in simple_specs + all_inter_specs:
                for mt in model_types:
                    all_orig_cols.append(f"{spec_name} ({mt})")

            # Имена тестов (порядок как в первой встреченной колонке)
            test_names = []
            for col in all_orig_cols:
                for name in tests_by_column.get(col, {}).keys():
                    if name not in test_names:
                        test_names.append(name)

            # Упорядоченные метки колонок (простые + разделитель + взаимодействия)
            def lag_col_labels(specs, suffix=''):
                seen = []
                for entry in sorted(
                    [x for x in shock_spec_meta if x['spec_name'] in specs],
                    key=lambda x: x['lag_num']
                ):
                    lbl = f"Lag {entry['lag_num']}{suffix}"
                    if lbl not in seen:
                        seen.append(lbl)
                return seen

            separator_labels = set()
            simple_lag_labels = lag_col_labels(simple_specs, '')
            all_col_labels = list(simple_lag_labels)
            _sep_count = 0
            for itype, ispecs in inter_specs_by_type.items():
                suffix = interaction_suffixes.get(itype, interaction_col_suffix)
                itype_labels = lag_col_labels(ispecs, suffix)
                if all_col_labels and itype_labels:
                    _sep_count += 1
                    sep = ' ' * _sep_count
                    separator_labels.add(sep)
                    all_col_labels.append(sep)
                all_col_labels.extend(itype_labels)

            # Строим полную таблицу один раз для этого типа шока
            all_specs = simple_specs + all_inter_specs
            tbl_full = aggregator.build_table(
                specification_filter=all_specs,
                include_pvalues=include_pvalues,
                decimals=decimals,
                var_rename_fn=normalize_shock_var_name,
            )

            # Нарезаем по типу оценщика и переименовываем колонки
            mt_tables = {}
            for mt in model_types:
                mt_cols = [c for c in tbl_full.columns if c.endswith(f'({mt})')]
                if not mt_cols:
                    continue
                tbl_mt = tbl_full[mt_cols].copy()
                tbl_mt.columns = [col_to_lag_label.get(c, c) for c in tbl_mt.columns]
                tbl_mt = tbl_mt.reindex(columns=all_col_labels, fill_value='')
                mt_tables[mt] = tbl_mt

            # Вспомогательная функция: p-значение теста Хаусмана для данной метки лага
            def get_hausman_p(lag_label):
                for orig_col, display in col_to_lag_label.items():
                    if display == lag_label:
                        v = tests_by_column.get(orig_col, {}).get(hausman_key)
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                pass
                return None

            # Блок «Рекомендованная модель»: для каждой колонки выбираем FE или RE
            tbl_fe = mt_tables.get('FE')
            tbl_re = mt_tables.get('RE')
            tbl_rec = None
            if tbl_fe is not None and tbl_re is not None:
                rec_data = {}
                for lbl in all_col_labels:
                    hausman_p = get_hausman_p(lbl)
                    chosen = select_estimator(hausman_p)
                    src = mt_tables.get(chosen, tbl_fe)
                    rec_data[lbl] = src[lbl] if lbl in src.columns else ''
                tbl_rec = pd.DataFrame(rec_data, index=tbl_fe.index)
                # Явно проставляем выбранный метод в строке «Метод оценки»
                if 'Метод оценки' in tbl_rec.index:
                    for lbl in all_col_labels:
                        if lbl in separator_labels:
                            continue
                        hausman_p = get_hausman_p(lbl)
                        tbl_rec.at['Метод оценки', lbl] = select_estimator(hausman_p)
            elif tbl_fe is not None:
                tbl_rec = tbl_fe.copy()
            elif tbl_re is not None:
                tbl_rec = tbl_re.copy()

            # Стекуем блоки: Рекомендованная модель, POOL, FE, RE
            stacked_parts = []

            def add_block(label, tbl):
                header = pd.DataFrame([{c: '' for c in all_col_labels}], index=[label])
                stacked_parts.append(pd.concat([header, tbl]))
                stacked_parts.append(
                    pd.DataFrame([{c: '' for c in all_col_labels}], index=[''])
                )

            if tbl_rec is not None:
                add_block('Рекомендованная модель', tbl_rec)

            for mt in model_types:
                if mt in mt_tables:
                    add_block(mt, mt_tables[mt])

            if not stacked_parts:
                continue

            sheet_table = pd.concat(stacked_parts)

            # Раздел ТЕСТЫ с предрассчитанной строкой «Выбор RE или FE»
            if test_names:
                test_rows = {}
                for tname in test_names:
                    row = {}
                    for lag_label in all_col_labels:
                        val = ''
                        for orig_col, display in col_to_lag_label.items():
                            if display == lag_label:
                                v = tests_by_column.get(orig_col, {}).get(tname)
                                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                                    try:
                                        val = f"{float(v):.{test_decimals}f}"
                                    except Exception:
                                        pass
                                    break
                        row[lag_label] = val
                    test_rows[tname] = row

                # Строка «Выбор RE или FE»
                vybor_row = {}
                for lag_label in all_col_labels:
                    if lag_label in separator_labels:
                        vybor_row[lag_label] = ''
                        continue
                    hausman_p = get_hausman_p(lag_label)
                    vybor_row[lag_label] = select_estimator(hausman_p) if hausman_p is not None else ''
                test_rows['Выбор RE или FE'] = vybor_row

                tests_header = pd.DataFrame([{c: '' for c in all_col_labels}], index=['ТЕСТЫ'])
                tests_df = pd.DataFrame.from_dict(test_rows, orient='index').reindex(
                    columns=all_col_labels).fillna('')
                blank_before = pd.DataFrame([{c: '' for c in all_col_labels}], index=[''])
                sheet_table = pd.concat([sheet_table, blank_before, tests_header, tests_df])

            sheet_table.to_excel(writer, sheet_name=shock_type)

    return out_path


def build_and_export_with_tests(
    aggregator,
    tests_by_column,
    out_path,
    include_pvalues=True,
    decimals=3,
    test_decimals=6
):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        ensure_results_dir(out_dir)

    full_table = aggregator.build_table(
        include_pvalues=include_pvalues,
        decimals=decimals
    )

    col_names = list(full_table.columns)
    test_names = []
    for col in col_names:
        col_tests = tests_by_column.get(col, {})
        for name in col_tests.keys():
            if name not in test_names:
                test_names.append(name)

    if not test_names:
        full_table.to_excel(out_path)
        return full_table

    test_rows = {}
    for test_name in test_names:
        row = {}
        for col in col_names:
            val = tests_by_column.get(col, {}).get(test_name)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                row[col] = ''
            else:
                try:
                    row[col] = f"{float(val):.{test_decimals}f}"
                except Exception:
                    row[col] = ''
        test_rows[test_name] = row

    tests_df = pd.DataFrame.from_dict(test_rows, orient='index')
    tests_df = tests_df.reindex(columns=col_names).fillna('')

    blank_row = pd.DataFrame([[''] * len(col_names)], index=[''], columns=col_names)
    header_row = pd.DataFrame([[''] * len(col_names)], index=['ТЕСТЫ'], columns=col_names)

    full_table = pd.concat([full_table, blank_row, header_row, tests_df], axis=0)
    full_table.to_excel(out_path)
    return full_table

import pickle
from pathlib import Path


def build_shock_variants(dependent_var, exog_vars_base, shock_vars):
    def as_list(value):
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    variants = []
    for shocks in shock_vars:
        expanded = list(exog_vars_base)
        expanded.extend(as_list(shocks))
        variants.append(expanded)
    return {
        "dependent_var": dependent_var,
        "exog_variants": variants,
    }


MODELS_PATH = Path("Models.pkl")
MODEL_PREFIX = "Модель"

def load_models(path=MODELS_PATH):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _next_model_name(existing):
    numbers = [
        int(str(name).replace(MODEL_PREFIX, "").strip())
        for name in existing.keys()
        if str(name).startswith(MODEL_PREFIX) and str(name).replace(MODEL_PREFIX, "").strip().isdigit()
    ]
    next_num = max(numbers, default=0) + 1
    return f"{MODEL_PREFIX} {next_num}"


def save_model_spec(dependent_var, exog_vars_base, shock_vars, model_name=None, path=MODELS_PATH):
    models = load_models(path)
    if not model_name or str(model_name).strip() == "":
        model_name = _next_model_name(models)
    models[model_name] = {
        "dependent_var": dependent_var,
        "exog_vars_base": list(exog_vars_base),
        "shock_vars": list(shock_vars),
    }
    with Path(path).open("wb") as f:
        pickle.dump(models, f)
    return model_name, models

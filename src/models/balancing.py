"""Modulo de balanceo de clases para datos desbalanceados."""

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def apply_smote_enn(X, y, random_state=42):
    """Aplica SMOTE + Edited Nearest Neighbors.

    SMOTE genera muestras sinteticas de la clase minoritaria,
    ENN elimina instancias ruidosas de ambas clases.

    Parameters
    ----------
    X : array-like
    y : array-like
    random_state : int

    Returns
    -------
    X_resampled, y_resampled
    """
    smote_enn = SMOTEENN(random_state=random_state)
    X_res, y_res = smote_enn.fit_resample(X, y)
    print(f"  SMOTE-ENN: {len(y)} -> {len(y_res)} samples "
          f"(pos: {y.sum()} -> {y_res.sum()})")
    return X_res, y_res


def apply_borderline_smote(X, y, random_state=42):
    """Aplica BorderlineSMOTE: genera muestras solo en la frontera de decision.

    Parameters
    ----------
    X : array-like
    y : array-like
    random_state : int

    Returns
    -------
    X_resampled, y_resampled
    """
    bsmote = BorderlineSMOTE(random_state=random_state)
    X_res, y_res = bsmote.fit_resample(X, y)
    print(f"  BorderlineSMOTE: {len(y)} -> {len(y_res)} samples "
          f"(pos: {y.sum()} -> {y_res.sum()})")
    return X_res, y_res


def create_balanced_pipeline(estimator, strategy="smote_enn", random_state=42):
    """Crea un ImbPipeline con resampling + estimator para usar con CV.

    Parameters
    ----------
    estimator : sklearn estimator
    strategy : str
        "smote_enn" o "borderline_smote"
    random_state : int

    Returns
    -------
    ImbPipeline
    """
    if strategy == "smote_enn":
        sampler = SMOTEENN(random_state=random_state)
    elif strategy == "borderline_smote":
        sampler = BorderlineSMOTE(random_state=random_state)
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")

    pipeline = ImbPipeline([
        ("resampler", sampler),
        ("classifier", estimator),
    ])
    return pipeline

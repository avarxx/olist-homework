import hashlib
import os
import re
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import sqlalchemy

from olist_model import fit_model, save_predictor, split_by_time

REQUIRED_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
    "price",
    "freight",
    "weight",
    "n_items",
    "installments",
    "est_days",
    "customer_state",
    "seller_state",
    "category",
    "payment_type",
    "late",
]
NUMERIC_COLUMNS = ["price", "freight", "weight", "n_items", "installments", "est_days"]
CATEGORICAL_COLUMNS = ["customer_state", "seller_state", "category", "payment_type"]
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
LEARNING_RATE = 0.003
EXPERIMENT_NAME = "olist_training"

ORDERS_QUERY = sqlalchemy.text(
    f"SELECT {', '.join(REQUIRED_COLUMNS)} FROM public.orders "
    "ORDER BY order_purchase_timestamp, order_id"
)


class ValidationError(ValueError):
    pass


def default_epochs():
    return int(os.environ.get("OLIST_EPOCHS", "5"))


def default_seed():
    return int(os.environ.get("OLIST_SEED", "42"))


def run_directory(run_key):
    work_dir = Path(os.environ.get("OLIST_WORKDIR", "/opt/olist/work"))
    safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_key))
    return (work_dir / "runs" / safe_key).resolve()


def read_orders(path):
    return pd.read_csv(path, encoding="utf-8", dtype={"order_id": str})


def write_csv(frame, path):
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(run_key):
    engine = sqlalchemy.create_engine(os.environ["DATABASE_URL"])
    try:
        with engine.connect() as connection:
            orders = pd.read_sql(ORDERS_QUERY, connection)
    finally:
        engine.dispose()

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    ).dt.strftime(TIMESTAMP_FORMAT)

    target_dir = run_directory(run_key)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot = target_dir / "orders.csv"
    write_csv(orders, snapshot)
    return str(snapshot)


def find_problems(orders):
    if orders.empty:
        return ["snapshot has no rows"]

    problems = []
    if not orders.columns.is_unique:
        problems.append("column names are not unique")
    missing = [column for column in REQUIRED_COLUMNS if column not in orders.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        return problems

    order_ids = orders["order_id"]
    if order_ids.isna().any():
        problems.append("order_id has empty values")
    duplicates = int(order_ids.duplicated().sum())
    if duplicates:
        problems.append(f"order_id has {duplicates} duplicated values")

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(orders[column], errors="coerce").to_numpy(dtype=np.float64)
        bad_rows = int((~np.isfinite(values)).sum())
        if bad_rows:
            problems.append(f"{column} has {bad_rows} missing or non-finite values")

    labels = pd.to_numeric(orders["late"], errors="coerce")
    if not labels.isin([0, 1]).all():
        problems.append("late must contain only 0 and 1")
    elif labels.nunique() != 2:
        problems.append("late must contain both classes")

    timestamps = pd.to_datetime(
        orders["order_purchase_timestamp"], format="ISO8601", errors="coerce", utc=True
    )
    bad_timestamps = int(timestamps.isna().sum())
    if bad_timestamps:
        problems.append(f"order_purchase_timestamp has {bad_timestamps} invalid values")

    for column in CATEGORICAL_COLUMNS:
        empty_rows = int(orders[column].isna().sum())
        if empty_rows:
            problems.append(f"{column} has {empty_rows} empty values")

    return problems


def validate(snapshot):
    path = Path(snapshot)
    if path.suffix != ".csv":
        raise ValidationError(f"expected a .csv file, got {path}")
    if not path.is_file():
        raise ValidationError(f"snapshot not found: {path}")

    try:
        orders = read_orders(path)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValidationError(f"cannot read snapshot {path}: {error}") from error

    problems = find_problems(orders)
    if problems:
        raise ValidationError(f"snapshot {path} is invalid: " + "; ".join(problems))
    return str(path)


def prepare(snapshot):
    snapshot = Path(snapshot)
    snapshot_sha256 = file_sha256(snapshot)
    cutoff = os.environ.get("OLIST_CUTOFF", "2017-07-31")

    orders = read_orders(snapshot)
    train_rows, validation_rows = split_by_time(orders, cutoff=cutoff)

    train_path = snapshot.parent / "train.csv"
    validation_path = snapshot.parent / "validation.csv"
    write_csv(train_rows, train_path)
    write_csv(validation_rows, validation_path)

    return {
        "snapshot": str(snapshot),
        "snapshot_sha256": snapshot_sha256,
        "cutoff": cutoff,
        "train": str(train_path),
        "validation": str(validation_path),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
    }


def train(prepared, *, dag_run_id, epochs=None, seed=None):
    epochs = default_epochs() if epochs is None else int(epochs)
    seed = default_seed() if seed is None else int(seed)

    train_rows = read_orders(prepared["train"])
    validation_rows = read_orders(prepared["validation"])
    model_dir = Path(prepared["train"]).parent / "model"

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ.get("OLIST_EXPERIMENT", EXPERIMENT_NAME))

    with mlflow.start_run(run_name=f"olist-{dag_run_id}") as run:
        mlflow.set_tags({"dag_run_id": dag_run_id, "snapshot_path": prepared["snapshot"]})
        mlflow.log_params({
            "epochs": epochs,
            "seed": seed,
            "lr": LEARNING_RATE,
            "cutoff": prepared["cutoff"],
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "snapshot_sha256": prepared["snapshot_sha256"],
        })

        predictor, history = fit_model(
            train_rows, validation_rows, epochs=epochs, seed=seed, lr=LEARNING_RATE
        )
        for row in history:
            mlflow.log_metrics(
                {"train_loss": row["train_loss"], "validation_pr_auc": row["validation_pr_auc"]},
                step=row["epoch"],
            )

        if model_dir.exists():
            shutil.rmtree(model_dir)
        save_predictor(predictor, model_dir)
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        return run.info.run_id

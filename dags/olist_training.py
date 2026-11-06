import os
from datetime import datetime, timedelta, timezone

from airflow.sdk import Param, dag, get_current_context, task

DEFAULT_EPOCHS = int(os.environ.get("OLIST_EPOCHS", "5"))
DEFAULT_SEED = int(os.environ.get("OLIST_SEED", "42"))


@dag(
    dag_id="olist_training",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=1),
    default_args={
        "retries": 0,
        "execution_timeout": timedelta(minutes=10),
    },
    params={
        "epochs": Param(DEFAULT_EPOCHS, type="integer", minimum=1, maximum=100),
        "seed": Param(DEFAULT_SEED, type="integer", minimum=0),
    },
    tags=["olist", "training"],
)
def olist_training():
    @task(retries=3, retry_delay=timedelta(seconds=30))
    def extract():
        import olist_pipeline

        context = get_current_context()
        return olist_pipeline.extract(context["dag_run"].run_id)

    @task(retries=0, trigger_rule="all_success", execution_timeout=timedelta(minutes=5))
    def validate(snapshot):
        import olist_pipeline

        return olist_pipeline.validate(snapshot)

    @task(trigger_rule="all_success")
    def prepare(snapshot):
        import olist_pipeline

        return olist_pipeline.prepare(snapshot)

    @task(trigger_rule="all_success", execution_timeout=timedelta(minutes=30))
    def train(prepared):
        import olist_pipeline

        context = get_current_context()
        return olist_pipeline.train(
            prepared,
            dag_run_id=context["dag_run"].run_id,
            epochs=context["params"]["epochs"],
            seed=context["params"]["seed"],
        )

    train(prepare(validate(extract())))


olist_training()

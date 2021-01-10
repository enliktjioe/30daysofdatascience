# Python standard modules
from datetime import datetime, timedelta# Airflow modules
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    # Start on 27th of June, 2020
    'start_date': datetime(2020, 6, 27),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    # In case of errors, do one retry
    'retries': 1,
    # Do the retry with 30 seconds delay after the error
    'retry_delay': timedelta(seconds=30),
    # Run once every 15 minutes
    'schedule_interval': '*/15 * * * *'
}

with DAG(dag_id='simple_bash_dag',
    default_args=default_args,
    schedule_interval=None,
    tags=[‘my_dags’],
) as dag:    #Here we define our first task
    t1 = BashOperator(bash_command=”touch ~/my_bash_file.txt”, task_id=”create_file”)    #Here we define our second task
    t2 = BashOperator(bash_command=”mv ~/my_bash_file.txt ~/my_bash_file_changed.txt”, task_id=”change_file_name”)    # Configure T2 to be dependent on T1’s execution
    t1 >> t2

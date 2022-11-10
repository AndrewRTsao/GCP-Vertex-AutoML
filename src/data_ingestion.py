import os
from google.cloud import bigquery

def ingest_data():

    def create_dataset(dataset_id):

        # Construct full Dataset object from client to send to the API
        dataset_ref = bigquery.DatasetReference.from_string(dataset_id, default_project=client.project)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = os.getenv('DATASET_LOCATION')

        # Send the dataset to the API for creation, with an explicit timeout.
        # Raises google.api_core.exceptions.Conflict if the Dataset already
        # exists within the project.
        dataset = client.create_dataset(dataset, timeout=60)
        print("Created dataset {}.{}".format(client.project, dataset.dataset_id))

    def load_table_uri_csv(table_name: str, schema: object):

        # Creating fully qualified table name
        table_id = GCP_PROJECT + "." + dataset_id + "." + table_name

        # Defining job config for loading data from GCS into BQ table
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            skip_leading_rows=1,
            source_format=bigquery.SourceFormat.CSV,
        )
        uri = INPUT_GCS_PATH + table_name + ".csv" # Path where file is stored on GCS bucket

        # Loading data into BQ table from GCS bucket
        load_job = client.load_table_from_uri(
            uri, table_id, job_config=job_config
        )  
        load_job.result() 

        # Retrieving results
        destination_table = client.get_table(table_id)  
        print("Loaded {} rows ".format(destination_table.num_rows) + "into " + table_id)

    # Initialize client, create BQ dataset, and then create corresponding tables from specified GCS location (assume Kaggle files loaded here)
    INPUT_GCS_PATH = os.getenv('INPUT_GCS_PATH')
    GCP_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset_id = os.getenv('DATASET_NAME')

    client = bigquery.Client()
    create_dataset(dataset_id)

    # Load data for PdM_errors
    table_name = "PdM_errors"
    schema = [
            bigquery.SchemaField("DATETIME", "TIMESTAMP"),
            bigquery.SchemaField("MACHINEID", "STRING"),
            bigquery.SchemaField("ERRORID", "STRING"),
        ]

    load_table_uri_csv(table_name, schema)

    # Load data for PdM_failures
    table_name = "PdM_failures"
    schema = [
            bigquery.SchemaField("DATETIME", "TIMESTAMP"),
            bigquery.SchemaField("MACHINEID", "STRING"),
            bigquery.SchemaField("FAILURE", "STRING"),
        ]

    load_table_uri_csv(table_name, schema)

    # Load data for PdM_machines
    table_name = "PdM_machines"
    schema = [
            bigquery.SchemaField("MACHINEID", "STRING"),
            bigquery.SchemaField("MODEL", "STRING"),
            bigquery.SchemaField("AGE", "INTEGER"),
        ]

    load_table_uri_csv(table_name, schema)

    # Load data for PdM_maint
    table_name = "PdM_maint"
    schema = [
            bigquery.SchemaField("DATETIME", "TIMESTAMP"),
            bigquery.SchemaField("MACHINEID", "STRING"),
            bigquery.SchemaField("COMP", "STRING"),
        ]

    load_table_uri_csv(table_name, schema)

    # Load data for PdM_telemetry
    table_name = "PdM_telemetry"
    schema = [
            bigquery.SchemaField("DATETIME", "TIMESTAMP"),
            bigquery.SchemaField("MACHINEID", "STRING"),
            bigquery.SchemaField("VOLT", "FLOAT"),
            bigquery.SchemaField("ROTATE", "FLOAT"),
            bigquery.SchemaField("PRESSURE", "FLOAT"),
            bigquery.SchemaField("VIBRATION", "FLOAT"),
        ]

    load_table_uri_csv(table_name, schema)

if __name__ == "__main__":

    ingest_data()
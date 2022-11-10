import os
import datetime
from typing import List, Union, Optional

from google.cloud import aiplatform

def create_featurestore(
    featurestore_id: str,
    online_store_fixed_node_count: int = 1,
    sync: bool = True,
):

    fs = aiplatform.Featurestore.create(
        featurestore_id=featurestore_id,
        online_store_fixed_node_count=online_store_fixed_node_count,
        sync=sync,
    )

    return fs

def create_entity_type_and_features(
    entity_type_id: str,
    featurestore_name: str,
    feature_configs: object,
    sync: bool = True,
):

    entity_type = aiplatform.EntityType.create(
        entity_type_id=entity_type_id, featurestore_name=featurestore_name
    )

    entity_type.batch_create_features(feature_configs=feature_configs, sync=sync)

    return entity_type

def batch_ingest_features(
    entity_type_id: str,
    featurestore_id: str,
    entity_id_field: str,
    feature_time: Union[str, datetime.datetime],
    bq_source_uri: str,
):
    
    entity_type = aiplatform.featurestore.EntityType(
        entity_type_name=entity_type_id, featurestore_id=featurestore_id
    )

    feature_ids = [feature.name for feature in entity_type.list_features()]

    entity_type.ingest_from_bq(
        feature_ids=feature_ids,
        feature_time=feature_time,
        bq_source_uri=bq_source_uri,
        entity_id_field=entity_id_field,
    )

def batch_serve_features_to_bq(
    featurestore_name: str,
    bq_destination_output_uri: str,
    read_instances_uri: str,
    serving_feature_ids: object,
    pass_through_fields: Optional[List[str]] = None,
    sync: bool = True,
):

    fs = aiplatform.featurestore.Featurestore(featurestore_name=featurestore_name)

    fs.batch_serve_to_bq(
        bq_destination_output_uri=bq_destination_output_uri,
        serving_feature_ids=serving_feature_ids,
        read_instances_uri=read_instances_uri,
        pass_through_fields=pass_through_fields,
        sync=sync,
    )

if __name__ == "__main__":

    # Initialize variables
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset_name = os.getenv('DATASET_NAME')
    region = os.getenv('VERTEX_REGION')
    fs_id = os.getenv('DATASET_NAME') + "_fs"
    fs_index = "machine_id"
    time_index = "ts"

    BQ_PATTERN = "bq://{project}.{dataset}.{table}" # To help construct BQ URIs

    # Initializing client and create feature store
    aiplatform.init(project=project_id, location=region)

    create_featurestore(fs_id)

    # Creating entity types and associated batch features (e.g. feature sets), then ingest the data

    # vm_errors table

    fs_name = "vm_errors"
    bq_source_uri = BQ_PATTERN.format(
        project=project_id, dataset=dataset_name, table=fs_name
    )
    features= {
        "error1": {
            "value_type": "BOOL",
            "description": "Whether VM encountered error 1",
        },
        "error2": {
            "value_type": "BOOL",
            "description": "Whether VM encountered error 2",
        },
        "error3": {
            "value_type": "BOOL",
            "description": "Whether VM encountered error 3",
        },
        "error4": {
            "value_type": "BOOL",
            "description": "Whether VM encountered error 4",
        },
        "error5": {
            "value_type": "BOOL",
            "description": "Whether VM encountered error 5",
        },
    }

    create_entity_type_and_features(fs_name, fs_id, features)
    batch_ingest_features(fs_name, fs_id, fs_index, time_index, bq_source_uri)

    # vm_failures table

    fs_name = "vm_failures"
    bq_source_uri = BQ_PATTERN.format(
        project=project_id, dataset=dataset_name, table=fs_name
    )
    features= {
        "comp1_failure": {
            "value_type": "BOOL",
            "description": "Whether component 1 failed",
        },
        "comp2_failure": {
            "value_type": "BOOL",
            "description": "Whether component 2 failed",
        },
        "comp3_failure": {
            "value_type": "BOOL",
            "description": "Whether component 3 failed",
        },
        "comp4_failure": {
            "value_type": "BOOL",
            "description": "Whether component 4 failed",
        },
    }

    create_entity_type_and_features(fs_name, fs_id, features)
    batch_ingest_features(fs_name, fs_id, fs_index, time_index, bq_source_uri)

    # vm_hourly_status table

    fs_name = "vm_hourly_status"
    bq_source_uri = BQ_PATTERN.format(
        project=project_id, dataset=dataset_name, table=fs_name
    )
    features= {
        "hours_since_last_maint": {
            "value_type": "INT64",
            "description": "Number of hours since last maintenance of machine",
        },
        "hours_since_last_comp1_maint": {
            "value_type": "INT64",
            "description": "Number of hours since last maintenance of component 1 in machine",
        },
        "hours_since_last_comp2_maint": {
            "value_type": "INT64",
            "description": "Number of hours since last maintenance of component 2 in machine",
        },
        "hours_since_last_comp3_maint": {
            "value_type": "INT64",
            "description": "Number of hours since last maintenance of component 3 in machine",
        },
        "hours_since_last_comp4_maint": {
            "value_type": "INT64",
            "description": "Number of hours since last maintenance of component 4 in machine",
        },
        "hours_since_last_failure": {
            "value_type": "INT64",
            "description": "Number of hours since last failure of machine",
        },
        "hours_since_last_comp1_failure": {
            "value_type": "INT64",
            "description": "Number of hours since last failure of component 1 in machine",
        },
        "hours_since_last_comp2_failure": {
            "value_type": "INT64",
            "description": "Number of hours since last failure of component 2 in machine",
        },
        "hours_since_last_comp3_failure": {
            "value_type": "INT64",
            "description": "Number of hours since last failure of component 3 in machine",
        },
        "hours_since_last_comp4_failure": {
            "value_type": "INT64",
            "description": "Number of hours since last failure of component 4 in machine",
        },
        "hours_since_last_error": {
            "value_type": "INT64",
            "description": "Number of hours since last error in machine",
        },
        "hours_since_last_error1": {
            "value_type": "INT64",
            "description": "Number of hours since last instance of error1 in machine",
        },
        "hours_since_last_error2": {
            "value_type": "INT64",
            "description": "Number of hours since last instance of error2 in machine",
        },
        "hours_since_last_error3": {
            "value_type": "INT64",
            "description": "Number of hours since last instance of error3 in machine",
        },
        "hours_since_last_error4": {
            "value_type": "INT64",
            "description": "Number of hours since last instance of error4 in machine",
        },
        "hours_since_last_error5": {
            "value_type": "INT64",
            "description": "Number of hours since last instance of error5 in machine",
        },
    }

    create_entity_type_and_features(fs_name, fs_id, features)
    batch_ingest_features(fs_name, fs_id, fs_index, time_index, bq_source_uri)

    # vm_machines table

    fs_name = "vm_machines"
    dummy_timestamp = datetime.datetime(year=2015, month=1, day=1, hour=0, minute=0, second=0)
    bq_source_uri = BQ_PATTERN.format(
        project=project_id, dataset=dataset_name, table=fs_name
    )
    features= {
        "model": {
            "value_type": "STRING",
            "description": "Model of machine",
        },
        "age": {
            "value_type": "INT64",
            "description": "Age of machine",
        },
    }

    create_entity_type_and_features(fs_name, fs_id, features)
    batch_ingest_features(fs_name, fs_id, fs_index, dummy_timestamp, bq_source_uri) # dummy timestamp because there is no time index in the base table

    # vm_maintenance table

    fs_name = "vm_maintenance"
    bq_source_uri = BQ_PATTERN.format(
        project=project_id, dataset=dataset_name, table=fs_name
    )
    features= {
        "comp1_maint": {
            "value_type": "BOOL",
            "description": "Whether component 1 was replaced during maintenance",
        },
        "comp2_maint": {
            "value_type": "BOOL",
            "description": "Whether component 2 was replaced during maintenance",
        },
        "comp3_maint": {
            "value_type": "BOOL",
            "description": "Whether component 3 was replaced during maintenance",
        },
        "comp4_maint": {
            "value_type": "BOOL",
            "description": "Whether component 4 was replaced during maintenance",
        },
    }

    create_entity_type_and_features(fs_name, fs_id, features)
    batch_ingest_features(fs_name, fs_id, fs_index, time_index, bq_source_uri)

    # vm_telemetry table

    fs_name = "vm_telemetry"
    bq_source_uri = BQ_PATTERN.format(
        project=project_id, dataset=dataset_name, table=fs_name
    )
    features= {
        "voltage": {
            "value_type": "DOUBLE",
            "description": "Voltage reading of machine during telemetry measurement (by hour)",
        },
        "rotation": {
            "value_type": "DOUBLE",
            "description": "Rotation reading of machine during telemetry measurement (by hour)",
        },
        "pressure": {
            "value_type": "DOUBLE",
            "description": "Pressure reading of machine during telemetry measurement (by hour)",
        },
        "vibration": {
            "value_type": "DOUBLE",
            "description": "Vibration reading of machine during telemetry measurement (by hour)",
        },
        "voltage_24hr_avg": {
            "value_type": "DOUBLE",
            "description": "Average voltage reading of machine during the last 24 hours",
        },
        "voltage_24hr_max": {
            "value_type": "DOUBLE",
            "description": "Max voltage reading of machine during the last 24 hours",
        },
        "voltage_24hr_min": {
            "value_type": "DOUBLE",
            "description": "Min voltage reading of machine during the last 24 hours",
        },
        "rotation_24hr_avg": {
            "value_type": "DOUBLE",
            "description": "Average rotation reading of machine during the last 24 hours",
        },
        "rotation_24hr_max": {
            "value_type": "DOUBLE",
            "description": "Max rotation reading of machine during the last 24 hours",
        },
        "rotation_24hr_min": {
            "value_type": "DOUBLE",
            "description": "Min rotation reading of machine during the last 24 hours",
        },
        "pressure_24hr_avg": {
            "value_type": "DOUBLE",
            "description": "Average pressure reading of machine during the last 24 hours",
        },
        "pressure_24hr_max": {
            "value_type": "DOUBLE",
            "description": "Max pressure reading of machine during the last 24 hours",
        },
        "pressure_24hr_min": {
            "value_type": "DOUBLE",
            "description": "Min pressure reading of machine during the last 24 hours",
        },
        "vibration_24hr_avg": {
            "value_type": "DOUBLE",
            "description": "Average vibration reading of machine during the last 24 hours",
        },
        "vibration_24hr_max": {
            "value_type": "DOUBLE",
            "description": "Max vibration reading of machine during the last 24 hours",
        },
        "vibration_24hr_min": {
            "value_type": "DOUBLE",
            "description": "Min vibration reading of machine during the last 24 hours",
        },
        "voltage_7day_avg": {
            "value_type": "DOUBLE",
            "description": "Average voltage reading of machine during the last 7 days",
        },
        "voltage_7day_max": {
            "value_type": "DOUBLE",
            "description": "Max voltage reading of machine during the last 7 days",
        },
        "voltage_7day_min": {
            "value_type": "DOUBLE",
            "description": "Min voltage reading of machine during the last 7 days",
        },
        "rotation_7day_avg": {
            "value_type": "DOUBLE",
            "description": "Average rotation reading of machine during the last 7 days",
        },
        "rotation_7day_max": {
            "value_type": "DOUBLE",
            "description": "Max rotation reading of machine during the last 7 days",
        },
        "rotation_7day_min": {
            "value_type": "DOUBLE",
            "description": "Min rotation reading of machine during the last 7 days",
        },
        "pressure_7day_avg": {
            "value_type": "DOUBLE",
            "description": "Average pressure reading of machine during the last 7 days",
        },
        "pressure_7day_max": {
            "value_type": "DOUBLE",
            "description": "Max pressure reading of machine during the last 7 days",
        },
        "pressure_7day_min": {
            "value_type": "DOUBLE",
            "description": "Min pressure reading of machine during the last 7 days",
        },
        "vibration_7day_avg": {
            "value_type": "DOUBLE",
            "description": "Average vibration reading of machine during the last 7 days",
        },
        "vibration_7day_max": {
            "value_type": "DOUBLE",
            "description": "Max vibration reading of machine during the last 7 days",
        },
        "vibration_7day_min": {
            "value_type": "DOUBLE",
            "description": "Min vibration reading of machine during the last 7 days",
        },
        "voltage_diff_from_24hr_avg": {
            "value_type": "DOUBLE",
            "description": "Current voltage difference from rolling average of last 24 hours (by hour)",
        },
        "rotation_diff_from_24hr_avg": {
            "value_type": "DOUBLE",
            "description": "Current rotation difference from rolling average of last 24 hours (by hour)",
        },
        "pressure_diff_from_24hr_avg": {
            "value_type": "DOUBLE",
            "description": "Current pressure difference from rolling average of last 24 hours (by hour)",
        },
        "vibration_diff_from_24hr_avg": {
            "value_type": "DOUBLE",
            "description": "Current vibration difference from rolling average of last 24 hours (by hour)",
        },
    }

    create_entity_type_and_features(fs_name, fs_id, features)
    batch_ingest_features(fs_name, fs_id, fs_index, time_index, bq_source_uri)

    # Serving features to final training dataset stored on BigQuery with model spine and features from feature store as input

    # Output table. Note: This table may need to be dropped and recreated if updated as BatchReadFeatureValues API may not be able to overwrite existing tables.
    FINAL_TRAINING_TABLE = "training_data"
    
    destination_table_uri = BQ_PATTERN.format(
        project=project_id, dataset=dataset_name, table=FINAL_TRAINING_TABLE
    )

    # Specifying where the model spine can be found
    model_spine = "vm_next_failure_" + os.getenv('PREDICTION_PERIOD')
    pass_through_fields = ["machine_id", "failure_in_" + os.getenv('PREDICTION_PERIOD'), "split"]
    read_instance_uri = BQ_PATTERN.format(
        project=project_id, dataset=dataset_name, table=model_spine
    )

    # Pulling all features from previously created entity types / feature sets

    serving_feature_ids = {
        "vm_errors": ['*'],
        "vm_failures": ['*'],
        "vm_hourly_status": ['*'],
        "vm_machines": ['*'],
        "vm_maintenance": ['*'],
        "vm_telemetry": ['*'],     
    }

    batch_serve_features_to_bq(fs_id, destination_table_uri, read_instance_uri, serving_feature_ids, pass_through_fields)
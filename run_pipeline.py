import os
import yaml
from typing import NamedTuple

from google.cloud import aiplatform
from google.cloud.aiplatform.pipeline_jobs import PipelineJob
from google_cloud_pipeline_components.aiplatform import (
        AutoMLTabularTrainingJobRunOp, EndpointCreateOp, ModelDeployOp,
        TabularDatasetCreateOp)

import kfp
from kfp import components
from kfp.v2 import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact, ClassificationMetrics, Input, Metrics,
                        Output, component)

def run_pipeline():

    # Instantiate env variables  
    input_gcs_path = os.getenv('INPUT_GCS_PATH')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset_name = os.getenv('DATASET_NAME')
    dataset_location = os.getenv('DATASET_LOCATION')
    credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    region = os.getenv('VERTEX_REGION')
    docker_image = os.getenv('VERTEX_DBT_DOCKER_IMAGE')
    prediction_period = os.getenv('PREDICTION_PERIOD')
    
    # Creating bucket and pipeline root where pipeline and model assets will be stored
    bucket_uri = "gs://" + os.getenv('BUCKET_NAME')
    pipeline_root = f"{bucket_uri}/pipeline_root"
    pipeline_root

    # Display name of your pipeline (different from pipeline name)
    pipeline_name = os.getenv('PIPELINE_NAME')
    pipeline_display_name = pipeline_name + "-display_name"

    # Other relevant pipeline variables
    vertex_dataset = os.getenv('VERTEX_DATASET_NAME')
    training_name = os.getenv('TRAINING_DISPLAY_NAME')
    model_name = os.getenv('MODEL_DISPLAY_NAME')
    endpoint_name = os.getenv('ENDPOINT_DISPLAY_NAME')
    machine_type = os.getenv('MACHINE_TYPE')

    # Retreive docker image name from build_image.sh and programmatically update component.yaml
    def update_docker_image_in_component_file(
        base_image: str,
        component_file: str = 'component.yaml',
    ):

        # Read and update input component.yaml file with new base image
        with open(component_file) as input:
            comp_file = yaml.safe_load(input)
            comp_file['implementation']['container']['image'] = base_image

        # Persist the changes to a temp yaml file first
        with open("/tmp/temp.yaml", "w") as output:
            yaml.dump(comp_file, output, default_flow_style=False, sort_keys=False)

        # Check that the new yaml file looks correct before renaming and overwriting current component.yaml file
        with open("/tmp/temp.yaml") as check:
            check_comp_file = yaml.safe_load(check)
        
        if check_comp_file['implementation']['container']['image'] == base_image:
            os.rename("/tmp/temp.yaml", component_file)

    # Defining custom model evaluation component
    @component(
        base_image="gcr.io/deeplearning-platform-release/tf2-cpu.2-6:latest",
        output_component_file="components/tabular_eval_component.yaml",
        packages_to_install=["google-cloud-aiplatform"],
    )
    def classification_model_eval_metrics(
        project: str,
        location: str,
        thresholds_dict_str: str,
        model: Input[Artifact],
        metrics: Output[Metrics],
        metricsc: Output[ClassificationMetrics],
    ) -> NamedTuple("Outputs", [("dep_decision", str)]):  # Return parameter.
        
        import json
        import logging

        from google.cloud import aiplatform

        aiplatform.init(project=project, location=location)
        
        # Fetch model eval info
        def get_eval_info(model):
            response = model.list_model_evaluations()
            metrics_list = []
            metrics_string_list = []
            for evaluation in response:
                evaluation = evaluation.to_dict()
                print("model_evaluation")
                print(" name:", evaluation["name"])
                print(" metrics_schema_uri:", evaluation["metricsSchemaUri"])
                metrics = evaluation["metrics"]
                for metric in metrics.keys():
                    logging.info("metric: %s, value: %s", metric, metrics[metric])
                metrics_str = json.dumps(metrics)
                metrics_list.append(metrics)
                metrics_string_list.append(metrics_str)

            return (
                evaluation["name"],
                metrics_list,
                metrics_string_list,
            )

        # Use the given metrics threshold(s) to determine whether the model is accurate enough to deploy.
        def classification_thresholds_check(metrics_dict, thresholds_dict):
            for k, v in thresholds_dict.items():
                logging.info("k {}, v {}".format(k, v))
                if k in ["auRoc", "auPrc"]:  # Higher is better
                    if metrics_dict[k] < v:  # If under threshold, don't deploy
                        logging.info("{} < {}; returning False".format(metrics_dict[k], v))
                        return False
            logging.info("threshold checks passed.")
            return True

        def log_metrics(metrics_list, metricsc):
            test_confusion_matrix = metrics_list[0]["confusionMatrix"]
            logging.info("rows: %s", test_confusion_matrix["rows"])

            # Log the ROC curve
            fpr = []
            tpr = []
            thresholds = []
            for item in metrics_list[0]["confidenceMetrics"]:
                fpr.append(item.get("falsePositiveRate", 0.0))
                tpr.append(item.get("recall", 0.0))
                thresholds.append(item.get("confidenceThreshold", 0.0))
            print(f"fpr: {fpr}")
            print(f"tpr: {tpr}")
            print(f"thresholds: {thresholds}")
            metricsc.log_roc_curve(fpr, tpr, thresholds)

            # Log the confusion matrix
            annotations = []
            for item in test_confusion_matrix["annotationSpecs"]:
                annotations.append(item["displayName"])
            logging.info("confusion matrix annotations: %s", annotations)
            metricsc.log_confusion_matrix(
                annotations,
                test_confusion_matrix["rows"],
            )

            # Log textual metrics info as well
            for metric in metrics_list[0].keys():
                if metric != "confidenceMetrics":
                    val_string = json.dumps(metrics_list[0][metric])
                    metrics.log_metric(metric, val_string)

        logging.getLogger().setLevel(logging.INFO)

        # Extract the model resource name from the input Model Artifact
        model_resource_path = model.metadata["resourceName"]
        logging.info("model path: %s", model_resource_path)

        # Get the trained model resource
        model = aiplatform.Model(model_resource_path)

        # Get model evaluation metrics from the the trained model
        eval_name, metrics_list, metrics_str_list = get_eval_info(model)
        logging.info("got evaluation name: %s", eval_name)
        logging.info("got metrics list: %s", metrics_list)
        log_metrics(metrics_list, metricsc)

        thresholds_dict = json.loads(thresholds_dict_str)
        deploy = classification_thresholds_check(metrics_list[0], thresholds_dict)
        if deploy:
            dep_decision = "true"
            print("Threshold has been met. Model is being deployed / updated.")
        else:
            dep_decision = "false"
            print("Threshold has not been met. Model is not being deployed / updated.")
        logging.info("deployment decision is %s", dep_decision)

        return (dep_decision,)

    # Pipeline definition
    @dsl.pipeline(name=pipeline_name, pipeline_root=pipeline_root)
    def pipeline(
        DATASET_DISPLAY_NAME: str,
        TRAINING_DISPLAY_NAME: str,
        MODEL_DISPLAY_NAME: str,
        ENDPOINT_DISPLAY_NAME: str,
        project: str,
        gcp_region: str,
        thresholds_dict_str: str,
        MACHINE_TYPE: str = "n1-standard-4",
    ):
        
        # Creating component ops for pipeline
        data_ingest_op = data_ingestion_component(input_gcs_path, project_id, dataset_name, dataset_location)
        dbt_op = dbt_component(project_id, dataset_name, credentials)
        feature_store_op = feature_store_component(project_id, dataset_name, region, prediction_period)

        
        BQ_PATTERN = "bq://{project}.{dataset}.{table}" # To help construct BQ URI for bq_source
        bq_source = BQ_PATTERN.format(
            project=project_id, dataset=dataset_name, table="training_data"
        )
        
        dataset_create_op = TabularDatasetCreateOp(
            project=project, display_name=DATASET_DISPLAY_NAME, bq_source=bq_source
        )
        
        training_op = AutoMLTabularTrainingJobRunOp(
            project=project,
            display_name=TRAINING_DISPLAY_NAME,
            optimization_prediction_type="classification",
            optimization_objective="minimize-log-loss",
            budget_milli_node_hours=1000,
            model_display_name=MODEL_DISPLAY_NAME,
            dataset=dataset_create_op.outputs["dataset"],
            target_column="failure_in_" + prediction_period,
            predefined_split_column_name="split",
        )
        

        eval_model_op = classification_model_eval_metrics(
            project,
            gcp_region,
            thresholds_dict_str,
            training_op.outputs["model"],
        )

        with dsl.Condition(
            eval_model_op.outputs["dep_decision"] == "true",
            name="deploy_decision",
        ):

            endpoint_op = EndpointCreateOp(
                project=project,
                location=gcp_region,
                display_name=ENDPOINT_DISPLAY_NAME,
            )

            ModelDeployOp(
                model=training_op.outputs["model"],
                endpoint=endpoint_op.outputs["endpoint"],
                dedicated_resources_min_replica_count=1,
                dedicated_resources_max_replica_count=1,
                dedicated_resources_machine_type=MACHINE_TYPE,
            )


        # Specifying order of pipeline components that don't have direct inputs / outputs
        dbt_op.after(data_ingest_op)
        feature_store_op.after(dbt_op)
        dataset_create_op.after(feature_store_op)


    # Compiles the pipeline defined in the previous function into a json file executable by Vertex AI Pipelines
    def compile():
        compiler.Compiler().compile(
            pipeline_func=pipeline, package_path='vertex_pipeline.json', type_check=False
        )


    # Triggers the pipeline, caching is disabled as this causes successive dbt pipeline steps to be skipped
    def trigger_pipeline():
        pl = PipelineJob(
            display_name=pipeline_display_name,
            enable_caching=False,
            template_path="vertex_pipeline.json",
            pipeline_root=pipeline_root,
            parameter_values={
                "project": project_id,
                "gcp_region": region,
                "thresholds_dict_str": '{"auRoc": 0.95}',
                "DATASET_DISPLAY_NAME": vertex_dataset,
                "TRAINING_DISPLAY_NAME": training_name,
                "MODEL_DISPLAY_NAME": model_name,
                "ENDPOINT_DISPLAY_NAME": endpoint_name,
                "MACHINE_TYPE": machine_type,
            },
        )

        pl.run(sync=True)


    # Initializing client and create feature store
    aiplatform.init(project=project_id, location=region)

    # Setting custom component paths (defined as YAML files)
    script_dir = os.path.dirname(__file__)
    data_ingestion_path = os.path.join(script_dir, 'components/data_ingestion_component.yaml')
    dbt_component_path = os.path.join(script_dir, 'components/dbt_component.yaml')
    feature_store_path = os.path.join(script_dir, 'components/feature_store_component.yaml')

    # Update custom component files with new docker image built from build_image.sh
    update_docker_image_in_component_file(docker_image, data_ingestion_path)
    update_docker_image_in_component_file(docker_image, dbt_component_path)
    update_docker_image_in_component_file(docker_image, feature_store_path)

    # Loads the custom component files as separate components for pipeline (initial data ingestion, dbt run, and feature store creation / serving)
    data_ingestion_component = components.load_component_from_file(data_ingestion_path)
    dbt_component = components.load_component_from_file(dbt_component_path)
    feature_store_component = components.load_component_from_file(feature_store_path)

    # Compile the pipeline components and trigger the run
    compile()
    trigger_pipeline()

if __name__ == '__main__':
    
    run_pipeline()
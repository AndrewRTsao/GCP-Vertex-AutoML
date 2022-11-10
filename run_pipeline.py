import os
import yaml

from google.cloud import aiplatform
from google.cloud.aiplatform.pipeline_jobs import PipelineJob
from kfp import components, dsl
from kfp.v2 import compiler


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


# Pipeline definition
@dsl.pipeline(name='vertex-dbt-pipeline')
def pipeline():
    
    # Creating component ops
    data_ingest_op = data_ingestion_component(input_gcs_path, project_id, dataset_name, dataset_location)
    dbt_op = dbt_component(project_id, dataset_name, credentials)
    fs_op = feature_store_component(project_id, dataset_name, region, prediction_period)

    # Specifying order of pipeline components that don't have direct inputs / outputs
    dbt_op.after(data_ingest_op)
    fs_op.after(dbt_op)


# Compiles the pipeline defined in the previous function into a json file executable by Vertex AI Pipelines
def compile():
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path='pipeline.json', type_check=False
    )


# Triggers the pipeline, caching is disabled as this causes successive dbt pipeline steps to be skipped
def trigger_pipeline():
    pl = PipelineJob(
      display_name="FirstPipeline",
      enable_caching=False,
      template_path="./pipeline.json",
      pipeline_root="gs://vertex-delivery-example"
    )

    pl.run(sync=True)


if __name__ == '__main__':
    
    # Instantiate relevant env variables as local variables  
    input_gcs_path = os.getenv('INPUT_GCS_PATH')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset_name = os.getenv('DATASET_NAME')
    dataset_location = os.getenv('DATASET_LOCATION')
    credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    region = os.getenv('VERTEX_REGION')
    docker_image = os.getenv('VERTEX_DBT_DOCKER_IMAGE')
    prediction_period = os.getenv('PREDICTION_PERIOD')

    # Initializing client and create feature store
    aiplatform.init(project=project_id, location=region)

    # Setting component paths
    script_dir = os.path.dirname(__file__)
    data_ingestion_path = os.path.join(script_dir, 'components/data_ingestion.yaml')
    dbt_component_path = os.path.join(script_dir, 'components/dbt_component.yaml')
    feature_store_path = os.path.join(script_dir, 'components/feature_store.yaml')

    # Update component files with new docker image built from build_image.sh
    update_docker_image_in_component_file(docker_image, data_ingestion_path)
    update_docker_image_in_component_file(docker_image, dbt_component_path)
    update_docker_image_in_component_file(docker_image, feature_store_path)

    # Loads the component files as separate components for pipeline
    data_ingestion_component = components.load_component_from_file(data_ingestion_path)
    dbt_component = components.load_component_from_file(dbt_component_path)
    feature_store_component = components.load_component_from_file(feature_store_path)

    compile()
    trigger_pipeline()
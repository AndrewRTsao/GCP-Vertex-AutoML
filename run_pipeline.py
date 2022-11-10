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
    with open("temp.yaml", "w") as output:
        yaml.dump(comp_file, output, default_flow_style=False, sort_keys=False)

    # Check that the new yaml file looks correct before renaming and overwriting current component.yaml file
    with open("temp.yaml") as check:
        check_comp_file = yaml.safe_load(check)
    
    if check_comp_file['implementation']['container']['image'] == base_image:
        os.rename("temp.yaml", "component.yaml")


# Pipeline definition occurs here
@dsl.pipeline(name='my-first-dbt-pipeline')
def dbt_pipeline():
    dbt = dbt_component(project_id, dataset_name, credentials)


# Compiles the pipeline defined in the previous function into a json file executable by Vertex AI Pipelines
def compile():
    compiler.Compiler().compile(
        pipeline_func=dbt_pipeline, package_path='pipeline.json', type_check=False
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
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset_name = os.getenv('DATASET_NAME')
    credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    region = os.getenv('VERTEX_REGION')
    docker_image = os.getenv('VERTEX_DBT_DOCKER_IMAGE')

    # Initializing client and create feature store
    aiplatform.init(project=project_id, location=region)

    # Update component file with new docker image built from build_image.sh
    update_docker_image_in_component_file(docker_image)

    # Loads the component file as a component
    dbt_component = components.load_component_from_file('./component.yaml')

    compile()
    trigger_pipeline()
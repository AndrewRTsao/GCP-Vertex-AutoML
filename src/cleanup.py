import os
from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import storage

def cleanup():

    def delete_dataset(
        dataset_id: str,
        delete_contents: bool = True,
        not_found_ok: bool = True,
    ):
        try: 
            client = bigquery.Client()

            client.delete_dataset(
                dataset_id, delete_contents=delete_contents, not_found_ok=not_found_ok
            )
            print("Deleted dataset '{}'.".format(dataset_id))

        except Exception as error:
            print(error)

    def delete_featurestore(
        project: str,
        location: str,
        featurestore_name: str,
        sync: bool = True,
        force: bool = True,
    ):

        try: 
            aiplatform.init(project=project, location=location)
            
            fs = aiplatform.featurestore.Featurestore(featurestore_name=featurestore_name)
            fs.delete(sync=sync, force=force)

        except Exception as error:
            print(error)

    def delete_pipeline_job(
        project: str,
        location: str,
        pipeline_display_name: str,
        sync: bool = True,
    ):
        
        try: 
            aiplatform.init(project=project, location=location)
            
            # Retreive pipeline jobs that match name
            pipeline_jobs = aiplatform.PipelineJob.list(
                filter=f"display_name={pipeline_display_name}", order_by="create_time"
            )

            if len(pipeline_jobs) > 0:
                pipeline_job = pipeline_jobs[0]
                pipeline_job.delete(sync=sync)

        except Exception as error:
            print(error)
    
    def delete_endpoint(
        project: str,
        location: str,
        endpoint_name: str,
        force: bool = True,
    ):

        try: 
            aiplatform.init(project=project, location=location)

            # Retrieve endpoints that match name
            endpoints = aiplatform.Endpoint.list(
                filter=f"display_name={endpoint_name}", order_by="create_time"
            )

            if len(endpoints) > 0:
                endpoint = endpoints[0]
                endpoint.undeploy_all() # undeploy models before deleting
                endpoint.delete(force=force)
        
        except Exception as error:
            print(error)
    
    def delete_model(
        project: str,
        location: str,
        model_name: str,
    ):

        try: 
            aiplatform.init(project=project, location=location)

            # Retrieve models that match name
            models = aiplatform.Model.list(
                filter=f"display_name={model_name}", order_by="create_time"
            )

            if len(models) > 0:
                model = models[0]
                model.delete()

        except Exception as error:
            print(error)

    def delete_vertex_dataset(
        project: str,
        location: str,
        vertex_dataset: str,
    ):

        try: 
            aiplatform.init(project=project, location=location)

            # Retrieve Vertex datasets that match name
            datasets = aiplatform.TabularDataset.list(
                filter=f"display_name={vertex_dataset}", order_by="create_time"
            )

            if len(datasets) > 0:
                dataset = datasets[0]
                dataset.delete()

        except Exception as error:
            print(error)
        
    def delete_GCS_bucket(
        bucket_name: str,
        force_delete: bool = False
    ):

        try: 
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(bucket_name)
           
            bucket.delete(force=force_delete)
            print(f"Bucket {bucket.name} deleted")
        
        except Exception as error:
            print(error)

    # Initialize variables
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    region = os.getenv('VERTEX_REGION')
    dataset_id = os.getenv('DATASET_NAME')
    featurestore_id = os.getenv('DATASET_NAME') + "_fs"
    pipeline_display_name = os.getenv('PIPELINE_NAME') + "-display_name"
    endpoint_name = os.getenv('ENDPOINT_DISPLAY_NAME')
    model_name = os.getenv('MODEL_DISPLAY_NAME')
    vertex_dataset = os.getenv('VERTEX_DATASET_NAME')
    bucket_name = os.getenv('BUCKET_NAME')
    force_delete = os.getenv('FORCE_DELETE_BUCKET')


    # Delete stuff
    delete_dataset(dataset_id)
    delete_featurestore(project_id, region, featurestore_id)
    delete_pipeline_job(project_id, region, pipeline_display_name)
    delete_endpoint(project_id, region, endpoint_name)
    delete_model(project_id, region, model_name)
    delete_vertex_dataset(project_id, region, vertex_dataset)
    delete_GCS_bucket(bucket_name, force_delete)

if __name__ == "__main__":

    cleanup()

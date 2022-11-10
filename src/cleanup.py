import os
from google.cloud import aiplatform
from google.cloud import bigquery

def cleanup():

    def delete_featurestore(
        project: str,
        location: str,
        featurestore_name: str,
        sync: bool = True,
        force: bool = True,
    ):

        aiplatform.init(project=project, location=location)

        fs = aiplatform.featurestore.Featurestore(featurestore_name=featurestore_name)
        fs.delete(sync=sync, force=force)

    def delete_dataset(dataset_id):
        client = bigquery.Client()
        
        client.delete_dataset(
            dataset_id, delete_contents=True, not_found_ok=True
        )
        print("Deleted dataset '{}'.".format(dataset_id))

    # Initialize variables
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    region = os.getenv('VERTEX_REGION')
    dataset_id = os.getenv('DATASET_NAME')
    featurestore_id = os.getenv('DATASET_NAME') + "_fs"

    delete_featurestore(project_id, region, featurestore_id)
    delete_dataset(dataset_id)

if __name__ == "__main__":

    cleanup()

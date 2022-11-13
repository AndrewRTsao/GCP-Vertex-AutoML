# Initial data related variables
export LOCAL_PROJECT="" # Fully qualified path of your locally cloned GCP-Vertex project
export SERVICE_KEY="" # File name of your service account API key in GCP (include the .json)
export GOOGLE_APPLICATION_CREDENTIALS="${LOCAL_PROJECT}/data/${SERVICE_KEY}" # WARNING: DON'T UPDATE THIS ENV VAR
export GOOGLE_CLOUD_PROJECT="" # Set to the default GCP project that you wish to use
export INPUT_GCS_PATH="" # Path to GCS bucket where you uploaded Kaggle files
export DATASET_NAME="" # What you want to name your dataset in BigQuery and your feature store in Vertex (should be lowercase)
export DATASET_LOCATION="" # Location of your BigQuery dataset (should match or be accessible by the Vertex region)

# Intermediary assets
export PREDICTION_PERIOD="" # Prediction interval that you would like to use for your predictive maintenance model, select either "1day", "7day", or "30day"
export DOCKER_REPO="" # Name of your docker repo that you'd like to use / create for your KFP component base image (follow proper syntax e.g. lowercase letters, numbers, and hyphens)

# Vertex related env variables
export VERTEX_REGION="" # Location where you wish to perform your Vertex AI operations (e.g. us-central1, europe-west4, or asia-northeast1). Check: https://cloud.google.com/vertex-ai/docs/general/locations
export BUCKET_NAME="" # Staging bucket where all data associated with model resources and pipeline will be stored (will be created under your GCP project)
export PIPELINE_NAME="" # What you would like to name your pipeline
export VERTEX_DATASET_NAME="vertex-${DATASET_NAME}" # (Optional) Change if you wish to update the display name for your Vertex AI managed dataset
export TRAINING_DISPLAY_NAME="" # Display name for the AutoML training job
export MODEL_DISPLAY_NAME="" # Display name for the Vertex AI Model generated as a result of the training job
export ENDPOINT_DISPLAY_NAME="" # Display name for the Vertex AI Endpoint where the model is deployed
export MACHINE_TYPE="n1-standard-4" # Machine type for the serving container (defaults to "n1-standard-4")

# Cleanup
export FORCE_DELETE_BUCKET="False" # If set to True, will empty the bucket before force deleting the staging bucket when running cleanup.py script (i.e. delete all resulting Vertex and model artifacts). Otherwise, bucket will be preserved when cleaning up other resources.

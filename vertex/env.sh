export GOOGLE_APPLICATION_CREDENTIALS="/Users/andrew/Continual/continual-dev-aef3ddfaa866.json"  # path to service account API key
export GOOGLE_CLOUD_PROJECT="continual-dev" # Set to the default GCP project that you wish to use
export DATASET_NAME="azure_vm" # What you want to name your dataset in BigQuery and your resulting feature store in Vertex (should be lowercase)
export DATASET_LOCATION="US" # Location of your BigQuery dataset (should match or be accessible by the Vertex region)
export VERTEX_REGION="us-central1" # Location where you wish to perform your Vertex AI operations. Check: https://cloud.google.com/vertex-ai/docs/general/locations
export GCS_PATH="gs://vertex-delivery-example/azure_vm/" # Path to parent folder in GCS bucket where you uploaded Kaggle files
export PREDICTION_PERIOD="7day" # Prediction interval that you would like to use for your predictive maintenance model, select either "1day", "7day", or "30day"

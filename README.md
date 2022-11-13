# Getting Started

1. Download the datasets from the Microsoft Azure Predictive Maintenance [Kaggle project](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) either directly or by using the Kaggle CLI. Then, upload it to a GCS bucket (in the correct region / project).

```sh 
kaggle datasets download -d arnabbiswas1/microsoft-azure-predictive-maintenance
```

2. Create a [service account key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#iam-service-account-keys-create-console) and copy the resulting json file into the *./data* subdirectory of this project.

3. Fill out the environment variables in **env.sh** and source the file. 

```sh 
source env.sh
```

4. Run the **build_image.sh** script to build the component container image and push it to the Artifact Registry (make sure your Docker daemon is running locally).

```sh
. ./build_image.sh
```

5. Run the **run_pipeline.py** script to trigger the prepackaged Vertex AI pipeline (ingests the data into BigQuery from step 1, dbt run, creates and pushes features into Vertex feature store, trains the model using Vertex AutoML, evaluates the model, and then finally deploys the model to a Vertex endpoint if a certain threshold has been met).

```sh
python run_pipeline.py
```

*NOTE: Pipeline will take approximately 4 hours to complete*

6. (Optional) If you would like, run the **cleanup.py** script once you're done and if don't need the underlying BigQuery dataset, feature store, model, and/or other Vertex AI resources anymore.

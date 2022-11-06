# Getting Started

1. Download the datasets from the Microsoft Azure Predictive Maintenance [Kaggle project](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) either directly or by using the Kaggle CLI:

```sh 
kaggle datasets download -d arnabbiswas1/microsoft-azure-predictive-maintenance
```

Afterwards, upload it to a GCS bucket (in the correct region).

2. Fill out the environment variables in ./vertex/env.sh and run

```sh 
source vertex/env.sh
```

3. Run the staging data_ingestion.py script to load your data from GCS into BigQuery

4. dbt
5. feature store
6. training pipelines
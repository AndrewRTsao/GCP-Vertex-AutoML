Predictive maintenance use case built for Azure VMs using BigQuery. Feature engineering is performed in this dbt project. 

### Using the project

To get started with this dbt project, update the dbt_project.yml file so that it uses a valid profile pointing to an existing BigQuery datasource. Make sure that your dbt project has been configured to use the dbt-bigquery adapter.

Then, run the following commands (this project utilizes [dbt-utils](https://hub.getdbt.com/dbt-labs/dbt_utils/latest/)):

```sh
dbt deps
dbt run
dbt test
```

If you want to check out the project documentation: 

```sh
dbt docs generate
dbt docs serve
```

### Resources:

This dbt project was built using the following [Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) datasets found through Kaggle. More examples (soon to come!) can be found by checking out [Continual's documentation](https://docs.continual.ai/).

select
  machine_id,
  model,
  age

from {{ ref('stg_machines') }}
{{
 config(
   meta = {
     "continual": {
       "type": "FeatureSet",
       "name": "vm_machine_info",
       "entity": "azure_vm",
       "index": "machine_id"
     }
   }
 ) 
}}

select
  machine_id,
  '2015-01-01 00:00:00.000' as ts,
  model,
  age

from {{ ref('stg_machines') }}
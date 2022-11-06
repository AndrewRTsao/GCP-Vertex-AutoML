{{
  config(
    materialized = 'table',
    meta = {
     "continual": {
       "type": "FeatureSet",
       "name": "vm_machine_hourly_status",
       "entity": "azure_vm",
       "index": "machine_id",
       "time_index": "ts"
     }
   }
  )
}}

with machine_status as (

    select * from {{ ref('int_vm_status') }}

),

final as (

    select
      machine_id,
      ts,

      date_diff(ts, most_recent_maint, hour) as hours_since_last_maint,
      date_diff(ts, most_recent_comp1_maint, hour) as hours_since_last_comp1_maint,
      date_diff(ts, most_recent_comp2_maint, hour) as hours_since_last_comp2_maint,
      date_diff(ts, most_recent_comp3_maint, hour) as hours_since_last_comp3_maint,
      date_diff(ts, most_recent_comp4_maint, hour) as hours_since_last_comp4_maint,

      date_diff(ts, most_recent_failure, hour) as hours_since_last_failure,
      date_diff(ts, most_recent_comp1_failure, hour) as hours_since_last_comp1_failure,
      date_diff(ts, most_recent_comp2_failure, hour) as hours_since_last_comp2_failure,
      date_diff(ts, most_recent_comp3_failure, hour) as hours_since_last_comp3_failure,
      date_diff(ts, most_recent_comp4_failure, hour) as hours_since_last_comp4_failure,

      date_diff(ts, most_recent_error, hour) as hours_since_last_error,
      date_diff(ts, most_recent_error1, hour) as hours_since_last_error1,
      date_diff(ts, most_recent_error2, hour) as hours_since_last_error2,
      date_diff(ts, most_recent_error3, hour) as hours_since_last_error3,
      date_diff(ts, most_recent_error4, hour) as hours_since_last_error4,
      date_diff(ts, most_recent_error5, hour) as hours_since_last_error5

    from machine_status
    order by machine_id asc, ts desc

)

select * from final
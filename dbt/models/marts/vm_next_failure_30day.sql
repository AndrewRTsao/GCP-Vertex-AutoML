with failures as (

    select * from {{ ref('vm_next_failure_base') }}

),

final as (

    select
      machine_id,
      machine_id as vm_errors,
      machine_id as vm_failures,
      machine_id as vm_hourly_status,
      machine_id as vm_machines,
      machine_id as vm_maintenance,
      machine_id as vm_telemetry,
      ts as timestamp,
      failure_in_30day,
      split

    from failures

)

select * from final
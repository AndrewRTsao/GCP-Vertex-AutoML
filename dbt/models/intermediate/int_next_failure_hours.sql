with failures as (

    select * from {{ ref('vm_failures') }}

),

telemetry as (

    select * from {{ ref('vm_telemetry') }}

),

joined as (

  select 
    machine_id,
    ts,

    (select min(ts) from failures where machine_id = telemetry.machine_id and ts >= telemetry.ts) as next_failure,
    (select min(ts) from failures where machine_id = telemetry.machine_id and ts >= telemetry.ts and comp1_failure is true) as next_comp1_failure,
    (select min(ts) from failures where machine_id = telemetry.machine_id and ts >= telemetry.ts and comp2_failure is true) as next_comp2_failure,
    (select min(ts) from failures where machine_id = telemetry.machine_id and ts >= telemetry.ts and comp3_failure is true) as next_comp3_failure,
    (select min(ts) from failures where machine_id = telemetry.machine_id and ts >= telemetry.ts and comp4_failure is true) as next_comp4_failure

  from telemetry

),

final as (

  select
    machine_id,
    ts,
    date_diff(next_failure, ts, hour) as hours_until_next_failure,
    date_diff(next_comp1_failure, ts, hour) as hours_until_next_comp1_failure,
    date_diff(next_comp2_failure, ts, hour) as hours_until_next_comp2_failure,
    date_diff(next_comp3_failure, ts, hour) as hours_until_next_comp3_failure,
    date_diff(next_comp4_failure, ts, hour) as hours_until_next_comp4_failure

  from joined

)

select * from final
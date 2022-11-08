with telemetry as (

    select * from {{ ref('stg_telemetry') }}

),

time_aggregated_int as (

    select 
      ts,
      machine_id,
      voltage,
      rotation,
      pressure,
      vibration,

      avg(voltage) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as voltage_24hr_avg,
      max(voltage) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as voltage_24hr_max,
      min(voltage) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as voltage_24hr_min,

      avg(rotation) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as rotation_24hr_avg,
      max(rotation) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as rotation_24hr_max,
      min(rotation) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as rotation_24hr_min,

      avg(pressure) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as pressure_24hr_avg,
      max(pressure) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as pressure_24hr_max,
      min(pressure) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as pressure_24hr_min,

      avg(vibration) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as vibration_24hr_avg,
      max(vibration) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as vibration_24hr_max,
      min(vibration) over (partition by machine_id order by ts desc rows between 1 following and 24 following) as vibration_24hr_min,

      avg(voltage) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as voltage_7day_avg,
      max(voltage) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as voltage_7day_max,
      min(voltage) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as voltage_7day_min,

      avg(rotation) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as rotation_7day_avg,
      max(rotation) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as rotation_7day_max,
      min(rotation) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as rotation_7day_min,

      avg(pressure) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as pressure_7day_avg,
      max(pressure) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as pressure_7day_max,
      min(pressure) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as pressure_7day_min,

      avg(vibration) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as vibration_7day_avg,
      max(vibration) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as vibration_7day_max,
      min(vibration) over (partition by machine_id order by ts desc rows between 1 following and 168 following) as vibration_7day_min

    from telemetry

),

time_aggregated_final as (

  select
    *,
    voltage - voltage_24hr_avg as voltage_diff_from_24hr_avg,
    rotation - rotation_24hr_avg as rotation_diff_from_24hr_avg,
    pressure - pressure_24hr_avg as pressure_diff_from_24hr_avg,
    vibration - vibration_24hr_avg as vibration_diff_from_24hr_avg

    from time_aggregated_int
)

select * from time_aggregated_final
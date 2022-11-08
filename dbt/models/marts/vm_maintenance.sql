{%- set components = dbt_utils.get_column_values(table = ref('stg_maintenance'), column='comp') -%}

with maintenance as (

    select * from {{ ref('stg_maintenance') }}

),

pivoted as (

    select
      ts,
      machine_id,

      {%- for component in components %}
      cast(count(case when comp = '{{ component }}' then 1 end) as bool) as {{ component }}_maint
      {{- "," if not loop.last -}}
      {% endfor %}
      
    from maintenance
    group by 1,2

)

select * from pivoted
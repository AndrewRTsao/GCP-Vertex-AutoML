{%- set components = dbt_utils.get_column_values(table = ref('stg_failures'), column='failure') -%}

with failures as (

    select * from {{ ref('stg_failures') }}

),

pivoted as (

    select
      ts,
      machine_id,

      {%- for component in components %}
      cast(count(case when failure = '{{ component }}' then 1 end) as bool) as {{ component }}_failure
      {{- "," if not loop.last -}}
      {% endfor %}
      
    from failures
    group by 1,2

)

select * from pivoted
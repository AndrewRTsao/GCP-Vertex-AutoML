{%- set errors = dbt_utils.get_column_values(table = ref('stg_errors'), column='error_id') -%}

with errors as (

    select * from {{ ref('stg_errors') }}

),

pivoted as (

    select
      ts,
      machine_id,

      {%- for error in errors %}
      cast(count(case when error_id = '{{ error }}' then 1 end) as bool) as {{ error }}
      {{- "," if not loop.last -}}
      {% endfor %}
      
    from errors
    group by 1,2
    order by machine_id asc, ts desc

)

select * from pivoted
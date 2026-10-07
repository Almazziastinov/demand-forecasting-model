"""Read-only verification of two-day metadata and checkout discount clusters."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_historical_panel_from_clickhouse_via_vm import load_env  # noqa: E402


VM_ENV = ROOT / ".codex/prod_vm.env"


REMOTE = r'''
from scripts.export_clickhouse_checks import create_client
c = create_client('.env')
print('CANDIDATE_COLUMNS')
print(c.query_df("""
select database, table, name, type
from system.columns
where database not in ('system', 'INFORMATION_SCHEMA', 'information_schema')
  and multiSearchAnyCaseInsensitiveUTF8(
      concat(table, ' ', name),
      ['вчера', 'yesterday', 'discount', 'скид', 'остат', 'old_', 'age',
       'sale_type', 'sales_type', 'price_type', 'promo', 'акци']
  )
order by database, table, position
""").to_csv(index=False))
print('META_1071')
print(c.query_df("""
select product_id, any(product_name) product_name,
       max(toUInt8(is_two_day)) is_two_day,
       max(toUInt8(is_on_demand)) is_on_demand,
       max(toUInt8(is_active)) is_active
from baking_sku_meta where product_id = '1071' group by product_id
""").to_csv(index=False))
print('TWO_DAY_ACTIVE')
print(c.query_df("""
select product_id, any(product_name) product_name
from baking_sku_meta where is_active=1 and is_two_day=1
group by product_id order by product_id
""").to_csv(index=False))
print('PRICE_CLUSTERS_1071')
print(c.query_df("""
select toStartOfMonth(check_date) month,
       round(toFloat64(line_amount)/nullIf(toFloat64(quantity),0), 2) unit_price,
       sum(toFloat64(quantity)) qty, count() lines
from (
  select distinct check_datetime, check_date, bakery_id, product_id, quantity, line_amount
  from Svezhar.fct_check_lines
  where hex(cash_event_type)='D09FD180D0BED0B4D0B0D0B6D0B0'
    and product_id='1071'
    and (check_date between '2026-01-01' and '2026-01-31'
         or check_date between '2026-08-01' and '2026-08-31')
    and toFloat64(quantity)>0
)
group by month, unit_price order by month, qty desc
limit 30 by month
""").to_csv(index=False))
'''


def main() -> None:
    config = load_env(VM_ENV)
    payload = base64.b64encode(REMOTE.encode()).decode()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        config["PROD_VM_HOST"],
        username=config["PROD_VM_USER"],
        password=config["PROD_VM_PASSWORD"],
        timeout=60,
        banner_timeout=60,
        auth_timeout=60,
    )
    command = (
        "cd /opt/demand-forecasting-model && .venv/bin/python -c "
        f'"import base64;exec(base64.b64decode(\'{payload}\'))"'
    )
    _, stdout, stderr = ssh.exec_command(command, timeout=300)
    print(stdout.read().decode("utf-8", "replace"))
    error = stderr.read().decode("utf-8", "replace")
    ssh.close()
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()

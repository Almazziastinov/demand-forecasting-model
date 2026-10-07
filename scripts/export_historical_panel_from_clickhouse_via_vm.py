"""Read-only monthly SKU-day export from ClickHouse through the production VM."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import paramiko


ROOT = Path(__file__).resolve().parents[1]
VM_ENV = ROOT / ".codex/prod_vm.env"
PILOT = ROOT / ".codex_tmp/pilot_management_sep01_09/detail.csv"
OUTPUT = ROOT / ".codex_tmp/historical_clickhouse_panel"
REMOTE_DIR = "/tmp/codex_historical_clickhouse_panel_20260910"


def load_env(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw and not raw.lstrip().startswith("#"):
            key, value = raw.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def remote_program(ids: list[int]) -> str:
    return f'''from pathlib import Path
import pandas as pd
from scripts.export_clickhouse_checks import create_client

c = create_client('.env')
out = Path({REMOTE_DIR!r})
out.mkdir(parents=True, exist_ok=True)
ids = {ids!r}
id_sql = ','.join(map(str, ids))

def query(sql):
    return c.query_df(sql)

for start in pd.date_range('2026-01-01', '2026-08-01', freq='MS'):
    end = start + pd.offsets.MonthEnd(0)
    lo, hi = str(start.date()), str(end.date())
    sales = query(f"""
        select check_date date, toInt64OrZero(bakery_id) bakery_id,
               toInt64OrZero(product_id) product_id,
               sum(toFloat64(quantity)) observed_sales_qty,
               sum(toFloat64(line_amount)) observed_sales_amount,
               min(check_datetime) first_sale_time,
               max(check_datetime) last_sale_time,
               sum(toFloat64(line_amount))/nullIf(sum(toFloat64(quantity)),0) avg_sales_price
        from (select distinct check_datetime,check_date,bakery_id,product_id,
              quantity,line_amount from Svezhar.fct_check_lines
              where hex(cash_event_type)='D09FD180D0BED0B4D0B0D0B6D0B0'
                and check_date between '{{lo}}' and '{{hi}}'
                and toInt64OrZero(bakery_id) in ({{id_sql}}))
        group by date,bakery_id,product_id
    """)
    release = query(f"""
        select rd date,toInt64OrZero(bid) bakery_id,toInt64OrZero(pid) product_id,sum(qty) release_qty
        from (select argMax(release_date,_updated_at) rd,argMax(bakery_id,_updated_at) bid,
              argMax(product_id,_updated_at) pid,toFloat64(argMax(quantity,_updated_at)) qty,
              argMax(is_deleted,_updated_at) deleted
              from Svezhar.fct_production_release where release_date between '{{lo}}' and '{{hi}}'
              group by release_id,line_id having deleted not in ('1','true','Да'))
        where toInt64OrZero(bid) in ({{id_sql}}) group by date,bakery_id,product_id
    """)
    moves = query(f"""
        select date,bakery_id,product_id,sum(incoming_move_qty) incoming_move_qty,
               sum(outgoing_move_qty) outgoing_move_qty from (
          select md date,toInt64OrZero(receiver) bakery_id,toInt64OrZero(pid) product_id,
                 qty incoming_move_qty,0. outgoing_move_qty from (
            select argMax(move_date,_updated_at) md,argMax(receiver_id,_updated_at) receiver,
                   argMax(product_id,_updated_at) pid,toFloat64(argMax(quantity,_updated_at)) qty,
                   argMax(is_deleted,_updated_at) deleted from Svezhar.fct_moves
            where move_date between '{{lo}}' and '{{hi}}' group by move_id,line_id
            having deleted not in ('1','true','Да')) where toInt64OrZero(receiver) in ({{id_sql}})
          union all
          select md date,toInt64OrZero(sender) bakery_id,toInt64OrZero(pid) product_id,
                 0.,qty from (
            select argMax(move_date,_updated_at) md,argMax(sender_id,_updated_at) sender,
                   argMax(product_id,_updated_at) pid,toFloat64(argMax(quantity,_updated_at)) qty,
                   argMax(is_deleted,_updated_at) deleted from Svezhar.fct_moves
            where move_date between '{{lo}}' and '{{hi}}' group by move_id,line_id
            having deleted not in ('1','true','Да')) where toInt64OrZero(sender) in ({{id_sql}})
        ) group by date,bakery_id,product_id
    """)
    writeoffs = query(f"""
        select wd date,toInt64OrZero(bid) bakery_id,toInt64OrZero(pid) product_id,
               sum(qty) written_off_qty from (
          select argMax(write_off_date,_updated_at) wd,argMax(bakery_id,_updated_at) bid,
                 argMax(write_off_product_id,_updated_at) pid,
                 toFloat64(argMax(write_off_qty,_updated_at)) qty,
                 argMax(is_deleted,_updated_at) deleted from Svezhar.fct_write_offs
          where write_off_date between '{{lo}}' and '{{hi}}' group by write_off_doc_num,line_id
          having deleted not in ('1','true','Да'))
        where toInt64OrZero(bid) in ({{id_sql}}) group by date,bakery_id,product_id
    """)
    keys = ['date','bakery_id','product_id']
    frame = sales
    for part in [release,moves,writeoffs]:
        frame = frame.merge(part,on=keys,how='outer')
    products = query("""select toInt64OrZero(product_id) product_id,
        any(product_name) product_name,any(category_name) category_name
        from Svezhar.dim_products group by product_id""")
    stores = query("""select toInt64OrZero(s.bakery_id) bakery_id,
        any(b.bakery_name) bakery_name,any(s.city) city from Svezhar.dim_stores s
        any left join Svezhar.dim_bakeries b on b.bakery_id=s.bakery_id group by bakery_id""")
    frame = frame.merge(products,on='product_id',how='left').merge(stores,on='bakery_id',how='left')
    for col in ['observed_sales_qty','observed_sales_amount','release_qty','incoming_move_qty',
                'outgoing_move_qty','written_off_qty']:
        frame[col] = pd.to_numeric(frame[col],errors='coerce').fillna(0.0)
    frame['date'] = pd.to_datetime(frame['date'])
    first_time = pd.to_datetime(frame['first_sale_time'])
    last_time = pd.to_datetime(frame['last_sale_time'])
    frame['first_sale_hour'] = first_time.dt.hour + first_time.dt.minute/60
    frame['last_sale_hour'] = last_time.dt.hour + last_time.dt.minute/60
    frame.drop(columns=['first_sale_time','last_sale_time'],inplace=True)
    path = out / f"{{start.strftime('%Y%m')}}.csv.gz"
    frame.to_csv(path,index=False,compression='gzip')
    print(path.name, len(frame), frame.bakery_id.nunique(), flush=True)
'''


def main() -> None:
    config = load_env(VM_ENV)
    ids = sorted(
        pd.read_csv(PILOT, usecols=["bakery_id"])["bakery_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    payload = base64.b64encode(remote_program(ids).encode()).decode()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        config["PROD_VM_HOST"],
        username=config["PROD_VM_USER"],
        password=config["PROD_VM_PASSWORD"],
        timeout=20,
    )
    command = (
        "cd /opt/demand-forecasting-model && .venv/bin/python -c "
        f"\"import base64;exec(base64.b64decode('{payload}'))\""
    )
    _, stdout, stderr = ssh.exec_command(command, timeout=1800)
    for line in iter(stdout.readline, ""):
        print(line.rstrip(), flush=True)
    error = stderr.read().decode("utf-8", "replace")
    if error:
        raise RuntimeError(error)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ssh.open_sftp() as sftp:
        for month in pd.date_range("2026-01-01", "2026-08-01", freq="MS"):
            name = f"{month.strftime('%Y%m')}.csv.gz"
            sftp.get(f"{REMOTE_DIR}/{name}", str(OUTPUT / name))
    ssh.close()
    files = sorted(OUTPUT.glob("2026??.csv.gz"))
    coverage = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        coverage.append(
            {
                "month": path.name[:6],
                "rows": len(frame),
                "days": frame["date"].nunique(),
                "bakeries": frame["bakery_id"].nunique(),
                "sales": frame["observed_sales_qty"].sum(),
            }
        )
    (OUTPUT / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    main()

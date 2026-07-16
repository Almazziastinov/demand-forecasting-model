"""
Monthly intraday sales profile comparison (May vs June 2026) for 10 pilot bakeries.
Uploads remote_script to VM via SFTP, runs it, collects JSON output.
Output: outputs/monthly_profile_comparison.xlsx + outputs/monthly_profile_comparison.md
"""
import sys
import time
import json
import io
from pathlib import Path
from dotenv import dotenv_values
import paramiko

secrets = dotenv_values(Path(__file__).resolve().parent / ".codex" / "prod_vm.env")
host = secrets["PROD_VM_HOST"]
user = secrets["PROD_VM_USER"]
password = secrets["PROD_VM_PASSWORD"]

# Remote script written as a plain string (no heredoc tricks needed - uploaded via SFTP)
REMOTE_SCRIPT_CONTENT = """\
import sys, json, os
sys.path.insert(0, '/opt/demand-forecasting-model')
os.chdir('/opt/demand-forecasting-model')
from pipelines.forecast_publish.load_forecast_run import create_client
c = create_client('.env')

pilot = [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]
pilot_str = ','.join(str(x) for x in pilot)

# Step 1: discover tables with sales data
print('=== STEP 1: discover tables ===', flush=True)
try:
    df_tbls = c.query_df(
        "select database, table, sum(bytes_on_disk) as bytes"
        " from system.parts where active = 1"
        " and (lower(table) like '%sale%' or lower(table) like '%check%'"
        "      or lower(table) like '%mart%' or lower(table) like '%fct%')"
        " group by database, table order by bytes desc limit 40"
    )
    print(df_tbls.to_string(index=False), flush=True)
except Exception as e:
    print(f'ERROR discovering tables: {e}', flush=True)

# Step 2: check date ranges in candidate tables
print('', flush=True)
print('=== STEP 2: date ranges ===', flush=True)
candidates = [
    ('fct_check_lines', 'check_date', 'check_datetime', 'quantity', 'bakery_id',
     "hex(cash_event_type) = 'D09FD180D0BED0B4D0B0D0B6D0B0'"),
    ('mart_sales', 'check_date', 'check_datetime', 'quantity', 'bakery_id',
     "hex(cash_event_type) = 'D09FD180D0BED0B4D0B0D0B6D0B0'"),
    ('mart_sales_60d', 'check_date', 'check_datetime', 'quantity', 'bakery_id',
     "hex(cash_event_type) = 'D09FD180D0BED0B4D0B0D0B6D0B0'"),
]
for (tbl, dc, dtc, qc, bc, evf) in candidates:
    try:
        ex = c.query_df(f"select count() as cnt from system.tables where name = '{tbl}'")
        if int(ex['cnt'].iloc[0]) == 0:
            print(f'{tbl}: not found', flush=True)
            continue
        rng = c.query_df(
            f"select min({dc}) as min_d, max({dc}) as max_d,"
            f" count(distinct {dc}) as days from {tbl}"
        )
        print(f"{tbl}.{dc}: {rng['min_d'].iloc[0]} .. {rng['max_d'].iloc[0]} ({rng['days'].iloc[0]} days)", flush=True)
    except Exception as e:
        print(f'{tbl}: ERROR {e}', flush=True)

# Step 3: hourly profile by month using best available table
print('', flush=True)
print('=== STEP 3: hourly profile by month ===', flush=True)
result_table = None
result_df = None
used_tbl_info = None

for (tbl, dc, dtc, qc, bc, evf) in candidates:
    try:
        ex = c.query_df(f"select count() as cnt from system.tables where name = '{tbl}'")
        if int(ex['cnt'].iloc[0]) == 0:
            continue
        df = c.query_df(
            f"select toMonth({dc}) as month, toHour({dtc}) as hour,"
            f" sum({qc}) as total_qty, count(distinct {dc}) as n_days"
            f" from {tbl}"
            f" where {evf}"
            f" and toInt64OrNull(toString({bc})) in ({pilot_str})"
            f" and {dc} >= '2026-05-01' and {dc} <= '2026-06-30'"
            f" group by month, hour order by month, hour"
        )
        if df.empty:
            print(f'{tbl}: empty result', flush=True)
            continue
        months_found = sorted(df['month'].unique().tolist())
        print(f'{tbl}: OK, months={months_found}, rows={len(df)}', flush=True)
        result_table = tbl
        result_df = df
        used_tbl_info = (tbl, dc, dtc, qc, bc, evf)
        break
    except Exception as e:
        print(f'{tbl}: ERROR {e}', flush=True)

if result_df is None:
    print('ERROR: could not get monthly profile from any table', flush=True)
    sys.exit(1)

tbl, dc, dtc, qc, bc, evf = used_tbl_info

# Step 4: per-bakery hourly profile
print('', flush=True)
print('=== STEP 4: per-bakery profile ===', flush=True)
df_bakery = c.query_df(
    f"select toInt64OrNull(toString({bc})) as bakery_id,"
    f" toMonth({dc}) as month, toHour({dtc}) as hour,"
    f" sum({qc}) as total_qty, count(distinct {dc}) as n_days"
    f" from {tbl}"
    f" where {evf}"
    f" and toInt64OrNull(toString({bc})) in ({pilot_str})"
    f" and {dc} >= '2026-05-01' and {dc} <= '2026-06-30'"
    f" group by bakery_id, month, hour order by bakery_id, month, hour"
)
print(f'per-bakery rows: {len(df_bakery)}', flush=True)

# Step 5: top-SKU profile
print('', flush=True)
print('=== STEP 5: top SKU profile ===', flush=True)
df_sku = None
try:
    cols = c.query_df(f"select name from system.columns where table = '{tbl}' and lower(name) like '%product%' limit 3")
    prod_col = cols['name'].iloc[0] if not cols.empty else None
except:
    prod_col = None

if prod_col:
    try:
        df_top = c.query_df(
            f"select toInt64OrNull(toString({bc})) as bakery_id, {prod_col} as product_id, sum({qc}) as total_qty"
            f" from {tbl} where {evf}"
            f" and toInt64OrNull(toString({bc})) in ({pilot_str})"
            f" and {dc} >= '2026-05-01' and {dc} <= '2026-06-30'"
            f" group by bakery_id, product_id order by bakery_id, total_qty desc"
        )
        df_top = df_top.groupby('bakery_id').head(5)
        top_products = df_top['product_id'].unique().tolist()
        top_str = ','.join(f"'{x}'" if isinstance(x, str) else str(x) for x in top_products)
        df_sku = c.query_df(
            f"select toInt64OrNull(toString({bc})) as bakery_id, {prod_col} as product_id,"
            f" toMonth({dc}) as month, toHour({dtc}) as hour, sum({qc}) as total_qty"
            f" from {tbl} where {evf}"
            f" and toInt64OrNull(toString({bc})) in ({pilot_str})"
            f" and {prod_col} in ({top_str})"
            f" and {dc} >= '2026-05-01' and {dc} <= '2026-06-30'"
            f" group by bakery_id, product_id, month, hour order by bakery_id, product_id, month, hour"
        )
        print(f'SKU profile rows: {len(df_sku)}', flush=True)
    except Exception as e:
        print(f'SKU profile ERROR: {e}', flush=True)
        df_sku = None
else:
    print('No product_id column found, skipping SKU profile', flush=True)

# Step 6: also get product names if possible
print('', flush=True)
print('=== STEP 6: product names ===', flush=True)
df_prod_names = None
if prod_col and df_sku is not None:
    try:
        name_col_candidates = ['product_name', 'nomenclature_name', 'name', 'nomenclature']
        name_col = None
        col_list = c.query_df(f"select name from system.columns where table = '{tbl}'")
        for nc in name_col_candidates:
            if nc in col_list['name'].values:
                name_col = nc
                break
        # also check sku_forecast_day_snapshots
        snap_cols = c.query_df("select name from system.columns where table = 'sku_forecast_day_snapshots' limit 20")
        print(f'sku_forecast_day_snapshots cols: {snap_cols[\"name\"].tolist()}', flush=True)

        if name_col:
            top_str2 = ','.join(f"'{x}'" if isinstance(x, str) else str(x) for x in df_sku['product_id'].unique().tolist())
            df_prod_names = c.query_df(
                f"select distinct {prod_col} as product_id, {name_col} as product_name from {tbl}"
                f" where {prod_col} in ({top_str2}) limit 1000"
            )
            print(f'Product names: {len(df_prod_names)}', flush=True)
        else:
            # try from sku_forecast_day_snapshots
            if 'product_name' in snap_cols['name'].values:
                top_str2 = ','.join(f"'{x}'" if isinstance(x, str) else str(x) for x in df_sku['product_id'].unique().tolist())
                df_prod_names = c.query_df(
                    f"select distinct product_id, product_name from sku_forecast_day_snapshots"
                    f" where product_id in ({top_str2}) limit 1000"
                )
                print(f'Product names from snapshots: {len(df_prod_names)}', flush=True)
    except Exception as e:
        print(f'Product names ERROR: {e}', flush=True)

# Save JSON to file (stdout gets truncated by pty for large payloads)
print('', flush=True)
print('=== SAVING JSON TO FILE ===', flush=True)
out = {
    'source_table': result_table,
    'overall': result_df.to_dict(orient='records'),
    'per_bakery': df_bakery.to_dict(orient='records'),
    'sku': df_sku.to_dict(orient='records') if df_sku is not None else [],
    'product_names': df_prod_names.to_dict(orient='records') if df_prod_names is not None else [],
}
OUT_PATH = '/tmp/monthly_profile_data.json'
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(out, f, default=str, ensure_ascii=False)
print(f'JSON saved to {OUT_PATH} ({len(json.dumps(out, default=str))} bytes)', flush=True)
print('=== DONE ===', flush=True)
"""


def run_remote_script(ssh_client, remote_path: str) -> tuple[str, int]:
    """Run a python script already uploaded to remote_path, return (stdout, exit_code)."""
    cmd = f"cd /opt/demand-forecasting-model && .venv/bin/python {remote_path}"
    _, stdout, stderr = ssh_client.exec_command(cmd, timeout=1200, get_pty=True)
    lines = []
    for line in iter(stdout.readline, ""):
        safe = line.encode("cp1251", errors="replace").decode("cp1251")
        sys.stdout.write(safe)
        sys.stdout.flush()
        lines.append(line)
    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        err = stderr.read().decode(errors="replace")
        print(f"\nERROR exit={exit_code}: {err}", file=sys.stderr)
    return "".join(lines), exit_code



def build_report(data: dict) -> None:
    import pandas as pd
    import numpy as np

    overall = pd.DataFrame(data["overall"])
    per_bakery = pd.DataFrame(data["per_bakery"])
    sku_df = pd.DataFrame(data["sku"]) if data["sku"] else pd.DataFrame()
    prod_names = pd.DataFrame(data["product_names"]) if data["product_names"] else pd.DataFrame()
    source = data["source_table"]

    Path("outputs").mkdir(exist_ok=True)
    months = sorted(overall["month"].unique().tolist()) if not overall.empty else []
    month_names = {5: "Май", 6: "Июнь", 7: "Июль"}

    def normalize_hourly(df, group_cols, qty_col="total_qty"):
        df = df.copy()
        totals = df.groupby(group_cols)[qty_col].transform("sum")
        df["share_pct"] = (df[qty_col] / totals * 100).round(2)
        return df

    # CH returns Decimal as string — cast to float
    for df in [overall, per_bakery, sku_df]:
        if not df.empty and "total_qty" in df.columns:
            df["total_qty"] = df["total_qty"].astype(float)

    if not overall.empty:
        overall = normalize_hourly(overall, ["month"])
    if not per_bakery.empty:
        per_bakery = normalize_hourly(per_bakery, ["bakery_id", "month"])
    if not sku_df.empty:
        sku_df = normalize_hourly(sku_df, ["bakery_id", "product_id", "month"])
        if not prod_names.empty:
            sku_df = sku_df.merge(prod_names, on="product_id", how="left")

    def month_pivot(df, index_cols, value_col="share_pct"):
        pivot = df.pivot_table(index=index_cols, columns="month", values=value_col, aggfunc="sum")
        pivot.columns = [month_names.get(int(m), str(m)) for m in pivot.columns]
        if "Июнь" in pivot.columns and "Май" in pivot.columns:
            pivot["Дельта"] = (pivot["Июнь"] - pivot["Май"]).round(2)
        return pivot.reset_index()

    # ----- Excel -----
    xl_path = "outputs/monthly_profile_comparison.xlsx"
    with pd.ExcelWriter(xl_path, engine="xlsxwriter") as writer:
        wb = writer.book
        red_fmt = wb.add_format({"bg_color": "#FFC7CE", "num_format": "+0.00;-0.00", "align": "center"})
        grn_fmt = wb.add_format({"bg_color": "#C6EFCE", "num_format": "+0.00;-0.00", "align": "center"})
        hdr_fmt = wb.add_format({"bold": True, "bg_color": "#4472C4", "font_color": "white"})

        # Sheet 1: overall
        if not overall.empty:
            piv = month_pivot(overall, ["hour"])
            piv.to_excel(writer, sheet_name="1_Общий", index=False)
            ws = writer.sheets["1_Общий"]
            ws.set_column("A:A", 8)
            ws.set_column("B:E", 12)
            # color delta column
            if "Дельта" in piv.columns:
                delta_col_idx = list(piv.columns).index("Дельта")
                for row_i, val in enumerate(piv["Дельта"]):
                    if pd.isna(val):
                        continue
                    fmt = grn_fmt if val > 0 else red_fmt
                    ws.write(row_i + 1, delta_col_idx, val, fmt)

        # Sheet 2: per-bakery
        if not per_bakery.empty:
            for bk in sorted(per_bakery["bakery_id"].dropna().unique()):
                sub = per_bakery[per_bakery["bakery_id"] == bk]
                piv = month_pivot(sub, ["hour"])
                sheet_name = f"Пек_{int(bk)}"
                piv.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                ws.set_column("A:A", 8)
                ws.set_column("B:E", 12)
                if "Дельта" in piv.columns:
                    delta_col_idx = list(piv.columns).index("Дельта")
                    for row_i, val in enumerate(piv["Дельта"]):
                        import pandas as pd2
                        if pd2.isna(val):
                            continue
                        fmt = grn_fmt if val > 0 else red_fmt
                        ws.write(row_i + 1, delta_col_idx, val, fmt)

        # Sheet 3: delta heatmap bakery x hour
        if not per_bakery.empty and 5 in per_bakery["month"].values and 6 in per_bakery["month"].values:
            may = per_bakery[per_bakery["month"] == 5].set_index(["bakery_id", "hour"])["share_pct"]
            jun = per_bakery[per_bakery["month"] == 6].set_index(["bakery_id", "hour"])["share_pct"]
            delta = (jun - may).rename("delta_pp").reset_index()
            heat = delta.pivot_table(index="bakery_id", columns="hour", values="delta_pp")
            heat.to_excel(writer, sheet_name="3_Дельта_матрица", index=True)
            ws = writer.sheets["3_Дельта_матрица"]
            ws.set_column("A:A", 14)
            nrows, ncols = heat.shape
            for r in range(nrows):
                for c_i in range(ncols):
                    val = heat.iloc[r, c_i]
                    if pd.isna(val):
                        continue
                    fmt = grn_fmt if val > 0 else red_fmt
                    ws.write(r + 1, c_i + 1, round(val, 2), fmt)

        # Sheet 4: SKU profile
        if not sku_df.empty:
            sku_df.to_excel(writer, sheet_name="4_SKU_профиль", index=False)

    print(f"\nExcel saved: {xl_path}")

    # ----- MD report -----
    import pandas as pd
    md = []
    md.append("# Сравнение часового профиля продаж: Май vs Июнь 2026")
    md.append("")
    md.append(f"**Источник:** `{source}`  ")
    md.append(f"**Пекарни:** 20, 21, 22, 28, 80, 89, 107, 221, 222, 257  ")
    md.append(f"**Месяцы в данных:** {', '.join(month_names.get(int(m), str(m)) for m in months)}")
    md.append("")
    md.append("---")
    md.append("")

    if not overall.empty:
        piv = month_pivot(overall, ["hour"])
        md.append("## Общий профиль по часам (% от дневных продаж)")
        md.append("")
        cols = list(piv.columns)
        md.append("| Час | " + " | ".join(cols[1:]) + " |")
        md.append("|-----|" + "|".join(["------"] * (len(cols) - 1)) + "|")
        for _, row in piv.iterrows():
            parts = [f"{int(row['hour']):02d}:00"]
            for col in cols[1:]:
                v = row.get(col, float("nan"))
                if pd.isna(v):
                    parts.append("-")
                elif col == "Дельта":
                    sign = "+" if v > 0 else ""
                    bold = abs(v) >= 0.5
                    s = f"{sign}{v:.2f}pp"
                    parts.append(f"**{s}**" if bold else s)
                else:
                    parts.append(f"{v:.2f}%")
            md.append("| " + " | ".join(parts) + " |")
        md.append("")

    # Top shifts
    if not overall.empty and "Дельта" in piv.columns:
        delta_col = piv.set_index("hour")["Дельта"]
        top_up = delta_col.nlargest(5)
        top_dn = delta_col.nsmallest(5)
        md.append("---")
        md.append("")
        md.append("## Ключевые сдвиги профиля")
        md.append("")
        md.append("**Часы, где доля выросла в июне (июнь > май):**")
        for h, v in top_up.items():
            md.append(f"- {int(h):02d}:00 → +{v:.2f}pp")
        md.append("")
        md.append("**Часы, где доля упала в июне (июнь < май):**")
        for h, v in top_dn.items():
            md.append(f"- {int(h):02d}:00 → {v:.2f}pp")
        md.append("")

    # Per-bakery biggest shifts
    if not per_bakery.empty and 5 in per_bakery["month"].values and 6 in per_bakery["month"].values:
        may_pb = per_bakery[per_bakery["month"] == 5].set_index(["bakery_id", "hour"])["share_pct"]
        jun_pb = per_bakery[per_bakery["month"] == 6].set_index(["bakery_id", "hour"])["share_pct"]
        delta_pb = (jun_pb - may_pb).reset_index().rename(columns={"share_pct": "delta_pp"})
        delta_pb["abs_delta"] = delta_pb["delta_pp"].abs()
        top_per_bakery = delta_pb.sort_values("abs_delta", ascending=False).groupby("bakery_id").head(3)
        top_per_bakery = top_per_bakery.sort_values(["bakery_id", "abs_delta"], ascending=[True, False])
        md.append("---")
        md.append("")
        md.append("## Топ сдвигов по пекарням (топ-3 часа с макс. дельтой)")
        md.append("")
        md.append("| Пекарня | Час | Дельта (июнь − май) |")
        md.append("|---------|-----|---------------------|")
        for _, r in top_per_bakery.iterrows():
            sign = "+" if r["delta_pp"] > 0 else ""
            md.append(f"| {int(r['bakery_id'])} | {int(r['hour']):02d}:00 | {sign}{r['delta_pp']:.2f}pp |")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## Вывод / гипотеза")
    md.append("")
    md.append("Если летний сдвиг пика спроса (16:00→18:00) подтверждается,")
    md.append("мы увидим **положительную дельту** в 17:00–18:00 и **отрицательную** в 15:00–16:00.")
    md.append("Это подтвердит, что текущий share-profile устарел и требует плановой актуализации")
    md.append("(уже запущен cron: воскресенье 02:00 UTC).")
    md.append("")
    md.append(f"Excel-детализация по каждой пекарне: `outputs/monthly_profile_comparison.xlsx`")

    md_path = "outputs/monthly_profile_comparison.md"
    Path(md_path).write_text("\n".join(md), encoding="utf-8")
    print(f"MD saved: {md_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
CACHE_PATH = Path("outputs/_monthly_profile_cache.json")

# If cache exists and is fresh (< 24h), skip SSH
import time as _time
if CACHE_PATH.exists() and (_time.time() - CACHE_PATH.stat().st_mtime) < 86400:
    print(f"Using cached data from {CACHE_PATH}")
    with open(CACHE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    build_report(data)
    print("\nDONE. Check outputs/monthly_profile_comparison.xlsx and .md")
    sys.exit(0)

print("Connecting to prod VM...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
for attempt in range(5):
    try:
        ssh.connect(host, username=user, password=password, timeout=30)
        break
    except Exception as e:
        print(f"SSH attempt {attempt+1} failed: {e}", file=sys.stderr)
        if attempt == 4:
            raise
        time.sleep(10)

print("Connected. Uploading remote script via SFTP...")
remote_script_path = "/tmp/analyze_monthly_profile_remote.py"
sftp = ssh.open_sftp()
sftp.putfo(io.BytesIO(REMOTE_SCRIPT_CONTENT.encode("utf-8")), remote_script_path)
sftp.close()
print(f"Uploaded to {remote_script_path}")

print("Running analysis on VM...\n")
raw_output, code = run_remote_script(ssh, remote_script_path)
ssh.close()

if code != 0:
    print(f"Remote script exited with code {code}", file=sys.stderr)
    if "=== DONE ===" not in raw_output:
        print("Remote script did not complete successfully.", file=sys.stderr)
        sys.exit(1)

print("\n\nDownloading JSON from VM via SFTP...")
Path("outputs").mkdir(exist_ok=True)
ssh2 = paramiko.SSHClient()
ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh2.connect(host, username=user, password=password, timeout=30)
sftp2 = ssh2.open_sftp()
sftp2.get("/tmp/monthly_profile_data.json", str(CACHE_PATH))
sftp2.close()
ssh2.close()
print(f"Downloaded to {CACHE_PATH}")

print("Parsing and building report...")
try:
    with open(CACHE_PATH, encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"Failed to load JSON: {e}", file=sys.stderr)
    sys.exit(1)

build_report(data)
print("\nDONE. Check outputs/monthly_profile_comparison.xlsx and .md")

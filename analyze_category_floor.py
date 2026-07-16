"""
Исследование category-floor uplift.
floor = category_hourly_shape × product_fraction_in_category
Сравниваем с текущим mean-floor.
"""
import sys, json, time
from pathlib import Path
from dotenv import dotenv_values
import paramiko
import pandas as pd
import numpy as np

secrets = dotenv_values(Path(".codex/prod_vm.env"))
host, user, password = secrets["PROD_VM_HOST"], secrets["PROD_VM_USER"], secrets["PROD_VM_PASSWORD"]

SCRATCHPAD = Path(r"C:\Users\dns\AppData\Local\Temp\claude\C--Users-dns-Desktop-Projects-demand-forecasting-model\bba476d3-7a0b-4cf9-81ce-28411a2a5519\scratchpad")
REMOTE_SCRIPT = SCRATCHPAD / "category_floor_remote.py"
REMOTE_PATH = "/tmp/category_floor_remote.py"
REMOTE_JSON = "/tmp/category_floor.json"
CACHE = Path("outputs/_category_floor_cache.json")


def run_on_vm():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(5):
        try:
            ssh.connect(host, username=user, password=password, timeout=30)
            break
        except Exception as e:
            print(f"SSH attempt {attempt+1}: {e}")
            if attempt == 4: raise
            time.sleep(10)

    print("Uploading script...")
    sftp = ssh.open_sftp()
    sftp.put(str(REMOTE_SCRIPT), REMOTE_PATH)
    sftp.close()

    print("Running on VM (may take 3-5 min)...")
    _, stdout, _ = ssh.exec_command(
        f"cd /opt/demand-forecasting-model && .venv/bin/python {REMOTE_PATH}",
        timeout=900, get_pty=True
    )
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line.encode("cp1251", errors="replace").decode("cp1251"))
        sys.stdout.flush()
    stdout.channel.recv_exit_status()

    print("\nDownloading results...")
    sftp = ssh.open_sftp()
    Path("outputs").mkdir(exist_ok=True)
    sftp.get(REMOTE_JSON, str(CACHE))
    sftp.close()
    ssh.close()
    print(f"Saved to {CACHE}")


def build_report(data: dict):
    df = pd.DataFrame(data["comparison"])
    summary = data["summary"]
    by_cat = pd.DataFrame(data["by_category"])
    by_nd  = pd.DataFrame(data["by_nd_bin"])
    top_cats = data.get("top_cats", [])

    for col in df.select_dtypes("object").columns:
        if col not in ["bakery_id","pid_int","category_name","dow","hour","nd_bin"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in by_cat.select_dtypes("object").columns:
        if col != "category_name":
            by_cat[col] = pd.to_numeric(by_cat[col], errors="coerce")

    xl_path = "outputs/category_floor.xlsx"
    with pd.ExcelWriter(xl_path, engine="xlsxwriter") as writer:
        wb = writer.book
        hdr_fmt = wb.add_format({"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1})
        grn_fmt = wb.add_format({"bg_color": "#C6EFCE", "border": 1})
        red_fmt = wb.add_format({"bg_color": "#FFC7CE", "border": 1})
        yel_fmt = wb.add_format({"bg_color": "#FFEB9C", "border": 1})
        num_fmt = wb.add_format({"num_format": "0.0000", "border": 1})
        pct_fmt = wb.add_format({"num_format": "0.0%", "border": 1})
        base_fmt= wb.add_format({"border": 1})

        # Sheet 1: Сводка
        rows = [
            ["Метрика", "Значение", "Интерпретация"],
            ["% ячеек: cat_floor > mean (агрессивнее)", f"{summary['pct_cat_higher']:.1%}",
             "Cat-floor тянет выше чем текущий mean"],
            ["% ячеек: cat_floor < mean (мягче)", f"{summary['pct_cat_lower']:.1%}",
             "Cat-floor ниже — меньше искажений"],
            ["avg floor_ratio (cat/mean)", f"{summary['avg_ratio']:.3f}", "1.0 = одинаково"],
            ["median floor_ratio", f"{summary['median_ratio']:.3f}", ""],
            ["median floor_n_days (стабильность floor)", f"{summary['floor_nd_median']:.0f}",
             "Сколько наблюдений у category-floor"],
            ["median current n_days", f"{summary['cur_nd_median']:.0f}",
             "Сколько наблюдений у текущего профиля"],
        ]
        pd.DataFrame(rows[1:], columns=rows[0]).to_excel(writer, sheet_name="1_Сводка", index=False)
        ws = writer.sheets["1_Сводка"]
        ws.set_column("A:A", 42); ws.set_column("B:B", 14); ws.set_column("C:C", 36)

        # Sheet 2: По категориям
        if len(by_cat):
            by_cat.to_excel(writer, sheet_name="2_По_категориям", index=False)
            ws = writer.sheets["2_По_категориям"]
            ws.set_column("A:A", 26); ws.set_column("B:G", 16)

        # Sheet 3: По бакетам n_days
        if len(by_nd):
            by_nd.to_excel(writer, sheet_name="3_По_n_days", index=False)
            ws = writer.sheets["3_По_n_days"]
            ws.set_column("A:D", 18)

        # Sheet 4: Детализация — где cat_floor сильно выше
        df_higher = df[df["cat_floor_higher"] == True].copy()
        df_higher["floor_diff"] = pd.to_numeric(df_higher["floor_diff"], errors="coerce")
        df_higher = df_higher.sort_values("floor_diff", ascending=False).head(200)
        if len(df_higher):
            df_higher[["bakery_id","pid_int","category_name","dow","hour",
                        "n_days","mean_sku_share_in_hour","cat_floor_mean",
                        "floor_diff","floor_ratio","floor_n_days"]].to_excel(
                writer, sheet_name="4_Floor_выше", index=False)
            ws = writer.sheets["4_Floor_выше"]
            ws.set_column("A:B", 10); ws.set_column("C:C", 24); ws.set_column("D:K", 16)

        # Sheet 5: Детализация — где cat_floor ниже
        df_lower = df[df["cat_floor_lower"] == True].copy()
        df_lower["floor_diff"] = pd.to_numeric(df_lower["floor_diff"], errors="coerce")
        df_lower = df_lower.sort_values("floor_diff").head(200)
        if len(df_lower):
            df_lower[["bakery_id","pid_int","category_name","dow","hour",
                       "n_days","mean_sku_share_in_hour","cat_floor_mean",
                       "floor_diff","floor_ratio","floor_n_days"]].to_excel(
                writer, sheet_name="5_Floor_ниже", index=False)
            ws = writer.sheets["5_Floor_ниже"]
            ws.set_column("A:B", 10); ws.set_column("C:C", 24); ws.set_column("D:K", 16)

    print(f"Excel: {xl_path}")

    # ── MD ──────────────────────────────────────────────────────────────────────
    md = []
    md.append("# Исследование Category-Floor Uplift")
    md.append("")
    md.append("## Концепция")
    md.append("")
    md.append("```")
    md.append("floor(bakery, product, DOW, hour)")
    md.append("  = category_share(bakery, category, DOW, hour)   ← форма кривой")
    md.append("  × product_fraction_in_category(bakery, product, DOW)  ← масштаб товара")
    md.append("```")
    md.append("")
    md.append("**Преимущества vs mean-floor:**")
    md.append("- Категорийный профиль строится на в 10-20× больше данных → стабильнее")
    md.append("- Floor повторяет форму дня категории, а не плоскую горизонталь")
    md.append("- Индивидуальность пекарни сохранена (данные берутся по конкретной пекарне)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Сравнение с текущим mean-floor")
    md.append("")
    md.append("| Метрика | Значение |")
    md.append("|---|---|")
    md.append(f"| cat_floor > mean (агрессивнее) | **{summary['pct_cat_higher']:.1%}** |")
    md.append(f"| cat_floor < mean (мягче) | **{summary['pct_cat_lower']:.1%}** |")
    md.append(f"| avg floor_ratio (cat/mean) | {summary['avg_ratio']:.3f} |")
    md.append(f"| median floor_ratio | {summary['median_ratio']:.3f} |")
    md.append(f"| median floor_n_days | **{summary['floor_nd_median']:.0f}** vs текущий {summary['cur_nd_median']:.0f} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## По категориям")
    md.append("")
    md.append("| Категория | n | avg ratio | % floor выше | avg floor_nd | avg cur_nd |")
    md.append("|---|---|---|---|---|---|")
    for _, r in by_cat.sort_values("avg_ratio", ascending=False).iterrows():
        md.append(f"| {r.get('category_name','')} | {int(r.get('n',0))} "
                  f"| {r.get('avg_ratio',1):.3f} | {r.get('pct_higher',0):.1%} "
                  f"| {r.get('avg_floor_nd',0):.0f} | {r.get('avg_cur_nd',0):.0f} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## По бакетам n_days (текущего профиля)")
    md.append("")
    md.append("| n_days | n | avg ratio | % cat_floor выше |")
    md.append("|---|---|---|---|")
    for _, r in by_nd.iterrows():
        md.append(f"| {r.get('nd_bin','')} | {int(r.get('n',0))} "
                  f"| {r.get('avg_ratio',1):.3f} | {r.get('pct_cat_higher',0):.1%} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Вывод")
    md.append("")
    ratio = summary['avg_ratio']
    pct_h = summary['pct_cat_higher']
    fnd   = summary['floor_nd_median']
    cnd   = summary['cur_nd_median']

    md.append(f"- **Стабильность:** category-floor строится на median {fnd:.0f} наблюдениях "
              f"vs {cnd:.0f} у текущего профиля — в {fnd/max(cnd,1):.1f}× больше данных")
    if ratio > 1.1:
        md.append(f"- **Cat-floor систематически выше** (avg ratio={ratio:.3f}): "
                  f"категорийный подход более агрессивен в подъёме долей")
    elif ratio < 0.9:
        md.append(f"- **Cat-floor систематически ниже** (avg ratio={ratio:.3f}): "
                  f"более мягкий floor → меньше искажений профиля")
    else:
        md.append(f"- **Cat-floor и mean-floor близки** (avg ratio={ratio:.3f}) "
                  f"но cat-floor значительно стабильнее")

    md.append(f"- **{pct_h:.0%} ячеек** получат более высокий floor при переходе на category-floor")
    md.append("")
    md.append(f"Excel: `outputs/category_floor.xlsx`")

    md_path = Path("outputs/category_floor.md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"MD: {md_path}")


# ── main ──────────────────────────────────────────────────────────────────────
if CACHE.exists() and (time.time() - CACHE.stat().st_mtime) < 86400:
    print(f"Using cache: {CACHE}")
    with open(CACHE, encoding="utf-8") as f:
        data = json.load(f)
else:
    run_on_vm()
    with open(CACHE, encoding="utf-8") as f:
        data = json.load(f)

build_report(data)
print("\nDONE. outputs/category_floor.xlsx + .md")

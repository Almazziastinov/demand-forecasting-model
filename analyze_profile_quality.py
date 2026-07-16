"""
Оценка качества SKU hour share профиля.
Запускает анализ на VM, скачивает, строит Excel + MD.
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
REMOTE_SCRIPT = SCRATCHPAD / "profile_quality_remote.py"
REMOTE_PATH = "/tmp/profile_quality_remote.py"
REMOTE_JSON = "/tmp/profile_quality.json"
CACHE = Path("outputs/_profile_quality_cache.json")


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

    print("Running on VM (may take 2-3 min)...")
    _, stdout, _ = ssh.exec_command(
        f"cd /opt/demand-forecasting-model && .venv/bin/python {REMOTE_PATH}",
        timeout=600, get_pty=True
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
    xl_path = "outputs/profile_quality.xlsx"

    by_cat = pd.DataFrame(data.get("by_category", []))
    by_hour = pd.DataFrame(data.get("by_hour", []))
    for col in by_cat.select_dtypes("object").columns:
        if col != "category_name":
            by_cat[col] = pd.to_numeric(by_cat[col], errors="coerce")
    for col in by_hour.select_dtypes("object").columns:
        if col != "hour":
            by_hour[col] = pd.to_numeric(by_hour[col], errors="coerce")

    mm = data.get("mean_median_summary", {})
    cv = data.get("cv_summary", {})
    nd = data.get("n_days_bins", {})

    with pd.ExcelWriter(xl_path, engine="xlsxwriter") as writer:
        wb = writer.book
        red_fmt   = wb.add_format({"bg_color": "#FFC7CE", "border": 1})
        yel_fmt   = wb.add_format({"bg_color": "#FFEB9C", "border": 1})
        grn_fmt   = wb.add_format({"bg_color": "#C6EFCE", "border": 1})
        hdr_fmt   = wb.add_format({"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1})
        pct_fmt   = wb.add_format({"num_format": "0.0%", "border": 1})
        num3_fmt  = wb.add_format({"num_format": "0.000", "border": 1})
        base_fmt  = wb.add_format({"border": 1})

        # Sheet 1: Общая сводка
        summary_rows = [
            ["Метрика", "Значение", "Интерпретация"],
            ["Всего строк профиля", data.get("profile_rows", ""), ""],
            ["", "", ""],
            ["--- n_days ---", "", ""],
        ]
        for k, v in nd.items():
            pct = v / data.get("profile_rows", 1) * 100
            summary_rows.append([f"n_days = {k}", v, f"{pct:.1f}%"])
        summary_rows += [
            ["", "", ""],
            ["--- Mean vs Median ---", "", ""],
            ["% строк где mean > median (+0.001)", f"{mm.get('pct_mean_inflated',0):.1%}", "Доля завышенных средних"],
            ["avg mean/median ratio", f"{mm.get('avg_ratio',1):.3f}", "> 1.0 = среднее завышено"],
            ["% строк mean/median > 1.2", f"{mm.get('pct_ratio_gt_1_2',0):.1%}", "Умеренное завышение"],
            ["% строк mean/median > 1.5", f"{mm.get('pct_ratio_gt_1_5',0):.1%}", "Сильное завышение"],
            ["", "", ""],
            ["--- CV (нестабильность) ---", "", ""],
            ["median CV (std/mean)", f"{cv.get('median_cv',0):.3f}", "0.3-0.5 = умеренно, >1 = высокая"],
            ["% строк с CV > 0.5", f"{cv.get('pct_cv_gt_05',0):.1%}", ""],
            ["% строк с CV > 1.0", f"{cv.get('pct_cv_gt_1',0):.1%}", "Сильная нестабильность профиля"],
        ]
        df_sum = pd.DataFrame(summary_rows[1:], columns=summary_rows[0])
        df_sum.to_excel(writer, sheet_name="1_Сводка", index=False)
        ws = writer.sheets["1_Сводка"]
        ws.set_column("A:A", 38); ws.set_column("B:B", 16); ws.set_column("C:C", 32)

        # Sheet 2: По категориям
        if len(by_cat):
            by_cat.to_excel(writer, sheet_name="2_По_категориям", index=False)
            ws = writer.sheets["2_По_категориям"]
            ws.set_column("A:A", 26); ws.set_column("B:F", 18)

        # Sheet 3: По часам
        if len(by_hour):
            by_hour_sorted = by_hour.sort_values("hour") if "hour" in by_hour.columns else by_hour
            by_hour_sorted.to_excel(writer, sheet_name="3_По_часам", index=False)
            ws = writer.sheets["3_По_часам"]
            ws.set_column("A:D", 18)

    print(f"Excel: {xl_path}")

    # ── MD ──────────────────────────────────────────────────────────────────────
    md = []
    md.append("# Оценка качества SKU hour share профиля")
    md.append("")
    md.append("**Пекарни:** 10 пилотных  |  **Профиль:** `sku_hour_share_profile_smoothed_embedded`")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Количество наблюдений (n_days)")
    md.append("")
    md.append("| n_days | строк | % |")
    md.append("|---|---|---|")
    total = data.get("profile_rows", 1)
    for k, v in nd.items():
        md.append(f"| {k} | {v} | {v/total*100:.1f}% |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Mean vs Median (чувствительность к выбросам)")
    md.append("")
    md.append("| Метрика | Значение |")
    md.append("|---|---|")
    md.append(f"| % строк где mean > median | **{mm.get('pct_mean_inflated',0):.1%}** |")
    md.append(f"| avg mean/median ratio | **{mm.get('avg_ratio',1):.3f}** |")
    md.append(f"| % mean > median × 1.2 | {mm.get('pct_ratio_gt_1_2',0):.1%} |")
    md.append(f"| % mean > median × 1.5 | {mm.get('pct_ratio_gt_1_5',0):.1%} |")
    md.append("")
    if len(by_cat):
        md.append("### По категориям")
        md.append("")
        md.append("| Категория | n | avg mean/median | % завышено | avg delta% |")
        md.append("|---|---|---|---|---|")
        for _, r in by_cat.sort_values("avg_ratio", ascending=False).iterrows():
            md.append(f"| {r.get('category_name','')} | {int(r.get('n',0))} "
                      f"| {r.get('avg_ratio',1):.3f} | {r.get('pct_inflated',0):.1%} "
                      f"| {r.get('avg_pct_delta',0):+.1f}% |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. CV — нестабильность профиля")
    md.append("")
    md.append("| Метрика | Значение |")
    md.append("|---|---|")
    md.append(f"| median CV | **{cv.get('median_cv',0):.3f}** |")
    md.append(f"| % строк CV > 0.5 | {cv.get('pct_cv_gt_05',0):.1%} |")
    md.append(f"| % строк CV > 1.0 | **{cv.get('pct_cv_gt_1',0):.1%}** |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Mean vs Median по часам")
    md.append("")
    if len(by_hour):
        md.append("| Час | avg mean/median | avg delta | % завышено |")
        md.append("|---|---|---|---|")
        for _, r in by_hour.sort_values("hour").iterrows():
            md.append(f"| {int(r.get('hour',0))}:00 | {r.get('avg_ratio',1):.3f} "
                      f"| {r.get('avg_delta',0):+.5f} | {r.get('pct_inflated',0):.1%} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Выводы и рекомендации")
    md.append("")
    pct_inf = mm.get("pct_mean_inflated", 0)
    ratio   = mm.get("avg_ratio", 1)
    pct_cv1 = cv.get("pct_cv_gt_1", 0)
    pct_nd5 = nd.get("1-3", 0) + nd.get("4-5", 0)
    pct_nd5_pct = pct_nd5 / total

    if ratio > 1.1:
        md.append(f"- **Среднее систематически завышено** ({pct_inf:.0%} строк, ratio={ratio:.3f}) — "
                  f"выбросы вверх тянут mean выше медианы. Рекомендуется перейти на **медиану** как основу профиля.")
    else:
        md.append(f"- Mean и median близки (ratio={ratio:.3f}) — завышение несущественное.")

    if pct_cv1 > 0.15:
        md.append(f"- **Высокая нестабильность** у {pct_cv1:.0%} строк (CV > 1.0) — "
                  f"профиль нестабилен для этих ячеек. Рекомендуется минимальный порог n_days и trimmed mean.")
    if pct_nd5_pct > 0.05:
        md.append(f"- **{pct_nd5_pct:.0%} строк имеют < 6 наблюдений** — статистики ненадёжны, "
                  f"нужен fallback на агрегированный профиль (категория / пекарня).")

    md.append(f"- **Smoothing (floor=mean)**: если mean завышен выбросами, "
              f"то floor поднимает все дни до завышенного уровня — двойное искажение.")
    md.append(f"- **Рекомендация**: использовать **trimmed mean (5-95%)** или **медиану** "
              f"вместо weighted mean + убрать smoothing или заменить floor на более мягкий percentile (p25).")
    md.append("")
    md.append(f"Excel: `outputs/profile_quality.xlsx`")

    md_path = Path("outputs/profile_quality.md")
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
print("\nDONE. outputs/profile_quality.xlsx + .md")

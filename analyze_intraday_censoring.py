"""
Внутридневные сигналы цензурирования: запуск на VM, построение отчёта.

Сигналы на уровне (bakery, product, date):
  S1. last_sale_hour < 16:00                          — последняя продажа слишком рано
  S2. >= 2 часов: пекарня торгует, SKU = 0            — покупатели есть, товара нет
  S3. разрыв нулей между продажами (mid-day gap)      — выкуп между партиями
  S4. last_sale_hour < typical_last_hour(DOW) - 2ч   — хвост обрезан vs типичный день
  S5. closing_stock = 0                               — распродано полностью

Классификация (bakery × product):
  CENSORED   — >= 3 сигналов срабатывают в >= 30% дней
  AMBIGUOUS  — 1-2 сигнала
  PATTERN    — 0 сигналов
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
REMOTE_SCRIPT = SCRATCHPAD / "intraday_censoring_remote.py"
REMOTE_PATH = "/tmp/intraday_censoring_remote.py"
REMOTE_JSON = "/tmp/intraday_censoring.json"
CACHE = Path("outputs/_intraday_censoring_cache.json")


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

    print("Running on VM (may take 4-6 min)...")
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
    df = pd.DataFrame(data["signals"])
    top_cats = data.get("top_cats", [])
    counts = data["counts"]
    total = sum(counts.values())

    LABEL_RU = {"CENSORED": "Цензурирован", "AMBIGUOUS": "Неоднозначно", "PATTERN": "Реальный паттерн"}
    signal_descriptions = {
        "s1_early_last_pct":   "S1: последняя продажа до 16:00",
        "s2_bk_active_pct":    "S2: пекарня торгует, SKU = 0 (≥2ч)",
        "s3_mid_gap_pct":      "S3: разрыв нулей посередине дня",
        "s4_tail_cutoff_pct":  "S4: хвост обрезан vs типичный день (-2ч)",
        "s5_closing_zero_pct": "S5: closing_stock = 0",
    }

    for col in [c for c in df.columns if '_pct' in c or 'n_signals' in c or 'avg_' in c]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    xl_path = "outputs/intraday_censoring.xlsx"
    with pd.ExcelWriter(xl_path, engine="xlsxwriter") as writer:
        wb = writer.book
        cens_fmt  = wb.add_format({"bg_color": "#FFC7CE", "border": 1})
        ambig_fmt = wb.add_format({"bg_color": "#FFEB9C", "border": 1})
        pat_fmt   = wb.add_format({"bg_color": "#C6EFCE", "border": 1})
        pct_fmt   = wb.add_format({"num_format": "0%", "border": 1})
        base_fmt  = wb.add_format({"border": 1})

        # Sheet 1: Сводка по категориям
        cat_lbl = df.groupby(["category_name","label"]).size().unstack(fill_value=0).reset_index()
        for lbl in ["CENSORED","AMBIGUOUS","PATTERN"]:
            if lbl not in cat_lbl.columns: cat_lbl[lbl] = 0
        cat_lbl["total"] = cat_lbl[["CENSORED","AMBIGUOUS","PATTERN"]].sum(axis=1)
        cat_lbl["censored_pct"] = (cat_lbl["CENSORED"] / cat_lbl["total"].replace(0, np.nan)).round(3)
        cat_lbl.to_excel(writer, sheet_name="1_Сводка", index=False)
        ws = writer.sheets["1_Сводка"]
        ws.set_column("A:A", 24); ws.set_column("B:G", 14)

        # Sheet 2: Все SKU × пекарня
        out_cols = ["bakery_id","pid_int","product_name","category_name","label","n_signals",
                    "s1_early_last_pct","s2_bk_active_pct","s3_mid_gap_pct",
                    "s4_tail_cutoff_pct","s5_closing_zero_pct",
                    "avg_last_hour","avg_typical_last","n_days"]
        df_out = df[[c for c in out_cols if c in df.columns]].copy()
        df_out.sort_values(["label","n_signals"], ascending=[True, False], inplace=True)
        df_out.to_excel(writer, sheet_name="2_Детализация", index=False)
        ws = writer.sheets["2_Детализация"]
        ws.set_column("A:B", 10); ws.set_column("C:C", 30)
        ws.set_column("D:D", 22); ws.set_column("E:N", 14)

        # Sheet 3: CENSORED
        df_cens = df[df["label"]=="CENSORED"].sort_values("n_signals", ascending=False)
        if len(df_cens):
            df_cens[[c for c in out_cols if c in df_cens.columns]].to_excel(
                writer, sheet_name="3_Цензурированные", index=False)
            ws = writer.sheets["3_Цензурированные"]
            ws.set_column("A:B", 10); ws.set_column("C:C", 30); ws.set_column("D:N", 14)

        # Sheet 4: PATTERN
        df_pat = df[df["label"]=="PATTERN"].sort_values("n_signals")
        if len(df_pat):
            df_pat[[c for c in out_cols if c in df_pat.columns]].to_excel(
                writer, sheet_name="4_Реальный_паттерн", index=False)

        # Sheet 5: Сигналы по категориям
        sig_cat = df.groupby("category_name").agg(
            n_pairs=("bakery_id","count"),
            censored=("label", lambda x: (x=="CENSORED").sum()),
            avg_s1=("s1_early_last_pct","mean"),
            avg_s2=("s2_bk_active_pct","mean"),
            avg_s3=("s3_mid_gap_pct","mean"),
            avg_s4=("s4_tail_cutoff_pct","mean"),
            avg_s5=("s5_closing_zero_pct","mean"),
        ).round(3).reset_index()
        sig_cat["censored_pct"] = (sig_cat["censored"] / sig_cat["n_pairs"]).round(3)
        sig_cat.to_excel(writer, sheet_name="5_Сигналы_по_категориям", index=False)
        ws = writer.sheets["5_Сигналы_по_категориям"]
        ws.set_column("A:A", 24); ws.set_column("B:J", 14)

    print(f"Excel: {xl_path}")

    # ── MD ──────────────────────────────────────────────────────────────────────
    sig_cat_md = df.groupby("category_name").agg(
        n_pairs=("bakery_id","count"),
        censored=("label", lambda x: (x=="CENSORED").sum()),
        pattern=("label", lambda x: (x=="PATTERN").sum()),
        ambiguous=("label", lambda x: (x=="AMBIGUOUS").sum()),
        avg_s2=("s2_bk_active_pct","mean"),
        avg_s4=("s4_tail_cutoff_pct","mean"),
        avg_s5=("s5_closing_zero_pct","mean"),
    ).round(3).reset_index()
    sig_cat_md["censored_pct"] = (sig_cat_md["censored"] / sig_cat_md["n_pairs"]).round(3)

    md = []
    md.append("# Внутридневное цензурирование спроса")
    md.append("")
    md.append("**Период:** май–июнь 2026  |  **Пекарни:** 10 пилотных  |  **Категории:** топ-6")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Сигналы цензурирования")
    md.append("")
    md.append("| Сигнал | Описание | Порог срабатывания |")
    md.append("|---|---|---|")
    md.append("| **S1** | Последняя продажа до 16:00 | ≥30% дней |")
    md.append("| **S2** | Пекарня торгует, SKU = 0 (≥2 часа) | ≥30% дней |")
    md.append("| **S3** | Разрыв нулей посередине дня | ≥30% дней |")
    md.append("| **S4** | Хвост продаж обрезан vs типичный DOW на 2+ ч | ≥30% дней |")
    md.append("| **S5** | closing_stock = 0 (распродано полностью) | ≥30% дней |")
    md.append("")
    md.append("**Классификация:** CENSORED ≥3 сигналов | AMBIGUOUS 1-2 | PATTERN 0")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Итог")
    md.append("")
    md.append("| Класс | SKU × пекарня | % |")
    md.append("|---|---|---|")
    for lbl, ru in LABEL_RU.items():
        n = counts.get(lbl, 0)
        md.append(f"| **{ru}** | {n} | {n/total*100:.0f}% |")
    md.append(f"| **Всего** | {total} | 100% |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## По категориям")
    md.append("")
    md.append("| Категория | Всего | Цензур. | Паттерн | Неод. | Цензур.% | S2 avg | S4 avg | S5 avg |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in sig_cat_md.sort_values("censored_pct", ascending=False).iterrows():
        md.append(
            f"| {r['category_name']} | {r['n_pairs']} | {r['censored']} | {r['pattern']} | {r['ambiguous']} "
            f"| {r['censored_pct']:.0%} | {r['avg_s2']:.0%} | {r['avg_s4']:.0%} | {r['avg_s5']:.0%} |"
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Топ цензурированных")
    md.append("")
    top_c = df[df["label"]=="CENSORED"].sort_values("n_signals", ascending=False).head(20)
    md.append("| Пекарня | Продукт | Кат. | Сигналов | S1 | S2 | S3 | S4 | S5 | avg last h | типичный h |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in top_c.iterrows():
        md.append(
            f"| {r['bakery_id']} | {r['product_name']} | {r['category_name']} | {int(r['n_signals'])} "
            f"| {r['s1_early_last_pct']:.0%} | {r['s2_bk_active_pct']:.0%} | {r['s3_mid_gap_pct']:.0%} "
            f"| {r['s4_tail_cutoff_pct']:.0%} | {r['s5_closing_zero_pct']:.0%} "
            f"| {r['avg_last_hour']:.0f}:00 | {r['avg_typical_last']:.0f}:00 |"
        )
    md.append("")
    md.append(f"Excel: `outputs/intraday_censoring.xlsx`")

    md_path = Path("outputs/intraday_censoring.md")
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
print("\nDONE. outputs/intraday_censoring.xlsx + .md")

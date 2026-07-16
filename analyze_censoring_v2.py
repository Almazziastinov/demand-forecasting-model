"""
Классификация цензурирования SKU через rolling inventory (closing_stock=0 rate).
Запускает анализ на VM, скачивает результат, строит Excel + MD.

Правило классификации:
  closing_stock=0 в >= 50% активных дней -> CENSORED
  10-50%                                  -> AMBIGUOUS
  < 10%                                   -> PATTERN
"""
import sys, io, json, time
from pathlib import Path
from dotenv import dotenv_values
import paramiko
import pandas as pd
import numpy as np

secrets = dotenv_values(Path(".codex/prod_vm.env"))
host, user, password = secrets["PROD_VM_HOST"], secrets["PROD_VM_USER"], secrets["PROD_VM_PASSWORD"]

SCRATCHPAD = Path(r"C:\Users\dns\AppData\Local\Temp\claude\C--Users-dns-Desktop-Projects-demand-forecasting-model\bba476d3-7a0b-4cf9-81ce-28411a2a5519\scratchpad")
REMOTE_SCRIPT = SCRATCHPAD / "censoring_v2_remote.py"
REMOTE_PATH = "/tmp/censoring_v2_remote.py"
REMOTE_JSON = "/tmp/censoring_v2.json"
CACHE = Path("outputs/_censoring_v2_cache.json")


def run_on_vm():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(5):
        try:
            ssh.connect(host, username=user, password=password, timeout=30)
            break
        except Exception as e:
            print(f"SSH attempt {attempt+1}: {e}")
            if attempt == 4:
                raise
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
    code = stdout.channel.recv_exit_status()
    if code != 0:
        print(f"Exit code {code} — trying to download anyway")

    print("\nDownloading results...")
    sftp = ssh.open_sftp()
    Path("outputs").mkdir(exist_ok=True)
    sftp.get(REMOTE_JSON, str(CACHE))
    sftp.close()
    ssh.close()
    print(f"Saved to {CACHE}")


def build_report(data: dict):
    df = pd.DataFrame(data["classification"])
    top_cats = data.get("top_cats", [])

    LABEL_COLOR = {"CENSORED": "#FFC7CE", "AMBIGUOUS": "#FFEB9C", "PATTERN": "#C6EFCE"}
    LABEL_RU = {"CENSORED": "Цензурирован", "AMBIGUOUS": "Неоднозначно", "PATTERN": "Реальный паттерн"}

    for col in ["zero_closing_pct", "avg_sell_through", "avg_produced", "avg_sold", "n_active_days", "n_zero_closing"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    counts = df["label"].value_counts().to_dict()
    total = len(df)

    xl_path = "outputs/censoring_rolling_sellthrough.xlsx"
    with pd.ExcelWriter(xl_path, engine="xlsxwriter") as writer:
        wb = writer.book
        hdr_fmt = wb.add_format({"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1})
        cens_fmt = wb.add_format({"bg_color": "#FFC7CE", "border": 1})
        ambig_fmt = wb.add_format({"bg_color": "#FFEB9C", "border": 1})
        pat_fmt  = wb.add_format({"bg_color": "#C6EFCE", "border": 1})
        pct_fmt  = wb.add_format({"num_format": "0%", "border": 1})
        num2_fmt = wb.add_format({"num_format": "0.00", "border": 1})
        base_fmt = wb.add_format({"border": 1})
        label_fmts = {"CENSORED": cens_fmt, "AMBIGUOUS": ambig_fmt, "PATTERN": pat_fmt}

        # --- Sheet 1: Сводка по категориям ---
        cat_sum = df.groupby(["category_name", "label"]).size().unstack(fill_value=0).reset_index()
        for lbl in ["CENSORED", "AMBIGUOUS", "PATTERN"]:
            if lbl not in cat_sum.columns:
                cat_sum[lbl] = 0
        cat_sum["total"] = cat_sum[["CENSORED", "AMBIGUOUS", "PATTERN"]].sum(axis=1)
        cat_sum["censored_pct"] = (cat_sum["CENSORED"] / cat_sum["total"].replace(0, np.nan)).round(3)
        cat_sum.to_excel(writer, sheet_name="1_Сводка", index=False)
        ws = writer.sheets["1_Сводка"]
        ws.set_column("A:A", 24)
        ws.set_column("B:G", 14)

        # --- Sheet 2: Детализация всех SKU × пекарня ---
        df_out = df[[
            "bakery_id", "pid_int", "product_name", "category_name", "label",
            "n_active_days", "n_zero_closing", "zero_closing_pct",
            "avg_sell_through", "avg_produced", "avg_sold"
        ]].copy()
        df_out["label_ru"] = df_out["label"].map(LABEL_RU)
        df_out.sort_values(["label", "zero_closing_pct"], ascending=[True, False], inplace=True)
        df_out.to_excel(writer, sheet_name="2_Детализация", index=False)
        ws = writer.sheets["2_Детализация"]
        ws.set_column("A:B", 10)
        ws.set_column("C:C", 30)
        ws.set_column("D:D", 22)
        ws.set_column("E:L", 14)

        # --- Sheet 3: только CENSORED ---
        df_cens = df[df["label"] == "CENSORED"].sort_values("zero_closing_pct", ascending=False)
        if len(df_cens):
            df_cens_out = df_cens[[
                "bakery_id", "pid_int", "product_name", "category_name",
                "n_active_days", "n_zero_closing", "zero_closing_pct",
                "avg_sell_through", "avg_produced", "avg_sold"
            ]].copy()
            df_cens_out.to_excel(writer, sheet_name="3_Цензурированные", index=False)
            ws = writer.sheets["3_Цензурированные"]
            ws.set_column("A:B", 10); ws.set_column("C:C", 30); ws.set_column("D:J", 14)

        # --- Sheet 4: только PATTERN ---
        df_pat = df[df["label"] == "PATTERN"].sort_values("zero_closing_pct")
        if len(df_pat):
            df_pat_out = df_pat[[
                "bakery_id", "pid_int", "product_name", "category_name",
                "n_active_days", "n_zero_closing", "zero_closing_pct",
                "avg_sell_through", "avg_produced", "avg_sold"
            ]].copy()
            df_pat_out.to_excel(writer, sheet_name="4_Реальный_паттерн", index=False)
            ws = writer.sheets["4_Реальный_паттерн"]
            ws.set_column("A:B", 10); ws.set_column("C:C", 30); ws.set_column("D:J", 14)

        # --- Sheet 5: AMBIGUOUS ---
        df_amb = df[df["label"] == "AMBIGUOUS"].sort_values("zero_closing_pct", ascending=False)
        if len(df_amb):
            df_amb_out = df_amb[[
                "bakery_id", "pid_int", "product_name", "category_name",
                "n_active_days", "n_zero_closing", "zero_closing_pct",
                "avg_sell_through", "avg_produced", "avg_sold"
            ]].copy()
            df_amb_out.to_excel(writer, sheet_name="5_Неоднозначно", index=False)
            ws = writer.sheets["5_Неоднозначно"]
            ws.set_column("A:B", 10); ws.set_column("C:C", 30); ws.set_column("D:J", 14)

    print(f"Excel: {xl_path}")

    # ---- MD report ----
    sig_agg = df.groupby("category_name").agg(
        n_sku_bakery=("bakery_id", "count"),
        censored=("label", lambda x: (x == "CENSORED").sum()),
        pattern=("label", lambda x: (x == "PATTERN").sum()),
        ambiguous=("label", lambda x: (x == "AMBIGUOUS").sum()),
        avg_sell_through=("avg_sell_through", "mean"),
        avg_zero_closing_pct=("zero_closing_pct", "mean"),
    ).round(3).reset_index()
    sig_agg["censored_pct"] = (sig_agg["censored"] / sig_agg["n_sku_bakery"]).round(3)

    md = []
    md.append("# Анализ цензурирования SKU — Rolling Sell-Through")
    md.append("")
    md.append("**Метод:** Rolling inventory (closing_stock = opening_stock + production - sales)")
    md.append("**Сигнал:** % активных дней, когда closing_stock = 0 (товар полностью распродан)")
    md.append("**Период:** май–июнь 2026  |  **Пекарни:** 10 пилотных")
    if top_cats:
        md.append(f"**Категории:** {', '.join(top_cats)}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Правило классификации")
    md.append("")
    md.append("| Класс | Условие | Интерпретация |")
    md.append("|---|---|---|")
    md.append("| **CENSORED** | closing_stock=0 в ≥ 50% дней | Систематический недовыпуск — спрос цензурирован |")
    md.append("| **AMBIGUOUS** | closing_stock=0 в 10–50% дней | Смешанная ситуация — неоднозначно |")
    md.append("| **PATTERN** | closing_stock=0 в < 10% дней | Реальный паттерн спроса — выпуск достаточен |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Итог классификации")
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
    md.append("| Категория | Всего | Цензур. | Паттерн | Неод. | Цензур.% | avg zero-closing% | avg sell-through |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in sig_agg.sort_values("censored_pct", ascending=False).iterrows():
        st = f"{r['avg_sell_through']:.0%}" if not pd.isna(r["avg_sell_through"]) else "—"
        zp = f"{r['avg_zero_closing_pct']:.0%}" if not pd.isna(r["avg_zero_closing_pct"]) else "—"
        md.append(
            f"| {r['category_name']} | {r['n_sku_bakery']} | {r['censored']} | {r['pattern']} | {r['ambiguous']} "
            f"| {r['censored_pct']:.0%} | {zp} | {st} |"
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Топ цензурированных (по zero_closing_pct)")
    md.append("")
    top_c = df[df["label"] == "CENSORED"].sort_values("zero_closing_pct", ascending=False).head(15)
    md.append("| Пекарня | Продукт | Категория | zero-closing% | avg sell-through | avg выпуск | avg продажи |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in top_c.iterrows():
        st = f"{r['avg_sell_through']:.0%}" if not pd.isna(r["avg_sell_through"]) else "—"
        md.append(
            f"| {r['bakery_id']} | {r['product_name']} | {r['category_name']} "
            f"| {r['zero_closing_pct']:.0%} | {st} | {r['avg_produced']:.0f} | {r['avg_sold']:.0f} |"
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Вывод")
    md.append("")
    n_cens = counts.get("CENSORED", 0)
    n_pat  = counts.get("PATTERN", 0)
    n_amb  = counts.get("AMBIGUOUS", 0)
    md.append(f"- **{n_cens} из {total}** ({n_cens/total*100:.0f}%) SKU × пекарня — **цензурированы** (closing_stock=0 ≥50% дней)")
    md.append(f"- **{n_pat} из {total}** ({n_pat/total*100:.0f}%) — **реальный паттерн** (выпуск достаточен)")
    md.append(f"- **{n_amb} из {total}** ({n_amb/total*100:.0f}%) — **неоднозначно** (10–50% дней)")
    md.append("")
    md.append("**Для цензурированных SKU:** floor-uplift (подтягивание провальных часов до среднего) обосновано.")
    md.append("**Для реального паттерна:** трогать профиль не нужно — это поведение клиента.")
    md.append("")
    md.append(f"Excel с полной детализацией: `outputs/censoring_rolling_sellthrough.xlsx`")

    md_path = Path("outputs/censoring_rolling_sellthrough.md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"MD: {md_path}")


# ---- main ----
if CACHE.exists() and (time.time() - CACHE.stat().st_mtime) < 86400:
    print(f"Using cache: {CACHE}")
    with open(CACHE, encoding="utf-8") as f:
        data = json.load(f)
else:
    run_on_vm()
    with open(CACHE, encoding="utf-8") as f:
        data = json.load(f)

build_report(data)
print("\nDONE. outputs/censoring_rolling_sellthrough.xlsx + .md")

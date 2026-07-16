"""
Анализ цензурирования часового профиля SKU.
Запускает анализ на VM, скачивает результат, строит Excel + MD.
"""
import sys, io, json, time
from pathlib import Path
from dotenv import dotenv_values
import paramiko
import pandas as pd
import numpy as np

secrets = dotenv_values(Path(".codex/prod_vm.env"))
host, user, password = secrets["PROD_VM_HOST"], secrets["PROD_VM_USER"], secrets["PROD_VM_PASSWORD"]
REMOTE_SCRIPT = Path(r"C:\Users\dns\AppData\Local\Temp\claude\C--Users-dns-Desktop-Projects-demand-forecasting-model\bba476d3-7a0b-4cf9-81ce-28411a2a5519\scratchpad\censoring_analysis_remote.py")
REMOTE_PATH = "/tmp/censoring_analysis.py"
REMOTE_JSON = "/tmp/censoring_analysis.json"
CACHE = Path("outputs/_censoring_analysis_cache.json")

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
    df = pd.DataFrame(data["signals"])
    counts = data["classification_counts"]
    has_prod = data.get("has_production", False)

    LABEL_COLOR = {"CENSORED": "#FFC7CE", "AMBIGUOUS": "#FFEB9C", "PATTERN": "#C6EFCE"}
    LABEL_RU = {"CENSORED": "Цензурирован", "AMBIGUOUS": "Неоднозначно", "PATTERN": "Реальный паттерн"}

    xl_path = "outputs/censoring_analysis.xlsx"
    with pd.ExcelWriter(xl_path, engine="xlsxwriter") as writer:
        wb = writer.book

        hdr = wb.add_format({"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1})
        cens_fmt = wb.add_format({"bg_color": "#FFC7CE", "border": 1})
        ambig_fmt = wb.add_format({"bg_color": "#FFEB9C", "border": 1})
        pat_fmt = wb.add_format({"bg_color": "#C6EFCE", "border": 1})
        pct_fmt = wb.add_format({"num_format": "0%", "border": 1})
        num_fmt = wb.add_format({"num_format": "0.00", "border": 1})
        base_fmt = wb.add_format({"border": 1})
        label_fmts = {"CENSORED": cens_fmt, "AMBIGUOUS": ambig_fmt, "PATTERN": pat_fmt}

        # --- Sheet 1: Общая сводка ---
        df_sum = df.groupby(["category", "label"]).size().unstack(fill_value=0).reset_index()
        df_sum["label_ru"] = df_sum.get("CENSORED", 0) * 0  # placeholder
        for lbl in ["CENSORED", "AMBIGUOUS", "PATTERN"]:
            if lbl not in df_sum.columns:
                df_sum[lbl] = 0
        df_sum["total"] = df_sum[["CENSORED","AMBIGUOUS","PATTERN"]].sum(axis=1)
        df_sum["censored_pct"] = (df_sum["CENSORED"] / df_sum["total"]).round(2)
        df_sum.to_excel(writer, sheet_name="1_Сводка", index=False)
        ws = writer.sheets["1_Сводка"]
        ws.set_column("A:A", 22); ws.set_column("B:G", 14)

        # --- Sheet 2: Все SKU × пекарня ---
        display_cols = [
            "bakery_id","product_name","category","label","score",
            "sell_through","zero_rate_in_dips","cv_dip_hours","cv_normal_hours",
            "corr_dip_quartile","dip_hour_mean","dip_hour_std","dip_freq","reasons"
        ]
        df_out = df[[c for c in display_cols if c in df.columns]].copy()
        df_out["label"] = df_out["label"].map(LABEL_RU)
        df_out.sort_values(["label","score"], ascending=[True, False], inplace=True)
        df_out.to_excel(writer, sheet_name="2_Детализация", index=False)
        ws = writer.sheets["2_Детализация"]
        ws.set_column("A:A", 10); ws.set_column("B:B", 28); ws.set_column("C:C", 18)
        ws.set_column("D:D", 16); ws.set_column("E:N", 14)

        # --- Sheet 3: только CENSORED ---
        df_cens = df[df["label"] == "CENSORED"].sort_values("score", ascending=False)
        if len(df_cens):
            df_cens_out = df_cens[[c for c in display_cols if c in df_cens.columns]].copy()
            df_cens_out["label"] = df_cens_out["label"].map(LABEL_RU)
            df_cens_out.to_excel(writer, sheet_name="3_Цензурированные", index=False)
            ws = writer.sheets["3_Цензурированные"]
            ws.set_column("A:A", 10); ws.set_column("B:B", 28); ws.set_column("C:N", 14)

        # --- Sheet 4: только PATTERN ---
        df_pat = df[df["label"] == "PATTERN"].sort_values("score")
        if len(df_pat):
            df_pat_out = df_pat[[c for c in display_cols if c in df_pat.columns]].copy()
            df_pat_out["label"] = df_pat_out["label"].map(LABEL_RU)
            df_pat_out.to_excel(writer, sheet_name="4_Реальный_паттерн", index=False)

        # --- Sheet 5: сигналы по категориям ---
        sig_agg = df.groupby("category").agg(
            n_sku_bakery=("bakery_id","count"),
            censored=("label", lambda x: (x=="CENSORED").sum()),
            pattern=("label", lambda x: (x=="PATTERN").sum()),
            ambiguous=("label", lambda x: (x=="AMBIGUOUS").sum()),
            avg_sell_through=("sell_through", lambda x: x.dropna().mean()),
            avg_zero_rate=("zero_rate_in_dips", lambda x: x.dropna().mean()),
            avg_cv_dip=("cv_dip_hours", lambda x: x.dropna().mean()),
            avg_dip_freq=("dip_freq", "mean"),
        ).round(3).reset_index()
        sig_agg["censored_pct"] = (sig_agg["censored"] / sig_agg["n_sku_bakery"]).round(2)
        sig_agg.to_excel(writer, sheet_name="5_По_категориям", index=False)
        ws = writer.sheets["5_По_категориям"]
        ws.set_column("A:A", 22); ws.set_column("B:K", 14)

    print(f"Excel: {xl_path}")

    # ---- MD report ----
    md = []
    md.append("# Анализ цензурирования часового профиля SKU")
    md.append("")
    md.append("**Период:** май–июнь 2026  |  **Пекарни:** 10 пилотных  |  **Категории:** топ-6")
    if not has_prod:
        md.append("\n> ⚠️ Данные о выпуске (`fct_production_release`) недоступны — сигнал sell-through не использован.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Итог классификации")
    md.append("")
    total = sum(counts.values())
    md.append("| Класс | SKU × пекарня | % |")
    md.append("|---|---|---|")
    for lbl, ru in LABEL_RU.items():
        n = counts.get(lbl, 0)
        md.append(f"| **{ru}** | {n} | {n/total*100:.0f}% |")
    md.append(f"| Всего | {total} | 100% |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## По категориям")
    md.append("")
    md.append("| Категория | Всего | Цензур. | Паттерн | Неод. | Цензур.% | avg sell-through | avg zero-rate |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in sig_agg.sort_values("censored_pct", ascending=False).iterrows():
        st = f"{r['avg_sell_through']:.0%}" if not pd.isna(r['avg_sell_through']) else "—"
        zr = f"{r['avg_zero_rate']:.0%}" if not pd.isna(r['avg_zero_rate']) else "—"
        md.append(f"| {r['category']} | {r['n_sku_bakery']} | {r['censored']} | {r['pattern']} | {r['ambiguous']} | {r['censored_pct']:.0%} | {st} | {zr} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Сигналы цензурирования")
    md.append("")
    md.append("| Сигнал | Что означает | Порог |")
    md.append("|---|---|---|")
    md.append("| **Sell-through** | продано/выпущено ≥95% → спрос цензурирован | ≥95% → +2 балла |")
    md.append("| **Zero-rate в дипах** | % нулей в провальные часы > 30% | >30% → +1 балл |")
    md.append("| **CV дип / CV норма** | нестабильные дипы vs стабильные нормальные часы | ratio >1.5 → +1 балл |")
    md.append("| **Корр. дип × квартиль** | дип глубже в высокотраффик → выкуп | >0.3 → +1 балл |")
    md.append("| **Std часа дипа** | час дипа гуляет → случайный выкуп | std >2.5 → +1 балл |")
    md.append("")
    md.append("**Итоговая классификация:** ≥2 баллов → CENSORED | ≤−1 → PATTERN | иначе → AMBIGUOUS")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Вывод")
    md.append("")
    n_cens = counts.get("CENSORED", 0)
    n_pat = counts.get("PATTERN", 0)
    md.append(f"- **{n_cens} из {total}** ({n_cens/total*100:.0f}%) SKU × пекарня классифицированы как цензурированные")
    md.append(f"- **{n_pat} из {total}** ({n_pat/total*100:.0f}%) — реальный поведенческий паттерн")
    md.append("")
    md.append("Для цензурированных SKU применение floor-uplift (подтягивание провальных часов до среднего) **обосновано**.")
    md.append("Для реального паттерна — трогать профиль **не нужно**, модель уже знает это поведение.")
    md.append("")
    md.append(f"Excel с полной детализацией: `outputs/censoring_analysis.xlsx`")

    md_path = Path("outputs/censoring_analysis.md")
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
print("\nDONE. outputs/censoring_analysis.xlsx + .md")

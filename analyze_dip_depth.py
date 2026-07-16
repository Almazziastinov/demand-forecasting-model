"""
Сравнение глубины дипов часового профиля: CENSORED vs PATTERN SKU.
Запускает анализ на VM, скачивает результат, строит Excel + MD.
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
REMOTE_SCRIPT = SCRATCHPAD / "dip_depth_remote.py"
REMOTE_PATH = "/tmp/dip_depth_remote.py"
REMOTE_JSON = "/tmp/dip_depth.json"
CACHE = Path("outputs/_dip_depth_cache.json")


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
    df_stats = pd.DataFrame(data["group_stats"])
    df_hour  = pd.DataFrame(data["hour_comparison"])

    for col in ["dip_depth", "relative_dip", "cv"]:
        if col in df_stats.columns:
            df_stats[col] = pd.to_numeric(df_stats[col], errors="coerce")

    LABEL_RU = {"CENSORED": "Цензурирован", "AMBIGUOUS": "Неоднозначно", "PATTERN": "Реальный паттерн"}

    xl_path = "outputs/dip_depth_comparison.xlsx"
    with pd.ExcelWriter(xl_path, engine="xlsxwriter") as writer:
        wb = writer.book
        hdr_fmt  = wb.add_format({"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1})
        cens_fmt = wb.add_format({"bg_color": "#FFC7CE", "border": 1})
        pat_fmt  = wb.add_format({"bg_color": "#C6EFCE", "border": 1})
        amb_fmt  = wb.add_format({"bg_color": "#FFEB9C", "border": 1})
        num_fmt  = wb.add_format({"num_format": "0.000", "border": 1})
        base_fmt = wb.add_format({"border": 1})

        # Sheet 1: Group summary
        grp = df_stats.groupby("label")[["dip_depth","relative_dip","cv"]].agg(["mean","median","std"]).round(3)
        grp.columns = ["_".join(c) for c in grp.columns]
        grp = grp.reset_index()
        grp.to_excel(writer, sheet_name="1_Сравнение_групп", index=False)
        ws = writer.sheets["1_Сравнение_групп"]
        ws.set_column("A:A", 16); ws.set_column("B:J", 14)

        # Sheet 2: Hourly profile comparison
        if "hour" in df_hour.columns:
            df_hour.to_excel(writer, sheet_name="2_По_часам", index=False)
            ws = writer.sheets["2_По_часам"]
            ws.set_column("A:C", 14)

        # Sheet 3: CENSORED details
        df_cens = df_stats[df_stats["label"]=="CENSORED"].sort_values("dip_depth", ascending=False)
        if len(df_cens):
            df_cens[["bakery_id","product_name","category_name","dip_depth","relative_dip","cv","n_hours"]].to_excel(
                writer, sheet_name="3_CENSORED", index=False)
            ws = writer.sheets["3_CENSORED"]
            ws.set_column("A:A", 10); ws.set_column("B:B", 30); ws.set_column("C:G", 14)

        # Sheet 4: PATTERN details
        df_pat = df_stats[df_stats["label"]=="PATTERN"].sort_values("dip_depth", ascending=False)
        if len(df_pat):
            df_pat[["bakery_id","product_name","category_name","dip_depth","relative_dip","cv","n_hours"]].to_excel(
                writer, sheet_name="4_PATTERN", index=False)
            ws = writer.sheets["4_PATTERN"]
            ws.set_column("A:A", 10); ws.set_column("B:B", 30); ws.set_column("C:G", 14)

    print(f"Excel: {xl_path}")

    # ---- MD ----
    grp_md = df_stats.groupby("label")[["dip_depth","relative_dip","cv"]].agg(["mean","median"]).round(3)

    md = []
    md.append("# Сравнение глубины дипов: CENSORED vs PATTERN")
    md.append("")
    md.append("**Метрики профиля:**")
    md.append("- `dip_depth` = (avg_share − min_share) / avg_share — насколько глубок дип относительно среднего")
    md.append("- `relative_dip` = min_share / avg_share — минимальная доля как % от средней (ниже = глубже дип)")
    md.append("- `cv` = std / mean почасовых долей — общая вариативность профиля")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Сравнение групп")
    md.append("")
    md.append("| Группа | dip_depth mean | dip_depth median | relative_dip mean | cv mean |")
    md.append("|---|---|---|---|---|")
    for lbl in ["CENSORED", "PATTERN", "AMBIGUOUS"]:
        sub = df_stats[df_stats["label"]==lbl]
        if len(sub) == 0: continue
        dd_m  = sub["dip_depth"].mean()
        dd_med= sub["dip_depth"].median()
        rd_m  = sub["relative_dip"].mean()
        cv_m  = sub["cv"].mean()
        ru = LABEL_RU.get(lbl, lbl)
        md.append(f"| **{ru}** ({len(sub)}) | {dd_m:.3f} | {dd_med:.3f} | {rd_m:.3f} | {cv_m:.3f} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Почасовой профиль: CENSORED vs PATTERN")
    md.append("")
    if "hour" in df_hour.columns:
        md.append("| Час | CENSORED avg share | PATTERN avg share | Разница |")
        md.append("|---|---|---|---|")
        for _, r in df_hour.sort_values("hour").iterrows():
            c_val = r.get("CENSORED", np.nan)
            p_val = r.get("PATTERN", np.nan)
            diff = c_val - p_val if not (pd.isna(c_val) or pd.isna(p_val)) else np.nan
            diff_str = f"{diff:+.4f}" if not pd.isna(diff) else "—"
            md.append(f"| {int(r['hour'])}:00 | {c_val:.4f} | {p_val:.4f} | {diff_str} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Вывод")
    md.append("")
    cens_sub = df_stats[df_stats["label"]=="CENSORED"]
    pat_sub  = df_stats[df_stats["label"]=="PATTERN"]
    cens_dd = cens_sub["dip_depth"].mean()
    pat_dd  = pat_sub["dip_depth"].mean()
    cens_rd = cens_sub["relative_dip"].mean()
    pat_rd  = pat_sub["relative_dip"].mean()
    if not (pd.isna(cens_dd) or pd.isna(pat_dd)):
        diff_pct = (cens_dd - pat_dd) / pat_dd * 100 if pat_dd > 0 else 0
        md.append(f"- CENSORED SKU имеют **dip_depth = {cens_dd:.3f}** vs PATTERN **{pat_dd:.3f}** (разница {diff_pct:+.0f}%)")
        md.append(f"- Минимальная доля (relative_dip): CENSORED **{cens_rd:.3f}** vs PATTERN **{pat_rd:.3f}**")
        if cens_dd > pat_dd * 1.1:
            md.append("")
            md.append("**Floor-uplift обоснован:** у цензурированных SKU дипы значительно глубже — подтягивание до среднего исправит заниженный прогноз в провальные часы.")
        else:
            md.append("")
            md.append("**Floor-uplift слабо обоснован:** разница в глубине дипов между группами незначительна — эффект от применения будет минимальным.")
    md.append("")
    md.append(f"Excel с детализацией: `outputs/dip_depth_comparison.xlsx`")

    md_path = Path("outputs/dip_depth_comparison.md")
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
print("\nDONE. outputs/dip_depth_comparison.xlsx + .md")

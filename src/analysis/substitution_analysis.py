"""
Analiz substitutsii: perelivaetsya li spros s zakonchivshikhsya tovarov
na ostavshiesya v toj zhe pekarni?

Gipoteza: kogda tovar A zakonchilsya (Ostatok=0), prodazhi tovarov B,C,D
v toj zhe pekarni rastut vyshe ikh srednego urovnya.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd

DATA_PATH = "data/processed/preprocessed_data_3month_enriched.csv"


def main():
    print("=" * 70)
    print("  ANALIZ SUBSTITUTSII (peretok sprosa mezhdu nomenklaturami)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df["Дата"] = pd.to_datetime(df["Дата"])

    print(f"\n  Strok: {len(df):,}")
    print(f"  Pekarni: {df['Пекарня'].nunique()}")
    print(f"  Nomenklatury: {df['Номенклатура'].nunique()}")
    print(f"  Dnej: {df['Дата'].nunique()}")

    # ========================================
    # 1. Srednie prodazhi po (pekarnya, nomenklatura) -- "norma"
    # ========================================
    print("\n--- 1. Raschet 'normy' prodazh po (pekarnya, nomenklatura) ---")
    avg_sales = df.groupby(["Пекарня", "Номенклатура"])["Продано"].mean()
    avg_sales.name = "avg_sales"
    df = df.merge(avg_sales, on=["Пекарня", "Номенклатура"], how="left")

    # Otkloneniye ot normy
    df["sales_vs_avg"] = df["Продано"] - df["avg_sales"]
    df["sales_ratio"] = df["Продано"] / (df["avg_sales"] + 0.1)

    # ========================================
    # 2. Stock-out metriki po (pekarnya, data)
    # ========================================
    print("--- 2. Podschet stock-out po pekarnyam/dnyam ---")

    # Kol-vo tovarov s Ostatok=0 v kazhdoj pekarni v kazhdyj den'
    stockout_flag = (df["Остаток"] == 0).astype(int)
    df["is_stockout"] = stockout_flag

    bakery_day = df.groupby(["Пекарня", "Дата"]).agg(
        total_items=("Номенклатура", "nunique"),
        stockout_items=("is_stockout", "sum"),
    ).reset_index()
    bakery_day["stockout_pct"] = bakery_day["stockout_items"] / bakery_day["total_items"] * 100

    print(f"  Srednee stock-out tovarov na pekarnu/den': "
          f"{bakery_day['stockout_items'].mean():.1f} iz {bakery_day['total_items'].mean():.1f}")
    print(f"  Srednij % stock-out: {bakery_day['stockout_pct'].mean():.1f}%")

    # ========================================
    # 3. Kol-vo stock-out DRUGIKH tovarov (ne sebya)
    # ========================================
    print("--- 3. Schitaem davlenie substitutsii ---")

    # Dlya kazhdoj stroki: skol'ko drugikh tovarov zakončilis' v etoj pekarni v etot den'
    df = df.merge(
        bakery_day[["Пекарня", "Дата", "stockout_items"]],
        on=["Пекарня", "Дата"], how="left"
    )
    # Vychitaem svoij stockout
    df["other_stockouts"] = df["stockout_items"] - df["is_stockout"]

    # To zhe po kategorii
    stockout_by_cat = df.groupby(["Пекарня", "Дата", "Категория"])["is_stockout"].sum().reset_index()
    stockout_by_cat.columns = ["Пекарня", "Дата", "Категория", "stockout_same_cat"]
    df = df.merge(stockout_by_cat, on=["Пекарня", "Дата", "Категория"], how="left")
    df["other_stockouts_same_cat"] = df["stockout_same_cat"] - df["is_stockout"]

    # ========================================
    # 4. GLAVNYJ TEST: prodazhi vs norma v zavisimosti ot stockout drugikh
    # ========================================
    print("\n--- 4. GLAVNYJ TEST ---")
    print("  Prodazhi tovara otnositel'no ego normy, v zavisimosti ot togo")
    print("  skol'ko DRUGIKH tovarov zakonchilos' v toj zhe pekarni\n")

    # Tol'ko tovary kotorye SAMI NE zakonchilis' (chtob ne smeshivat')
    available = df[df["is_stockout"] == 0].copy()
    print(f"  Strok s Ostatok > 0 (dostupnye tovary): {len(available):,}")

    # Buckety po kol-vu stockout drugikh
    bins = [0, 0.5, 5, 10, 15, 100]
    labels = ["0", "1-5", "6-10", "11-15", "16+"]
    available["stockout_bucket"] = pd.cut(
        available["other_stockouts"], bins=bins, labels=labels, right=True
    )

    result = available.groupby("stockout_bucket", observed=True).agg(
        count=("Продано", "size"),
        avg_prodano=("Продано", "mean"),
        avg_normal=("avg_sales", "mean"),
        avg_delta=("sales_vs_avg", "mean"),
        avg_ratio=("sales_ratio", "mean"),
    )

    print(f"  {'Stockout drug.':<16} {'N':>8} {'Prodano':>9} {'Norma':>9} "
          f"{'Delta':>9} {'Ratio':>7}")
    print(f"  {'-' * 60}")
    for idx, row in result.iterrows():
        print(f"  {str(idx):<16} {row['count']:>8,.0f} {row['avg_prodano']:>9.2f} "
              f"{row['avg_normal']:>9.2f} {row['avg_delta']:>+9.3f} {row['avg_ratio']:>7.3f}")

    # ========================================
    # 5. TO ZHE, NO PO KATEGORII (same-cat stockout)
    # ========================================
    print("\n--- 5. Substitutsiya VNUTRI KATEGORII ---")
    print("  Prodazhi tovara kogda drugie v TOJ ZHE kategorii zakonchilis'\n")

    bins_cat = [-0.5, 0.5, 2, 5, 100]
    labels_cat = ["0", "1-2", "3-5", "6+"]
    available["stockout_cat_bucket"] = pd.cut(
        available["other_stockouts_same_cat"], bins=bins_cat, labels=labels_cat, right=True
    )

    result_cat = available.groupby("stockout_cat_bucket", observed=True).agg(
        count=("Продано", "size"),
        avg_prodano=("Продано", "mean"),
        avg_normal=("avg_sales", "mean"),
        avg_delta=("sales_vs_avg", "mean"),
        avg_ratio=("sales_ratio", "mean"),
    )

    print(f"  {'Stockout same cat':<18} {'N':>8} {'Prodano':>9} {'Norma':>9} "
          f"{'Delta':>9} {'Ratio':>7}")
    print(f"  {'-' * 62}")
    for idx, row in result_cat.iterrows():
        print(f"  {str(idx):<18} {row['count']:>8,.0f} {row['avg_prodano']:>9.2f} "
              f"{row['avg_normal']:>9.2f} {row['avg_delta']:>+9.3f} {row['avg_ratio']:>7.3f}")

    # ========================================
    # 6. TEST PO KATEGORIYAM OTDEL'NO
    # ========================================
    print("\n--- 6. Effekt po kategoriyam ---")
    print("  Delta prodazh (vs norma) pri 0 vs 5+ stockout drugikh v pekarni\n")

    for cat in sorted(df["Категория"].unique()):
        cat_data = available[available["Категория"] == cat]
        low = cat_data[cat_data["other_stockouts"] <= 0]["sales_vs_avg"].mean()
        high = cat_data[cat_data["other_stockouts"] >= 5]["sales_vs_avg"].mean()
        n_low = (cat_data["other_stockouts"] <= 0).sum()
        n_high = (cat_data["other_stockouts"] >= 5).sum()
        diff = high - low if not (np.isnan(high) or np.isnan(low)) else 0
        print(f"  {str(cat):<25} 0 stockout: {low:>+7.3f} (n={n_low:,})  "
              f"5+ stockout: {high:>+7.3f} (n={n_high:,})  "
              f"raznitsa: {diff:>+7.3f}")

    # ========================================
    # 7. Korrelyatsiya
    # ========================================
    print("\n--- 7. Korrelyatsiya ---")
    corr_all = available["sales_vs_avg"].corr(available["other_stockouts"])
    corr_cat = available["sales_vs_avg"].corr(available["other_stockouts_same_cat"])
    print(f"  corr(sales_delta, other_stockouts):          {corr_all:.4f}")
    print(f"  corr(sales_delta, other_stockouts_same_cat): {corr_cat:.4f}")

    # To zhe no tol'ko dlya vysokosprosa
    high_demand = available[available["avg_sales"] >= 10]
    if len(high_demand) > 0:
        corr_h = high_demand["sales_vs_avg"].corr(high_demand["other_stockouts"])
        corr_hc = high_demand["sales_vs_avg"].corr(high_demand["other_stockouts_same_cat"])
        print(f"  corr (tol'ko high-demand, avg>=10):          {corr_h:.4f}")
        print(f"  corr same_cat (tol'ko high-demand):           {corr_hc:.4f}")

    print("\nGotovo!")


if __name__ == "__main__":
    main()

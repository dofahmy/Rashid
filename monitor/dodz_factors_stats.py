"""DODZ statistics by exact simultaneous-factor combination.

Run: python -m monitor.dodz_factors_stats --output-dir /tmp/dodz-factor-stats
Reads the existing PostgreSQL dodz_hourly_trades table; does not rescan candles.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from sqlalchemy import text
from core import database

TABLE = "dodz_hourly_trades"


def factor_key(value: str | None) -> tuple[int, ...]:
    """Normalize '4,2' and '2,4' into one exact combination."""
    if not value or not str(value).strip():
        return ()
    try:
        return tuple(sorted(set(int(part.strip()) for part in str(value).split(",") if part.strip())))
    except ValueError:
        return ()


def run(output_dir: str) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with database()() as session:
        rows = session.execute(text(f"""
            SELECT factors, squaring_factor, status, exit_reason
            FROM {TABLE}
        """)).mappings()

        groups = {}
        total = 0
        for row in rows:
            key = factor_key(row["factors"])
            if not key:
                key = (int(row["squaring_factor"]),)
            data = groups.setdefault(key, {"total": 0, "target": 0, "time": 0, "open": 0, "other": 0})
            data["total"] += 1
            total += 1
            if row["exit_reason"] == "TARGET_3":
                data["target"] += 1
            elif row["exit_reason"] == "TIME":
                data["time"] += 1
            elif row["status"] == "OPEN":
                data["open"] += 1
            else:
                data["other"] += 1

    columns = ["العوامل المتزامنة", "عدد التوصيات", "أغلق عند هدف 3%", "نسبة تحقيق الهدف %", "إغلاق زمني", "مفتوحة", "أخرى"]
    details = []
    for key, stat in sorted(groups.items(), key=lambda item: (-item[1]["total"], item[0])):
        details.append({
            "العوامل المتزامنة": ",".join(map(str, key)),
            "عدد التوصيات": stat["total"],
            "أغلق عند هدف 3%": stat["target"],
            "نسبة تحقيق الهدف %": round(100 * stat["target"] / stat["total"], 2),
            "إغلاق زمني": stat["time"],
            "مفتوحة": stat["open"],
            "أخرى": stat["other"],
        })

    filepath = output / "dodz_simultaneous_factors_stats.csv"
    with filepath.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(details)

    target_total = sum(s["target"] for s in groups.values())
    time_total = sum(s["time"] for s in groups.values())
    open_total = sum(s["open"] for s in groups.values())
    print("DODZ — إحصائيات مجموعات العوامل المتزامنة (كل مجموعة مستقلة)")
    print(f"إجمالي الإشارات: {total:,} | حققت +3%: {target_total:,} | إغلاق زمني: {time_total:,} | مفتوحة: {open_total:,}")
    print(f"{'Factors':>17} {'Total':>9} {'Target 3%':>11} {'Rate':>9} {'Time':>9} {'Open':>9}")
    for row in details:
        print(f"{row['العوامل المتزامنة']:>17} {row['عدد التوصيات']:>9} {row['أغلق عند هدف 3%']:>11} {row['نسبة تحقيق الهدف %']:>8.2f}% {row['إغلاق زمني']:>9} {row['مفتوحة']:>9}")
    print(f"CSV: {filepath}")
    return {"total": total, "target": target_total, "groups": len(details), "csv": str(filepath)}


def main() -> None:
    parser = argparse.ArgumentParser(description="DODZ factor-combination statistics from PostgreSQL")
    parser.add_argument("--output-dir", default="/tmp/dodz-factor-stats")
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()

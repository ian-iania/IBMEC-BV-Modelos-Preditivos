#!/usr/bin/env python3
"""Generate synthetic FP&A datasets for Aula 06 warm-up.

Outputs:
- data/bv_originacao_auto_sintetico.csv
- data/bv_originacao_auto_sintetico.xlsx

The script is dependency-free (stdlib only) so it runs offline.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import random
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def month_starts(start_year: int, start_month: int, periods: int) -> list[dt.date]:
    items: list[dt.date] = []
    y, m = start_year, start_month
    for _ in range(periods):
        items.append(dt.date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return items


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_rows(seed: int = 42, periods: int = 60) -> list[dict[str, float | int | str]]:
    rng = random.Random(seed)
    months = month_starts(2020, 1, periods)

    rows: list[dict[str, float | int | str]] = []
    for t, month in enumerate(months):
        selic = 4.5 + 0.08 * t + 2.0 * math.sin(2 * math.pi * t / 18) + rng.gauss(0, 0.35)
        selic = clamp(selic, 2.0, 16.0)

        desemprego = 9.0 + 0.02 * t + 0.8 * math.sin(2 * math.pi * (t + 4) / 24) + rng.gauss(0, 0.25)
        desemprego = clamp(desemprego, 6.0, 14.0)

        share_digital = 0.18 + 0.004 * t + 0.03 * math.sin(2 * math.pi * t / 12) + rng.gauss(0, 0.01)
        share_digital = clamp(share_digital, 0.15, 0.55)

        season = 0.9 + 0.08 * math.sin(2 * math.pi * t / 12) + 0.05 * math.sin(2 * math.pi * t / 6)
        base = 520 + 3.2 * t
        origin = (base * season) * (1 - 0.015 * (selic - 8)) * (1 - 0.01 * (desemprego - 9)) * (1 + 0.35 * (share_digital - 0.25))
        origin += rng.gauss(0, 18)
        origin = max(origin, 300)

        ticket = 42000 + 120 * t + 3500 * math.sin(2 * math.pi * (t + 2) / 18) + rng.gauss(0, 900)
        ticket = clamp(ticket, 28000, 75000)

        contratos = int((origin * 1_000_000) / ticket)
        contratos = max(contratos, 4000)

        inad_30d = 0.028 + 0.0025 * (selic / 10) + 0.0015 * (desemprego / 10) - 0.01 * (share_digital - 0.25) + rng.gauss(0, 0.0012)
        inad_30d = clamp(inad_30d, 0.018, 0.065)

        rows.append(
            {
                "mes": month.strftime("%Y-%m"),
                "originacao_auto_m": round(origin, 1),
                "contratos_qtd": contratos,
                "ticket_medio": int(round(ticket, 0)),
                "selic_aa": round(selic, 2),
                "desemprego": round(desemprego, 2),
                "share_digital": round(share_digital, 3),
                "inad_30d": round(inad_30d, 4),
            }
        )

    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    headers = [
        "mes",
        "originacao_auto_m",
        "contratos_qtd",
        "ticket_medio",
        "selic_aa",
        "desemprego",
        "share_digital",
        "inad_30d",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _cell_ref(col_idx: int, row_idx: int) -> str:
    letters = ""
    n = col_idx
    while True:
        n, r = divmod(n, 26)
        letters = chr(ord("A") + r) + letters
        if n == 0:
            break
        n -= 1
    return f"{letters}{row_idx}"


def write_xlsx(rows: list[dict[str, float | int | str]], path: Path) -> None:
    headers = [
        "mes",
        "originacao_auto_m",
        "contratos_qtd",
        "ticket_medio",
        "selic_aa",
        "desemprego",
        "share_digital",
        "inad_30d",
    ]

    table: list[list[str | float | int]] = [headers]
    for item in rows:
        table.append([item[h] for h in headers])

    rows_xml: list[str] = []
    for r_idx, row in enumerate(table, start=1):
        cells: list[str] = []
        for c_idx, value in enumerate(row):
            ref = _cell_ref(c_idx, r_idx)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        rows_xml.append(f"<row r=\"{r_idx}\">{''.join(cells)}</row>")

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(rows_xml)}</sheetData>"
        "</worksheet>"
    )

    content_types = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">
  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>
  <Default Extension=\"xml\" ContentType=\"application/xml\"/>
  <Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>
  <Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>
  <Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>
</Types>
""".strip()

    rels = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>
</Relationships>
""".strip()

    workbook = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">
  <sheets>
    <sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\"/>
  </sheets>
</workbook>
""".strip()

    wb_rels = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>
  <Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>
</Relationships>
""".strip()

    styles = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">
  <fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>
  <fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills>
  <borders count=\"1\"><border/></borders>
  <cellStyleXfs count=\"1\"><xf/></cellStyleXfs>
  <cellXfs count=\"1\"><xf xfId=\"0\"/></cellXfs>
  <cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>
</styleSheet>
""".strip()

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/styles.xml", styles)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = generate_rows(seed=42, periods=60)

    csv_path = data_dir / "bv_originacao_auto_sintetico.csv"
    xlsx_path = data_dir / "bv_originacao_auto_sintetico.xlsx"

    write_csv(rows, csv_path)
    write_xlsx(rows, xlsx_path)

    print(f"CSV generated: {csv_path}")
    print(f"XLSX generated: {xlsx_path}")


if __name__ == "__main__":
    main()

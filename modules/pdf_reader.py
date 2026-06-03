import pandas as pd
import pdfplumber


def _deduplicate_columns(columns):
    seen = {}
    cleaned = []

    for index, column in enumerate(columns):
        name = str(column).strip() if column else f"column_{index + 1}"
        count = seen.get(name, 0)
        cleaned.append(name if count == 0 else f"{name}_{count + 1}")
        seen[name] = count + 1

    return cleaned


def extract_pdf_data(pdf):
    all_tables = []

    with pdfplumber.open(pdf) as file:
        for page in file.pages:
            tables = page.extract_tables() or []

            for table in tables:
                if not table or len(table) < 2:
                    continue

                header = _deduplicate_columns(table[0])
                width = len(header)
                rows = [
                    (row + [None] * width)[:width]
                    for row in table[1:]
                    if row and any(cell not in (None, "") for cell in row)
                ]

                if rows:
                    all_tables.append(pd.DataFrame(rows, columns=header))

    if not all_tables:
        return pd.DataFrame()

    return pd.concat(all_tables, ignore_index=True)

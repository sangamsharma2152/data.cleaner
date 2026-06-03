# AI Data Cleaning Pipeline

AI Data Cleaning Pipeline is a Streamlit app for cleaning, merging, and preparing tabular datasets for machine learning. It accepts CSV, Excel, and PDF table files, then produces cleaned merged data plus train/test CSV exports.

## Features

- Upload two CSV, Excel, or PDF datasets.
- Preview raw data and basic EDA summaries.
- Remove empty rows, fill missing values, remove duplicates, and filter numeric outliers.
- Match similar columns with SentenceTransformer embeddings.
- Override AI matching with manual merge column selection.
- Merge with outer, inner, left, or right join behavior.
- Generate before/after cleaning reports.
- Encode categorical columns, drop constant columns, and scale numeric columns after train/test splitting.
- Download merged, training, and testing datasets as CSV files.

## Project Structure

```text
.
├── app.py
├── modules/
│   ├── ai_engine.py
│   ├── cleaner.py
│   ├── merger.py
│   ├── pdf_reader.py
│   └── preprocessing.py
├── samples/
│   ├── customers.csv
│   └── orders.csv
├── tests/
│   ├── test_cleaner.py
│   ├── test_merger.py
│   └── test_preprocessing.py
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/sangamsharma2152/data.cleaner.git
cd data.cleaner
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Open the local Streamlit URL in your browser. Upload two datasets or use the sample files in `samples/`.

## Sample Workflow

1. Upload `samples/customers.csv` as the first dataset.
2. Upload `samples/orders.csv` as the second dataset.
3. Choose `Manual column selection`.
4. Select `customer_id` for both merge columns.
5. Start the complete pipeline.
6. Download the merged, train, or test CSV output.

## Testing

```bash
python -m unittest discover -s tests
```

## Notes and Limitations

- PDF table extraction depends on the PDF layout. CSV and Excel files are more reliable.
- AI column matching uses `all-MiniLM-L6-v2`, which may download the first time it runs.
- The app prepares features only. It does not train a model or choose a prediction target.
- Unknown categories in the test split are encoded as `-1` to avoid leakage.

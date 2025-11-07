# desktop-screenshots-dataset

## Sync environment
```bash
uv sync
```

## 1. Get most popular projects
Get first 80 pages
```bash
uv run -m src.scripts.scrape_sourceforge -p 80
```

## 2. Download screenshots
Download 10 screenshots concurrently
```bash
uv run -m src.scripts.download_screenshots --max-workers 10
```

## 3. Run notebook

Clean and upload to HF the dataset `Vokturz/sourceforge-app-screenshots`
- [src/notebooks/00_data_processing.ipynb](src/notebooks/00_data_processing.ipynb)

## 4. Validate screenshots
```bash
uv run -m src.scripts.validate_dataset
```

## 5. Get Image Analysis and OCR
```bash
# First run the analysis
uv run -m src.scripts.analyze_validated_dataset
# Then run ocr
uv run -m src.scripts.ocr_validated_dataset
```

## 6. Run notebook

Clean and upload to HF the dataset `Vokturz/sourceforge-app-screenshots-cr`
- [src/notebooks/01_check_ocr.ipynb](src/notebooks/01_check_ocr.ipynb)

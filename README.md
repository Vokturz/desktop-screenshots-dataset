# desktop-screenshots-dataset

## Sync environment
```bash
uv sync
```

## Get most popular projects
Get first 80 pages
```bash
uv run -m src.scripts.scrape_sourceforge -p 80
```

## Download screenshots
Download 10 screenshots concurrently
```bash
uv run -m src.scripts.download_screenshots --max-workers 10
```

# Data Setup

Place the raw dataset files here before training:

- `data/raw/True.csv`
- `data/raw/Fake.csv`

Expected columns from the original notebook workflow:

- `title`
- `text`
- `subject`
- `date`

The training pipeline will:

- label `True.csv` rows as `1`
- label `Fake.csv` rows as `0`
- optionally merge `title` into `text`
- optionally drop metadata columns like `subject` and `date`

Keep raw data out of version control. The `.gitignore` file already excludes `data/raw/`.


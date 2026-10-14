# Real-Time Industrial Predictive Maintenance (Software Mini Project)

Beginner-friendly project using Python, SQLite, Scikit-learn, and Streamlit.

## What it does
- Loads the AI4I 2020 Predictive Maintenance dataset.
- Trains a Random Forest classifier to estimate the probability of a machine-failure label.
- Simulates streaming rows from the dataset.
- Stores simulated readings and model predictions in SQLite.
- Displays readings, risk levels, charts, and recent alerts in a Streamlit dashboard.

**Important limitation:** AI4I 2020 is a synthetic dataset designed to reflect industrial data, not live sensor data. The label indicates failure at that data point; it does not inherently mean "failure in the next 24 hours". Treat the risk score as a learning demonstration, not a production-ready forecast.

## 1. Requirements
- Python 3.10 or newer
- Internet connection to download the dataset (or manually download it)
- No physical sensors required

## 2. Setup

Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate the environment:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 3. Dataset
Download `ai4i2020.csv` from the official UCI dataset page:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Place it at:
`data/ai4i2020.csv`

The app also attempts to download the public CSV automatically if the file is missing.

## 4. Run the dashboard
```bash
streamlit run app.py
```

The app will:
1. Load the CSV.
2. Exclude identifiers and failure-mode columns that would leak the target.
3. Train a model on a stratified train/test split.
4. Show test metrics.
5. Start a simulated stream from held-out rows.
6. Store each simulated row and its predicted risk in `maintenance.db`.

Click **Next reading** to simulate one new machine reading, or **Run 10 readings** to add several rows.

## 5. Database tables
- `readings`: sensor features, actual dataset label (for demo/evaluation), predicted risk, predicted class, and timestamp.
- `maintenance_events`: alerts generated for high-risk readings.

SQLite file: `maintenance.db` (created automatically).

## 6. Suggested tests
- Confirm the dashboard loads and shows dataset row count.
- Click Next reading and confirm the readings count increases by one.
- Run 10 readings and confirm the count increases by ten.
- Check that high-risk predictions create maintenance events.
- Compare precision, recall, F1 score, and confusion matrix; do not report accuracy alone because failure labels are rare.
- Restart the app and confirm historical rows remain in SQLite.

## 7. SQL examples
Open the database with Python's sqlite3 module or an SQLite viewer.

```sql
SELECT COUNT(*) AS total_readings FROM readings;

SELECT risk_level, COUNT(*) AS count
FROM readings
GROUP BY risk_level;

SELECT recorded_at, rotational_speed_rpm, torque_nm, tool_wear_min,
       failure_probability, risk_level
FROM readings
ORDER BY id DESC
LIMIT 20;

SELECT recorded_at, risk_level, message
FROM maintenance_events
ORDER BY id DESC;
```

## 8. Project report outline
1. Title page
2. Abstract
3. Introduction and problem statement
4. Objectives
5. Existing system and proposed system
6. Architecture and technologies
7. Dataset and attributes
8. Methodology and model training
9. Database design
10. Dashboard screenshots
11. Testing and evaluation
12. Limitations
13. Future enhancements
14. Conclusion
15. References

## 9. Dataset reference
AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository.
https://doi.org/10.24432/C5HS5C
Licensed under CC BY 4.0. Give credit to the dataset authors/source when reusing it.

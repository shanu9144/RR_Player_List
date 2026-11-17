# SupeRR Selector

SupeRR Selector ranks T20 batters and bowlers by their performance in high-pressure situations, then compares player strengths with Rajasthan Royals' phase-by-phase needs. It keeps the performance rankings separate from auction availability.

## What the project produces

- Historical batter and bowler ratings, including pressure impact, skill, consistency, sample sizes, and ranking stability.
- RR pressure-gap tables and player phase profiles.
- Separate provisional batter and bowler auction watchlists. These use recorded IPL 2026 unsold results, exclude players listed in the supplied IPL 2026 squads, and apply the Pakistan and user-requested exclusions.
- Charts, diagnostics, and a PDF methodology report.

An unsold result from IPL 2026 does not confirm that a player registered for, or will be available at, a later auction. Check the latest official retention, release, and registration lists before treating anyone as an auction target.

## Project files

- `Data Pack/` - supplied ball-by-ball, match, competition, team, squad-mapping, and playing-eleven data.
- `pressure/` - data preparation, models, metrics, RR fit, auction screening, and charts.
- `auction_bat_market_evidence.csv` and `auction_bowl_market_evidence.csv` - sourced IPL 2026 auction-status evidence used by the separate watchlists.
- `run_pipeline.py` - builds ratings, diagnostics, phase profiles, and watchlists in `outputs/`.
- `make_report.py` - builds `outputs/SupeRR_Selector_R1_Methodology_Report.pdf` from pipeline outputs.
- `check_config.py` - evaluates sensitivity to key model settings.
- `analysis.ipynb` - a step-by-step notebook for exploring the analysis.
- `outputs/` - generated CSVs, charts, run log, and PDF report.

## Setup

Use Python 3.10 or later. From the project folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Run the analysis

```powershell
python run_pipeline.py
python make_report.py
```

The pipeline builds `outputs/cache_scored_balls.pkl` to speed up later runs. It is derived data and can be deleted; the pipeline recreates it when needed. To rebuild that cache and the win-probability scores from scratch, use:

```powershell
python run_pipeline.py --refresh
```

Run the optional configuration checks with:

```powershell
python check_config.py
```

The supplied `Data Pack/` files and both auction evidence CSVs must remain in place for the pipeline to run.

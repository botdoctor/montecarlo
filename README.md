# Music Catalog Revenue Simulator

A Monte Carlo simulation engine built in Python for projecting streaming revenue and evaluating investment performance of music catalogs. Developed for use in private equity music rights valuation.

## Overview

Music catalogs are increasingly popular alternative assets in private equity. Valuing them requires forecasting future streaming revenue across songs of varying ages, genres, and historical performance — all of which carry significant uncertainty. This tool uses Monte Carlo methods to model that uncertainty and produce probabilistic revenue projections, giving investors a data-driven view of potential outcomes rather than a single-point estimate.

Given a portfolio of songs grouped by release age and their current streaming volumes, the simulator runs thousands of iterations with randomized year-over-year growth rates, age-based decay factors, and market-level multipliers to generate a distribution of possible revenue outcomes over a configurable projection horizon.

## How It Works

1. **Input Configuration** — A JSON file defines the simulation parameters: number of iterations, projection end year, historical YOY growth rates (mean and standard deviation) by song age, market multipliers by calendar year, song age groups with current streaming volumes, and financial parameters (initial investment, revenue per stream, discount rate, target ROI).
1. **Stochastic Growth Modeling** — For each trial, the simulation projects streaming volumes forward year by year. At each step, growth is sampled from a normal distribution parameterized by the song’s age, then adjusted by an age-based decay multiplier (with its own random variation) and a fixed market multiplier for the calendar year.
1. **Revenue Computation** — Projected streams are converted to revenue using a configurable per-stream rate. Cumulative revenue is tracked across all song age groups and all simulation years.
1. **Statistical Analysis** — The simulator computes confidence intervals at every 10% band (90% CI through 10% CI) for cumulative revenue, along with mean and median projections across all trials.
1. **Investment Metrics** — Net Present Value (NPV) is calculated per trial using the specified discount rate. The tool also computes ROI distributions, probability of breaking even, and probability of achieving a target ROI threshold.

## Output

- **Excel Report** (`revenue_simulation.xlsx`) — Two sheets:
  - *Revenue Simulation*: Mean yearly revenue by song age group, plus confidence intervals for total cumulative revenue at each projection year
  - *Investment Metrics*: Mean/median NPV, break-even probability, and target ROI probability
- **Visualization** (`cumulative_revenue.png`) — Line plot showing all simulation trials for total cumulative revenue over time, with mean (white dotted) and 90% confidence interval bounds (black dashed)
- **Console Output** — Detailed per-age-group and total portfolio statistics with full confidence interval breakdowns

## Input Format

The simulator reads from `inputs.json`:

```json
{
  "iterations": 1000,
  "end_year": 2035,
  "mean_yoy_by_age": {
    "0": 0.15,
    "1": 0.12,
    "2": 0.10,
    "3": 0.08,
    "5": 0.05,
    "10": 0.02
  },
  "std_yoy_by_age": {
    "0": 0.05,
    "1": 0.04,
    "2": 0.03,
    "5": 0.02
  },
  "multiplier_by_year": {},
  "song_age_groups": {
    "1": 5.2,
    "3": 12.8,
    "7": 3.1
  },
  "initial_investment": 5000000,
  "revenue_per_stream": 0.004,
  "discount_rate": 0.08,
  "target_roi": 0.15
}
```

|Parameter           |Description                                                        |
|--------------------|-------------------------------------------------------------------|
|`iterations`        |Number of Monte Carlo trials to run                                |
|`end_year`          |Final year of the projection horizon                               |
|`mean_yoy_by_age`   |Expected year-over-year streaming growth rate, keyed by song age   |
|`std_yoy_by_age`    |Standard deviation of YOY growth, keyed by song age                |
|`multiplier_by_year`|Fixed market-level adjustment factors by calendar year             |
|`song_age_groups`   |Current streaming volume (in millions) grouped by song age in years|
|`initial_investment`|Total capital deployed for the catalog acquisition                 |
|`revenue_per_stream`|Revenue earned per individual stream                               |
|`discount_rate`     |Discount rate for NPV calculation                                  |
|`target_roi`        |ROI threshold for probability analysis (e.g., 0.15 = 15%)          |

## Requirements

```
numpy
pandas
matplotlib
xlsxwriter
```

Install dependencies:

```bash
pip install numpy pandas matplotlib xlsxwriter
```

## Usage

1. Configure your simulation parameters in `inputs.json`
1. Run the simulation:

```bash
python simulation.py
```

1. Review `revenue_simulation.xlsx` and `cumulative_revenue.png` for results

## Technical Details

- **Age decay model**: Base multiplier decreases by 5% per year of song age, with Gaussian noise (σ = 0.05), floored at 0.5x to prevent unrealistic collapse
- **Growth sampling**: YOY growth rates drawn from normal distributions with age-specific parameters, allowing the model to capture both the general trend of declining growth in older catalogs and the uncertainty around that trend
- **NPV calculation**: Per-trial discounted cash flow using the configured discount rate, computed across the full projection horizon
- **Confidence intervals**: Computed at 10% increments from 10% to 90%, providing a granular view of the outcome distribution rather than just a single CI band
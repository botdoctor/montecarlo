import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
from datetime import datetime

# Get current year
CURRENT_YEAR = datetime.now().year

def get_age_multiplier(age):
    """Calculate age-based growth multiplier with random variation."""
    base_multiplier = 1.0 - 0.05 * age
    variation = np.random.normal(0, 0.05)
    return max(base_multiplier + variation, 0.5)

def read_input_file(filename):
    """Read and validate simulation inputs from JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Input file '{filename}' not found.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in input file.")

    # Validate iterations
    iterations = data.get('iterations')
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("Invalid iterations: must be a positive integer")

    # Validate end_year
    end_year = data.get('end_year')
    if not isinstance(end_year, int) or end_year <= CURRENT_YEAR:
        raise ValueError(f"Invalid end_year: must be integer > {CURRENT_YEAR}")

    # Validate mean_yoy_by_age and std_yoy_by_age
    mean_yoy_by_age = data.get('mean_yoy_by_age', {})
    std_yoy_by_age = data.get('std_yoy_by_age', {})
    # Convert keys to integers
    new_mean_yoy_by_age = {}
    new_std_yoy_by_age = {}
    max_age = end_year - (CURRENT_YEAR - 10)
    for age in range(max_age + 1):
        age_str = str(age)
        if age_str in mean_yoy_by_age:
            new_mean_yoy_by_age[age] = float(mean_yoy_by_age[age_str])
        else:
            new_mean_yoy_by_age[age] = 0.05
        if age_str in std_yoy_by_age:
            new_std_yoy_by_age[age] = float(std_yoy_by_age[age_str])
        else:
            new_std_yoy_by_age[age] = 0.02
        if not isinstance(new_mean_yoy_by_age[age], (int, float)):
            raise ValueError(f"Invalid mean_yoy for age {age}: must be numeric")
        if not isinstance(new_std_yoy_by_age[age], (int, float)) or new_std_yoy_by_age[age] < 0:
            raise ValueError(f"Invalid std_yoy for age {age}: must be numeric and non-negative")
    mean_yoy_by_age = new_mean_yoy_by_age
    std_yoy_by_age = new_std_yoy_by_age

    # Validate multiplier_by_year (fixed values per year)
    multiplier_by_year = data.get('multiplier_by_year', {})
    years = range(CURRENT_YEAR, end_year + 1)
    for year in years:
        if year not in multiplier_by_year:
            multiplier_by_year[year] = 1.0
        if not isinstance(multiplier_by_year[year], (int, float)) or multiplier_by_year[year] <= 0:
            raise ValueError(f"Invalid multiplier for year {year}: must be numeric and positive")

    # Validate song_age_groups
    song_age_groups = data.get('song_age_groups', {})
    if not song_age_groups or not isinstance(song_age_groups, dict):
        raise ValueError("Invalid or empty song_age_groups: must be a non-empty dictionary")
    new_song_age_groups = {}
    for age, streams in song_age_groups.items():
        try:
            age_int = int(age)
            if not isinstance(streams, (int, float)) or streams < 0:
                raise ValueError(f"Invalid streams for song age {age}: must be numeric and non-negative")
            # Interpret streams as millions
            new_song_age_groups[age_int] = float(streams) * 1_000_000
        except ValueError:
            raise ValueError(f"Invalid song age key: {age}")
    song_age_groups = new_song_age_groups

    # Validate initial investment
    initial_investment = data.get('initial_investment')
    if not isinstance(initial_investment, (int, float)) or initial_investment < 0:
        raise ValueError("Invalid initial_investment: must be numeric and non-negative")

    # Validate revenue per stream
    revenue_per_stream = data.get('revenue_per_stream')
    if not isinstance(revenue_per_stream, (int, float)) or revenue_per_stream <= 0:
        raise ValueError("Invalid revenue_per_stream: must be numeric and positive")

    # Validate discount rate
    discount_rate = data.get('discount_rate')
    if not isinstance(discount_rate, (int, float)) or discount_rate < 0:
        raise ValueError("Invalid discount_rate: must be numeric and non-negative")

    # Validate target ROI
    target_roi = data.get('target_roi')
    if not isinstance(target_roi, (int, float)):
        raise ValueError("Invalid target_roi: must be numeric")

    return (iterations, end_year, mean_yoy_by_age, std_yoy_by_age, multiplier_by_year, 
            song_age_groups, initial_investment, revenue_per_stream, discount_rate, target_roi)

def monte_carlo_simulation(initial_streams, release_year, mean_yoy_by_age, std_yoy_by_age, 
                          multiplier_by_year, years, trials, revenue_per_stream):
    """Run Monte Carlo simulation to compute revenue from streams."""
    print(f"Starting simulation for {trials} trials, {years + 1} years", flush=True)
    yearly_cumulative_streams = np.zeros((trials, years + 1))
    yearly_revenue = np.zeros((trials, years + 1))
    yearly_cumulative_revenue = np.zeros((trials, years + 1))
    
    for i in range(trials):
        cumulative_streams = 0
        streams = initial_streams
        cumulative_revenue = 0
        for year_offset in range(years + 1):
            sim_year = CURRENT_YEAR + year_offset
            song_age = sim_year - release_year
            if song_age < 0:
                yearly_cumulative_streams[i, year_offset] = cumulative_streams
                yearly_cumulative_revenue[i, year_offset] = cumulative_revenue
                continue
            # Initialize cumulative streams with initial streams in 2025
            if sim_year == CURRENT_YEAR:
                cumulative_streams = streams
                yearly_cumulative_streams[i, year_offset] = cumulative_streams
                # Initial streams in 2025 generate revenue in 2025
                revenue = streams * revenue_per_stream
                cumulative_revenue += revenue
                yearly_revenue[i, year_offset] = revenue
                yearly_cumulative_revenue[i, year_offset] = cumulative_revenue
                continue
            print(f"Trial {i+1}, Year {sim_year}, Song age {song_age}", flush=True)
            mean_yoy = mean_yoy_by_age.get(song_age, 0.05)
            std_yoy = std_yoy_by_age.get(song_age, 0.02)
            growth = np.random.normal(mean_yoy, std_yoy)
            print(f"  YOY Growth: {growth:.4f} (mean={mean_yoy}, std={std_yoy})", flush=True)
            age_multiplier = get_age_multiplier(song_age)
            print(f"  Age Multiplier: {age_multiplier:.4f} (base={1.0 - 0.05 * song_age})", flush=True)
            year_multiplier = multiplier_by_year.get(sim_year, 1.0)
            print(f"  Year Multiplier: {year_multiplier:.4f} (fixed)", flush=True)
            streams *= (1 + growth)
            streams *= age_multiplier
            streams *= year_multiplier
            # Ensure streams is non-negative
            streams = max(streams, 0)
            cumulative_streams += streams
            # Ensure cumulative streams is non-decreasing
            cumulative_streams = max(cumulative_streams, yearly_cumulative_streams[i, year_offset - 1])
            yearly_cumulative_streams[i, year_offset] = cumulative_streams
            # Compute incremental streams and revenue
            incremental_streams = streams
            revenue = incremental_streams * revenue_per_stream
            cumulative_revenue += revenue
            yearly_revenue[i, year_offset] = revenue
            yearly_cumulative_revenue[i, year_offset] = cumulative_revenue
    print("Simulation complete", flush=True)
    return yearly_cumulative_streams, yearly_revenue, yearly_cumulative_revenue

def analyze_results(cumulative_revenue, song_age, years, release_year, end_year):
    """Analyze and print simulation results for cumulative revenue with multiple confidence intervals."""
    final_cumulative_revenue = cumulative_revenue[:, -1]
    mean_revenue = np.mean(final_cumulative_revenue)
    median_revenue = np.median(final_cumulative_revenue)
    
    # Compute confidence intervals at 10% intervals (90% CI, 80% CI, ..., 10% CI)
    confidence_intervals = {}
    for ci in range(90, 0, -10):
        lower_percentile = (100 - ci) / 2
        upper_percentile = 100 - lower_percentile
        ci_lower = np.percentile(final_cumulative_revenue, lower_percentile)
        ci_upper = np.percentile(final_cumulative_revenue, upper_percentile)
        confidence_intervals[ci] = (ci_lower, ci_upper)
    
    print(f"\nCumulative Revenue Results for songs aged {song_age} in {CURRENT_YEAR} (released {release_year}, projected to {end_year}):", flush=True)
    print(f"Mean cumulative revenue: ${mean_revenue:.2f}", flush=True)
    print(f"Median cumulative revenue: ${median_revenue:.2f}", flush=True)
    for ci in sorted(confidence_intervals.keys(), reverse=True):
        ci_lower, ci_upper = confidence_intervals[ci]
        print(f"{ci}% Confidence Interval: ${ci_lower:.2f} to ${ci_upper:.2f}", flush=True)
    
    return cumulative_revenue

def plot_results(all_cumulative_revenue, years, end_year, trials):
    """Plot line graph of total cumulative revenue with mean and 90% CI lines."""
    print("Generating plot", flush=True)
    plt.figure(figsize=(12, 6))
    years_range = list(range(CURRENT_YEAR, end_year + 1))
    total_cumulative_revenue = np.zeros((trials, years + 1))
    for song_age, trial_results in all_cumulative_revenue.items():
        total_cumulative_revenue += trial_results
    
    # Plot individual trials without legend labels
    for i in range(trials):
        plt.plot(years_range, total_cumulative_revenue[i], alpha=0.5, label=None)  # No label for trials
    
    # Compute and plot mean line (white dotted)
    mean_revenue = np.mean(total_cumulative_revenue, axis=0)
    plt.plot(years_range, mean_revenue, color='white', linestyle='dotted', linewidth=2, label='Mean')
    
    # Compute and plot 90% CI lines (black dashed)
    ci_lower = np.percentile(total_cumulative_revenue, 5, axis=0)
    ci_upper = np.percentile(total_cumulative_revenue, 95, axis=0)
    plt.plot(years_range, ci_lower, color='black', linestyle='dashed', linewidth=1, label='90% CI Lower')
    plt.plot(years_range, ci_upper, color='black', linestyle='dashed', linewidth=1, label='90% CI Upper')
    
    # Add "Property of Courage Music" annotation
    plt.text(0.95, 0.05, 'Property of Courage Music', 
             transform=plt.gca().transAxes, 
             fontsize=10, 
             verticalalignment='bottom', 
             horizontalalignment='right', 
             color='black')
    
    plt.title(f"Total Cumulative Revenue Over Time ({trials} Trials, Projected to {end_year})")
    plt.xlabel("Calendar Year")
    plt.ylabel("Total Cumulative Revenue ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cumulative_revenue.png')
    plt.close()
    print("Plot saved as 'cumulative_revenue.png'", flush=True)

def main():
    try:
        input_file = 'inputs.json'
        print("Reading input file", flush=True)
        (iterations, end_year, mean_yoy_by_age, std_yoy_by_age, multiplier_by_year, 
         song_age_groups, initial_investment, revenue_per_stream, discount_rate, target_roi) = read_input_file(input_file)
        
        years = end_year - CURRENT_YEAR
        all_cumulative_streams = {}
        all_yearly_revenue = {}
        all_cumulative_revenue = {}
        total_final_revenue = []
        
        for song_age, initial_streams in song_age_groups.items():
            release_year = CURRENT_YEAR - int(song_age)
            print(f"\nSimulating for songs aged {song_age} in {CURRENT_YEAR} (released {release_year})", flush=True)
            cumulative_streams, yearly_revenue, cumulative_revenue = monte_carlo_simulation(
                initial_streams, release_year, mean_yoy_by_age, std_yoy_by_age, multiplier_by_year, 
                years, trials=iterations, revenue_per_stream=revenue_per_stream
            )
            all_cumulative_streams[song_age] = cumulative_streams
            all_yearly_revenue[song_age] = yearly_revenue
            all_cumulative_revenue[song_age] = cumulative_revenue
            analyzed_results = analyze_results(cumulative_revenue, song_age, years, release_year, end_year)
            total_final_revenue.append(analyzed_results[:, -1])
        
        print("\nPlotting results", flush=True)
        plot_results(all_cumulative_revenue, years, end_year, iterations)
        
        print("\nSaving Excel file", flush=True)
        # Compute mean yearly revenue per age group per year
        mean_revenue_results = {age: np.mean(results, axis=0) for age, results in all_yearly_revenue.items()}
        df_revenue_means = pd.DataFrame(mean_revenue_results, index=range(CURRENT_YEAR, end_year + 1))
        df_revenue_means.index.name = 'Year'
        
        # Compute total cumulative revenue across all age groups per trial per year
        total_cumulative_revenue = np.zeros((iterations, years + 1))
        for song_age, trial_results in all_cumulative_revenue.items():
            total_cumulative_revenue += trial_results
        
        # Compute confidence intervals for total cumulative revenue per year
        ci_data = {}
        for ci in range(90, 0, -10):
            lower_percentile = (100 - ci) / 2
            upper_percentile = 100 - lower_percentile
            ci_lower = np.percentile(total_cumulative_revenue, lower_percentile, axis=0)
            ci_upper = np.percentile(total_cumulative_revenue, upper_percentile, axis=0)
            ci_data[f"{ci}% CI Lower"] = ci_lower
            ci_data[f"{ci}% CI Upper"] = ci_upper
        
        # Create DataFrame for CIs
        df_cis = pd.DataFrame(ci_data, index=range(CURRENT_YEAR, end_year + 1))
        
        # Compute NPV per trial
        npv_per_trial = []
        for i in range(iterations):
            annual_revenues = total_cumulative_revenue[i, :]
            discounted_revenues = np.array([
                annual_revenues[t] / ((1 + discount_rate) ** t) for t in range(len(annual_revenues))
            ])
            npv = discounted_revenues.sum() - initial_investment
            npv_per_trial.append(npv)
        
        # Compute ROI per trial at the end year
        roi_per_trial = (total_cumulative_revenue[:, -1] - initial_investment) / initial_investment * 100
        
        # Compute probabilities
        prob_break_even = np.mean(total_cumulative_revenue[:, -1] >= initial_investment) * 100
        prob_target_roi = np.mean(roi_per_trial >= (target_roi * 100)) * 100
        
        # Combine means, CIs, and metrics into a single DataFrame
        metrics = {
            'Mean NPV ($)': [np.mean(npv_per_trial)],
            'Median NPV ($)': [np.median(npv_per_trial)],
            'Probability Break Even (%)': [prob_break_even],
            f'Probability ROI >= {target_roi*100}% (%)': [prob_target_roi]
        }
        df_metrics = pd.DataFrame(metrics, index=['Metrics'])
        
        # Write to Excel with formatting
        output_file = 'revenue_simulation.xlsx'
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Write revenue means and CIs
            df_combined = pd.concat([df_revenue_means, df_cis], axis=1)
            df_combined.to_excel(writer, sheet_name='Revenue Simulation', float_format="%.2f")
            
            # Write metrics on a separate sheet
            df_metrics.to_excel(writer, sheet_name='Investment Metrics', float_format="%.2f", startrow=0)
            
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            revenue_worksheet = writer.sheets['Revenue Simulation']
            metrics_worksheet = writer.sheets['Investment Metrics']
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            
            # Apply header formatting to Revenue Simulation sheet
            for col_num, value in enumerate(df_combined.columns.values):
                revenue_worksheet.write(0, col_num + 1, value, header_format)  # +1 for index column
            
            # Adjust column widths in Revenue Simulation sheet
            for column in df_combined.columns:
                column_width = max(len(str(column)) + 2, 10)  # Minimum width of 10
                col_idx = df_combined.columns.get_loc(column) + 1  # +1 for index column
                revenue_worksheet.set_column(col_idx, col_idx, column_width)
            revenue_worksheet.set_column(0, 0, 10)  # Index column width
            
            # Apply header formatting to Investment Metrics sheet
            for col_num, value in enumerate(df_metrics.columns.values):
                metrics_worksheet.write(0, col_num + 1, value, header_format)  # +1 for index column
            
            # Adjust column widths in Investment Metrics sheet
            for column in df_metrics.columns:
                column_width = max(len(str(column)) + 2, 10)  # Minimum width of 10
                col_idx = df_metrics.columns.get_loc(column) + 1  # +1 for index column
                metrics_worksheet.set_column(col_idx, col_idx, column_width)
            metrics_worksheet.set_column(0, 0, 10)  # Index column width
        
        print(f"Results saved to '{output_file}'", flush=True)
        
        # Compute confidence intervals for total revenue at the end year
        total_final_sum = np.sum(total_final_revenue, axis=0)
        total_mean = np.mean(total_final_sum)
        total_median = np.median(total_final_sum)
        confidence_intervals = {}
        for ci in range(90, 0, -10):
            lower_percentile = (100 - ci) / 2
            upper_percentile = 100 - lower_percentile
            ci_lower = np.percentile(total_final_sum, lower_percentile)
            ci_upper = np.percentile(total_final_sum, upper_percentile)
            confidence_intervals[ci] = (ci_lower, ci_upper)
        
        print(f"\nTotal Cumulative Revenue Across All Age Groups at {end_year}:", flush=True)
        print(f"Mean: ${total_mean:.2f}", flush=True)
        print(f"Median: ${total_median:.2f}", flush=True)
        for ci in sorted(confidence_intervals.keys(), reverse=True):
            ci_lower, ci_upper = confidence_intervals[ci]
            print(f"{ci}% Confidence Interval: ${ci_lower:.2f} to ${ci_upper:.2f}", flush=True)
        
        print("\nInvestment Performance Metrics:", flush=True)
        print(f"Mean NPV: ${np.mean(npv_per_trial):.2f}")
        print(f"Median NPV: ${np.median(npv_per_trial):.2f}")
        print(f"Probability of Breaking Even: {prob_break_even:.2f}%")
        print(f"Probability of ROI >= {target_roi*100}%: {prob_target_roi:.2f}%")
        
        print("\nSimulation completed successfully.", flush=True)
    except Exception as e:
        print(f"Error during simulation: {e}", flush=True)
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
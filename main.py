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
            new_song_age_groups[age_int] = float(streams)
        except ValueError:
            raise ValueError(f"Invalid song age key: {age}")
    song_age_groups = new_song_age_groups

    return iterations, end_year, mean_yoy_by_age, std_yoy_by_age, multiplier_by_year, song_age_groups

def monte_carlo_simulation(initial_streams, release_year, mean_yoy_by_age, std_yoy_by_age, multiplier_by_year, years, trials):
    """Run Monte Carlo simulation with age-dependent YOY growth and fixed year multipliers."""
    print(f"Starting simulation for {trials} trials, {years + 1} years", flush=True)
    yearly_cumulative = np.zeros((trials, years + 1))
    for i in range(trials):
        cumulative_streams = 0
        streams = initial_streams
        for year_offset in range(years + 1):
            sim_year = CURRENT_YEAR + year_offset
            song_age = sim_year - release_year
            if song_age < 0:
                yearly_cumulative[i, year_offset] = cumulative_streams
                continue
            # Initialize cumulative streams with initial streams in 2025
            if sim_year == CURRENT_YEAR:
                cumulative_streams = streams
                yearly_cumulative[i, year_offset] = cumulative_streams
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
            cumulative_streams = max(cumulative_streams, yearly_cumulative[i, year_offset - 1])
            yearly_cumulative[i, year_offset] = cumulative_streams
    print("Simulation complete", flush=True)
    return yearly_cumulative

def analyze_results(results, song_age, years, release_year, end_year):
    """Analyze and print simulation results for cumulative streams with multiple confidence intervals."""
    final_cumulative = results[:, -1]
    mean_streams = np.mean(final_cumulative)
    median_streams = np.median(final_cumulative)
    
    # Compute confidence intervals at 10% intervals (90% CI, 80% CI, ..., 10% CI)
    confidence_intervals = {}
    for ci in range(90, 0, -10):
        lower_percentile = (100 - ci) / 2
        upper_percentile = 100 - lower_percentile
        ci_lower = np.percentile(final_cumulative, lower_percentile)
        ci_upper = np.percentile(final_cumulative, upper_percentile)
        confidence_intervals[ci] = (ci_lower, ci_upper)
    
    print(f"\nCumulative Results for songs aged {song_age} in {CURRENT_YEAR} (released {release_year}, projected to {end_year}):", flush=True)
    print(f"Mean cumulative streams: {mean_streams:.2f} million", flush=True)
    print(f"Median cumulative streams: {median_streams:.2f} million", flush=True)
    for ci in sorted(confidence_intervals.keys(), reverse=True):
        ci_lower, ci_upper = confidence_intervals[ci]
        print(f"{ci}% Confidence Interval: {ci_lower:.2f} to {ci_upper:.2f} million", flush=True)
    
    return results

def plot_results(all_results, years, end_year, trials):
    """Plot line graph of total cumulative streams with mean and 90% CI lines."""
    print("Generating plot", flush=True)
    plt.figure(figsize=(12, 6))
    years_range = list(range(CURRENT_YEAR, end_year + 1))
    total_trials = np.zeros((trials, years + 1))
    for song_age, trial_results in all_results.items():
        total_trials += trial_results
    
    # Plot individual trials without legend labels
    for i in range(trials):
        plt.plot(years_range, total_trials[i], alpha=0.5, label=None)  # No label for trials
    
    # Compute and plot mean line (white dotted)
    mean_trials = np.mean(total_trials, axis=0)
    plt.plot(years_range, mean_trials, color='white', linestyle='dotted', linewidth=2, label='Mean')
    
    # Compute and plot 90% CI lines (black dashed)
    ci_lower = np.percentile(total_trials, 5, axis=0)
    ci_upper = np.percentile(total_trials, 95, axis=0)
    plt.plot(years_range, ci_lower, color='black', linestyle='dashed', linewidth=1, label='90% CI Lower')
    plt.plot(years_range, ci_upper, color='black', linestyle='dashed', linewidth=1, label='90% CI Upper')
    
    
    plt.title(f"Total Cumulative Streams Over Time ({trials} Trials, Projected to {end_year})")
    plt.xlabel("Calendar Year")
    plt.ylabel("Total Cumulative Streams (millions)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cumulative_streams.png')
    plt.close()
    print("Plot saved as 'cumulative_streams.png'", flush=True)

def main():
    try:
        input_file = 'inputs.json'
        print("Reading input file", flush=True)
        iterations, end_year, mean_yoy_by_age, std_yoy_by_age, multiplier_by_year, song_age_groups = read_input_file(input_file)
        
        years = end_year - CURRENT_YEAR
        all_results = {}
        total_final_streams = []
        
        for song_age, initial_streams in song_age_groups.items():
            release_year = CURRENT_YEAR - int(song_age)
            print(f"\nSimulating for songs aged {song_age} in {CURRENT_YEAR} (Initial streams: {initial_streams} million, released {release_year})", flush=True)
            results = monte_carlo_simulation(
                initial_streams, release_year, mean_yoy_by_age, std_yoy_by_age, multiplier_by_year, years, trials=iterations
            )
            all_results[song_age] = results
            analyzed_results = analyze_results(results, song_age, years, release_year, end_year)
            total_final_streams.append(analyzed_results[:, -1])
        
        print("\nPlotting results", flush=True)
        plot_results(all_results, years, end_year, iterations)
        
        print("\nSaving Excel file", flush=True)
        # Compute mean cumulative streams per age group per year
        mean_results = {age: np.mean(results, axis=0) for age, results in all_results.items()}
        df_means = pd.DataFrame(mean_results, index=range(CURRENT_YEAR, end_year + 1))
        df_means.index.name = 'Year'
        
        # Compute total cumulative streams across all age groups per trial per year
        total_trials = np.zeros((iterations, years + 1))
        for song_age, trial_results in all_results.items():
            total_trials += trial_results
        
        # Compute confidence intervals for total streams per year
        ci_data = {}
        for ci in range(90, 0, -10):
            lower_percentile = (100 - ci) / 2
            upper_percentile = 100 - lower_percentile
            ci_lower = np.percentile(total_trials, lower_percentile, axis=0)
            ci_upper = np.percentile(total_trials, upper_percentile, axis=0)
            ci_data[f"{ci}% CI Lower"] = ci_lower
            ci_data[f"{ci}% CI Upper"] = ci_upper
        
        # Create DataFrame for CIs
        df_cis = pd.DataFrame(ci_data, index=range(CURRENT_YEAR, end_year + 1))
        
        # Combine means and CIs into a single DataFrame
        df = pd.concat([df_means, df_cis], axis=1)
        
        # Write to Excel with formatting
        output_file = 'song_streams_simulation.xlsx'
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Write DataFrame to Excel
            df.to_excel(writer, sheet_name='Simulation Results', float_format="%.2f")
            
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Simulation Results']
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            
            # Apply header formatting
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num + 1, value, header_format)  # +1 for index column
            
            # Adjust column widths based on content
            for column in df.columns:
                column_width = max(len(str(column)) + 2, 10)  # Minimum width of 10
                col_idx = df.columns.get_loc(column) + 1  # +1 for index column
                worksheet.set_column(col_idx, col_idx, column_width)
            
            # Adjust index column width
            worksheet.set_column(0, 0, 10)
        
        print(f"Results saved to '{output_file}'", flush=True)
        
        # Compute confidence intervals for total streams at the end year
        total_final_sum = np.sum(total_final_streams, axis=0)
        total_mean = np.mean(total_final_sum)
        total_median = np.median(total_final_sum)
        confidence_intervals = {}
        for ci in range(90, 0, -10):
            lower_percentile = (100 - ci) / 2
            upper_percentile = 100 - lower_percentile
            ci_lower = np.percentile(total_final_sum, lower_percentile)
            ci_upper = np.percentile(total_final_sum, upper_percentile)
            confidence_intervals[ci] = (ci_lower, ci_upper)
        
        print(f"\nTotal Cumulative Streams Across All Age Groups at {end_year}:", flush=True)
        print(f"Mean: {total_mean:.2f} million", flush=True)
        print(f"Median: {total_median:.2f} million", flush=True)
        for ci in sorted(confidence_intervals.keys(), reverse=True):
            ci_lower, ci_upper = confidence_intervals[ci]
            print(f"{ci}% Confidence Interval: {ci_lower:.2f} to {ci_upper:.2f} million", flush=True)
        
        print("\nSimulation completed successfully.", flush=True)
    except Exception as e:
        print(f"Error during simulation: {e}", flush=True)
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()

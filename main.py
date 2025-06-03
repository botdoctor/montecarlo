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
    """Calculate age-based growth multiplier."""
    return max(1.0 - 0.05 * age, 0.5)  # Decreases by 0.05 per year, min 0.5

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
    max_age = end_year - (CURRENT_YEAR - 10)  # Max age based on oldest song
    for age in range(max_age + 1):
        if age not in mean_yoy_by_age:
            mean_yoy_by_age[age] = 0.05
        if age not in std_yoy_by_age:
            std_yoy_by_age[age] = 0.02
        if not isinstance(mean_yoy_by_age[age], (int, float)):
            raise ValueError(f"Invalid mean_yoy for age {age}: must be numeric")
        if not isinstance(std_yoy_by_age[age], (int, float)) or std_yoy_by_age[age] < 0:
            raise ValueError(f"Invalid std_yoy for age {age}: must be numeric and non-negative")

    # Validate multiplier_by_year
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
    """Run Monte Carlo simulation with age-dependent YOY growth and multipliers."""
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
            print(f"Trial {i+1}, Year {sim_year}, Song age {song_age}", flush=True)
            mean_yoy = mean_yoy_by_age.get(song_age, 0.05)
            std_yoy = std_yoy_by_age.get(song_age, 0.02)
            growth = np.random.normal(mean_yoy, std_yoy)
            streams *= (1 + growth)
            streams *= get_age_multiplier(song_age)
            streams *= multiplier_by_year.get(sim_year, 1.0)
            cumulative_streams += streams
            yearly_cumulative[i, year_offset] = cumulative_streams
    print("Simulation complete", flush=True)
    return yearly_cumulative

def analyze_results(results, song_age, years, release_year, end_year):
    """Analyze and print simulation results for cumulative streams."""
    final_cumulative = results[:, -1]
    mean_streams = np.mean(final_cumulative)
    median_streams = np.median(final_cumulative)
    ci_lower = np.percentile(final_cumulative, 5)
    ci_upper = np.percentile(final_cumulative, 95)
    
    print(f"\nCumulative Results for songs aged {song_age} in {CURRENT_YEAR} (released {release_year}, projected to {end_year}):", flush=True)
    print(f"Mean cumulative streams: {mean_streams:.2f} million", flush=True)
    print(f"Median cumulative streams: {median_streams:.2f} million", flush=True)
    print(f"90% Confidence Interval: {ci_lower:.2f} to {ci_upper:.2f} million", flush=True)
    
    return results

def plot_results(all_results, years, end_year, trials):
    """Plot line graph of total cumulative streams per year for each trial."""
    print("Generating plot", flush=True)
    plt.figure(figsize=(12, 6))
    years_range = list(range(CURRENT_YEAR, end_year + 1))
    total_trials = np.zeros((trials, years + 1))
    for song_age, trial_results in all_results.items():
        total_trials += trial_results
    for i in range(trials):
        plt.plot(years_range, total_trials[i], alpha=0.3, color='blue', label='Trials' if i == 0 else None)
    plt.title(f"Total Cumulative Streams Over Time ({trials} Trials, Age- and Year-Based Multipliers, Projected to {end_year})")
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
        
        print("\nSaving CSV", flush=True)
        mean_results = {age: np.mean(results, axis=0) for age, results in all_results.items()}
        df = pd.DataFrame(mean_results, index=range(CURRENT_YEAR, end_year + 1))
        df.to_csv('song_streams_simulation.csv')
        print("Results saved to 'song_streams_simulation.csv'", flush=True)
        
        # Summarize total streams
        total_mean = np.mean(np.sum(total_final_streams, axis=0))
        total_median = np.median(np.sum(total_final_streams, axis=0))
        total_ci_lower = np.percentile(np.sum(total_final_streams, axis=0), 5)
        total_ci_upper = np.percentile(np.sum(total_final_streams, axis=0), 95)
        print(f"\nTotal Cumulative Streams Across All Age Groups at {end_year}:", flush=True)
        print(f"Mean: {total_mean:.2f} million", flush=True)
        print(f"Median: {total_median:.2f} million", flush=True)
        print(f"90% Confidence Interval: {total_ci_lower:.2f} to {total_ci_upper:.2f} million", flush=True)
        
        print("\nSimulation completed successfully.", flush=True)
    except Exception as e:
        print(f"Error during simulation: {e}", flush=True)
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
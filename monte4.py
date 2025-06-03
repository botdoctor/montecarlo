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
            mean_yoy_by_age[age] = 0.05  # Default
        if age not in std_yoy_by_age:
            std_yoy_by_age[age] = 0.02  # Default
        if not isinstance(mean_yoy_by_age[age], (int, float)):
            raise ValueError(f"Invalid mean_yoy for age {age}: must be numeric")
        if not isinstance(std_yoy_by_age[age], (int, float)) or std_yoy_by_age[age] < 0:
            raise ValueError(f"Invalid std_yoy for age {age}: must be numeric and non-negative")

    # Validate multiplier_by_year
    multiplier_by_year = data.get('multiplier_by_year', {})
    years = range(CURRENT_YEAR, end_year + 1)
    for year in years:
        if year not in multiplier_by_year:
            multiplier_by_year[year] = 1.0  # Default
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
    """
    Run Monte Carlo simulation with age-dependent YOY growth and multipliers.
    Returns cumulative streams per year for each trial.
    """
    yearly_cumulative = np.zeros((trials, years + 1))  # Include end year
    for i in range(trials):
        cumulative_streams = 0
        streams = initial_streams
        for year_offset in range(years + 1):
            sim_year = CURRENT_YEAR + year_offset
            song_age = sim_year - release_year
            if song_age < 0:
                yearly_cumulative[i, year_offset] = cumulative_streams
                continue  # Skip years before release
            # Sample YOY growth based on song age
            mean_yoy = mean_yoy_by_age.get(song_age, 0.05)
            std_yoy = std_yoy_by_age.get(song_age, 0.02)
            growth = np.random.normal(mean_yoy, std_yoy)
            streams *= (1 + growth)  # Apply YOY growth
            # Apply age-based multiplier
            streams *= get_age_multiplier(song_age)
            # Apply year-based multiplier
            streams *= multiplier_by_year.get(sim_year, 1.0)
            # Add to cumulative total
            cumulative_streams += streams
            yearly_cumulative[i, year_offset] = cumulative_streams
    return yearly_cumulative

def analyze_results(results, song_age, years, release_year, end_year):
    """Analyze and print simulation results for cumulative streams."""
    final_cumulative = results[:, -1]  # Cumulative streams at end year
    mean_streams = np.mean(final_cumulative)
    median_streams = np.median(final_cumulative)
    ci_lower = np.percentile(final_cumulative, 5)  # 90% CI
    ci_upper = np.percentile(final_cumulative, 95)
    
    print(f"\nCumulative Results for songs aged {song_age} in {CURRENT_YEAR} (released {release_year}, projected to {end_year}):")
    print(f"Mean cumulative streams: {mean_streams:.2f} million")
    print(f"Median cumulative streams: {median_streams:.2f} million")
    print(f"90% Confidence Interval: {ci_lower:.2f} to {ci_upper:.2f} million")
    
    return results  # Return full results for trial-by-trial plotting

def plot_results(all_results, years, end_year, trials):
    """Plot line graph of total cumulative streams per year for each trial."""
    plt.figure(figsize=(12, 6))
    years_range = list(range(CURRENT_YEAR, end_year + 1))
    # Sum cumulative streams across all age groups for each trial
    total_trials = np.zeros((trials, years + 1))
    for song_age, trial_results in all_results.items():
        total_trials += trial_results
    # Plot each trial
    for i in range(trials):
        plt.plot(years_range, total_trials[i], alpha=0.3, color='blue', label='Trials' if i == 0 else None)
    plt.title(f"Total Cumulative Streams Over Time ({trials} Trials, Age- and Year-Based Multipliers, Projected to {end_year})")
    plt.xlabel("Calendar Year")
    plt.ylabel("Total Cumulative Streams (millions)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cumulative_streams.png')
    plt.close()
    print("Plot saved as 'cumulative_streams.png'")

def main():
    # Read inputs from file
    input_file = 'inputs.json'
    try:
        iterations, end_year, mean_yoy_by_age, std_yoy_by_age, multiplier_by_year, song_age_groups = read_input_file(input_file)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    years = end_year - CURRENT_YEAR
    all_results = {}
    
    # Run simulation for each song age group
    for song_age, initial_streams in song_age_groups.items():
        release_year = CURRENT_YEAR - int(song_age)
        print(f"\nSimulating for songs aged {song_age} in {CURRENT_YEAR} (Initial streams: {initial_streams} million, released {release_year})")
        results = monte_carlo_simulation(
            initial_streams, release_year, mean_yoy_by_age, std_yoy_by_age, multiplier_by_year, years, trials=iterations
        )
        all_results[song_age] = results
    
    # Plot results
    plot_results(all_results, years, end_year, iterations)
    
    # Save results to CSV
    mean_results = {age: np.mean(results, axis=0) for age, results in all_results.items()}
    df = pd.DataFrame(mean_results, index=range(CURRENT_YEAR, end_year + 1))
    df.to_csv('song_streams_simulation.csv')
    print("\nResults saved to 'song_streams_simulation.csv'")
    print("Simulation completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
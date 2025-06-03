import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Get current year
CURRENT_YEAR = datetime.now().year

def get_user_inputs():
    """Prompt user for simulation inputs."""
    # Song release year
    release_year = int(input(f"Enter song release year (<= {CURRENT_YEAR}): "))
    while release_year > CURRENT_YEAR:
        print(f"Release year must be <= {CURRENT_YEAR}.")
        release_year = int(input("Enter song release year: "))
    
    # End year
    end_year = int(input(f"Enter end year for simulation (> {CURRENT_YEAR}): "))
    while end_year <= CURRENT_YEAR:
        print(f"End year must be after {CURRENT_YEAR}.")
        end_year = int(input("Enter end year: "))
    
    # Year-specific mean YOY and std
    years = range(CURRENT_YEAR, end_year + 1)
    mean_yoy_by_year = {}
    std_yoy_by_year = {}
    print(f"\nEnter mean YOY growth and std for each year ({CURRENT_YEAR} to {end_year}).")
    print("Example: '2025: 0.05, 0.02' (mean, std). Press Enter for default (0.05, 0.02).")
    for year in years:
        inp = input(f"YOY mean, std for {year} (or Enter for default): ").strip()
        if inp:
            try:
                mean, std = map(float, inp.split(','))
                mean_yoy_by_year[year] = mean
                std_yoy_by_year[year] = std
            except ValueError:
                print("Invalid format. Using default.")
                mean_yoy_by_year[year] = 0.05
                std_yoy_by_year[year] = 0.02
        else:
            mean_yoy_by_year[year] = 0.05
            std_yoy_by_year[year] = 0.02
    
    # Age- and year-specific growth multipliers
    max_age = end_year - release_year + 1  # Max age song can reach
    growth_multipliers = {}
    print(f"\nEnter growth multipliers for song age (0 to {max_age}) by year.")
    print("Example: '2025, 3: 1.1' (year, age: multiplier). Enter 'done' when finished.")
    print("Press Enter for default (1.0 for all).")
    while True:
        inp = input("Year, age: multiplier (or 'done'): ").strip()
        if inp.lower() == 'done':
            break
        if not inp:
            continue
        try:
            year_age, multiplier = inp.split(':')
            year, age = map(int, year_age.split(','))
            if year not in years or age > max_age:
                print(f"Invalid year ({year}) or age ({age}).")
                continue
            growth_multipliers[(year, age)] = float(multiplier)
        except ValueError:
            print("Invalid format. Try again.")
    
    # Default multipliers to 1.0 if not specified
    for year in years:
        for age in range(max_age + 1):
            if (year, age) not in growth_multipliers:
                growth_multipliers[(year, age)] = 1.0
    
    # Streams by age group
    age_groups = {}
    print("\nEnter initial streams (in millions) for each age group.")
    print("Example: 'teens: 10.5, adults: 15.2'. Enter 'done' when finished.")
    while True:
        inp = input("Age group and streams (e.g., 'teens: 10.5') or 'done': ")
        if inp.lower() == 'done':
            if not age_groups:
                print("At least one age group required.")
                continue
            break
        try:
            group, streams = inp.split(':')
            age_groups[group.strip()] = float(streams.strip())
        except ValueError:
            print("Invalid format. Use 'group: streams'.")
    
    return release_year, end_year, mean_yoy_by_year, std_yoy_by_year, growth_multipliers, age_groups

def monte_carlo_simulation(initial_streams, release_year, mean_yoy_by_year, std_yoy_by_year, growth_multipliers, years, trials=10000):
    """
    Run Monte Carlo simulation with year- and age-dependent growth.
    Returns array of final streams for each trial.
    """
    results = np.zeros(trials)
    for i in range(trials):
        streams = initial_streams
        for year_offset in range(years):
            sim_year = CURRENT_YEAR + year_offset
            song_age = sim_year - release_year
            if song_age < 0:
                continue  # Skip years before release
            # Sample YOY growth for the year
            mean_yoy = mean_yoy_by_year[sim_year]
            std_yoy = std_yoy_by_year[sim_year]
            growth = np.random.normal(mean_yoy, std_yoy)
            streams *= (1 + growth)  # Apply YOY growth
            # Apply age- and year-specific multiplier
            multiplier = growth_multipliers.get((sim_year, song_age), 1.0)
            streams *= multiplier
        results[i] = streams
    return results

def analyze_results(results, age_group, years, release_year, end_year):
    """Analyze and print simulation results."""
    mean_streams = np.mean(results)
    median_streams = np.median(results)
    ci_lower = np.percentile(results, 5)  # 90% CI
    ci_upper = np.percentile(results, 95)
    
    print(f"\nResults for {age_group} after {years} years (song released {release_year}, projected to {end_year}):")
    print(f"Mean streams: {mean_streams:.2f} million")
    print(f"Median streams: {median_streams:.2f} million")
    print(f"90% Confidence Interval: {ci_lower:.2f} to {ci_upper:.2f} million")
    
    return mean_streams, ci_lower, ci_upper

def plot_results(all_results, years, release_year, end_year):
    """Plot histogram of results for each age group."""
    plt.figure(figsize=(10, 6))
    for age_group, results in all_results.items():
        plt.hist(results, bins=50, alpha=0.5, label=age_group, density=True)
    plt.title(f"Distribution of Streams After {years} Years (Released {release_year}, Projected to {end_year})")
    plt.xlabel("Streams (millions)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # Get inputs
    release_year, end_year, mean_yoy_by_year, std_yoy_by_year, growth_multipliers, age_groups = get_user_inputs()
    years = end_year - CURRENT_YEAR
    
    all_results = {}
    
    # Run simulation for each age group
    for age_group, initial_streams in age_groups.items():
        print(f"\nSimulating for {age_group} (Initial streams: {initial_streams} million)")
        results = monte_carlo_simulation(
            initial_streams, release_year, mean_yoy_by_year, std_yoy_by_year, growth_multipliers, years
        )
        all_results[age_group] = results
        analyze_results(results, age_group, years, release_year, end_year)
    
    # Plot results
    plot_results(all_results, years, release_year, end_year)
    
    # Save results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv('song_streams_simulation.csv', index=False)
    print("\nResults saved to 'song_streams_simulation.csv'")

if __name__ == "__main__":
    main()
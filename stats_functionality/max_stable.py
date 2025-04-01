import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Function to fit GEV distribution to block maxima at each spatial location
def fit_gev(data, block_size=24):
    """
    Fits a GEV distribution to temporal block maxima at each spatial point.
    
    Parameters:
    -----------
    data : xarray.DataArray
        The variable of interest (e.g., temperature, precipitation)
    block_size : int
        Size of temporal blocks for calculating maxima (e.g., 24 for daily)
    
    Returns:
    --------
    xarray.Dataset
        Dataset with GEV parameters (shape, location, scale) for each spatial point
    """
    # Reshape time into blocks
    n_blocks = data.time.size // block_size
    data_blocked = data.isel(time=slice(0, n_blocks * block_size))
    
    # Calculate block maxima
    data_reshaped = data_blocked.values.reshape(n_blocks, block_size, data.latitude.size, data.longitude.size)
    block_maxima = np.max(data_reshaped, axis=1)
    
    # Create new time coordinate for blocks
    block_times = data.time[::block_size].values[:n_blocks]
    
    # Create DataArray for block maxima
    maxima_da = xr.DataArray(
        block_maxima,
        dims=['time', 'latitude', 'longitude'],
        coords={
            'time': block_times,
            'latitude': data.latitude.values,
            'longitude': data.longitude.values
        }
    )
    
    # Initialize arrays to store GEV parameters
    shape = np.zeros((data.latitude.size, data.longitude.size))
    loc = np.zeros((data.latitude.size, data.longitude.size))
    scale = np.zeros((data.latitude.size, data.longitude.size))
    
    # Fit GEV at each location
    for i in range(data.latitude.size):
        for j in range(data.longitude.size):
            max_values = maxima_da.isel(latitude=i, longitude=j).values
            # Skip locations with NaN or infinite values
            if not np.all(np.isfinite(max_values)):
                shape[i, j] = np.nan
                loc[i, j] = np.nan
                scale[i, j] = np.nan
                continue
            
            try:
                # Fit GEV distribution
                shape_param, loc_param, scale_param = stats.genextreme.fit(max_values)
                shape[i, j] = -shape_param  # scipy uses negative shape parameter
                loc[i, j] = loc_param
                scale[i, j] = scale_param
            except:
                shape[i, j] = np.nan
                loc[i, j] = np.nan
                scale[i, j] = np.nan
    
    # Create and return Dataset with GEV parameters
    gev_params = xr.Dataset(
        data_vars={
            'shape': (('latitude', 'longitude'), shape),
            'location': (('latitude', 'longitude'), loc),
            'scale': (('latitude', 'longitude'), scale)
        },
        coords={
            'latitude': data.latitude.values,
            'longitude': data.longitude.values
        }
    )
    
    return gev_params, maxima_da

# Function to estimate spatial dependence using extremal coefficient
def estimate_extremal_coefficients(maxima_da, threshold_quantile=0.95):
    """
    Estimates pairwise extremal coefficients between locations.
    
    Parameters:
    -----------
    maxima_da : xarray.DataArray
        DataArray of block maxima
    threshold_quantile : float
        Quantile threshold for defining extremes
    
    Returns:
    --------
    numpy.ndarray, numpy.ndarray
        Distance matrix and corresponding extremal coefficient matrix
    """
    # Sample points for analysis (too many points would be computationally intensive)
    lat_indices = np.linspace(0, maxima_da.latitude.size-1, 50, dtype=int)
    lon_indices = np.linspace(0, maxima_da.longitude.size-1, 50, dtype=int)
    
    # Extract coordinates
    lats = maxima_da.latitude.values[lat_indices]
    lons = maxima_da.longitude.values[lon_indices]
    
    # Create a grid of points
    points = []
    for lat in lats:
        for lon in lons:
            points.append((lat, lon))
    
    # Compute distance matrix
    distance_matrix = euclidean_distances(points)
    
    # Get maxima at sampled points
    maxima_values = []
    for lat_idx in lat_indices:
        for lon_idx in lon_indices:
            values = maxima_da.isel(latitude=lat_idx, longitude=lon_idx).values
            if np.all(np.isfinite(values)):
                maxima_values.append(values)
            else:
                maxima_values.append(np.zeros_like(values))
    
    maxima_values = np.array(maxima_values)
    
    # Calculate marginal transformations to unit Fréchet
    n_points = len(points)
    transformed_maxima = np.zeros_like(maxima_values)
    
    for i in range(n_points):
        if np.all(maxima_values[i] == 0):
            transformed_maxima[i] = np.zeros_like(maxima_values[i])
            continue
            
        ranks = stats.rankdata(maxima_values[i])
        empirical_cdf = ranks / (len(maxima_values[i]) + 1)
        transformed_maxima[i] = -1.0 / np.log(empirical_cdf)
    
    # Compute pairwise extremal coefficients
    extremal_coefs = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            if np.all(transformed_maxima[i] == 0) or np.all(transformed_maxima[j] == 0):
                extremal_coefs[i, j] = extremal_coefs[j, i] = np.nan
                continue
                
            # Compute extremal coefficient
            maxima_ratio = np.minimum(transformed_maxima[i], transformed_maxima[j]) / np.maximum(transformed_maxima[i], transformed_maxima[j])
            extremal_coefs[i, j] = extremal_coefs[j, i] = 2 * np.mean(maxima_ratio)
    
    return distance_matrix, extremal_coefs

# Function to fit a max-stable process model
def fit_smith_model(distance_matrix, extremal_coefs):
    """
    Fits a Smith model (Gaussian max-stable process) to the extremal coefficients.
    
    Parameters:
    -----------
    distance_matrix : numpy.ndarray
        Matrix of distances between locations
    extremal_coefs : numpy.ndarray
        Matrix of extremal coefficients
    
    Returns:
    --------
    float
        Fitted range parameter
    """
    # Flatten the matrices to get pairs of distances and coefficients
    distances = distance_matrix.flatten()
    coefficients = extremal_coefs.flatten()
    
    # Remove NaN values
    valid_indices = ~np.isnan(coefficients)
    valid_distances = distances[valid_indices]
    valid_coefficients = coefficients[valid_indices]
    
    # Theoretical model for Smith process: θ(h) = 2Φ(h/(2σ))
    # We'll estimate σ by minimizing squared error
    
    def smith_model(h, sigma):
        return 2 * stats.norm.cdf(h / (2 * sigma))
    
    def objective(sigma):
        predicted = smith_model(valid_distances, sigma)
        return np.sum((predicted - valid_coefficients) ** 2)
    
    # Optimization to find best sigma

    result = minimize_scalar(objective, bounds=(0.1, 100), method='bounded')
    
    return result.x

# Function to estimate return levels
def calculate_return_levels(gev_params, return_period=100):
    """
    Calculates return levels for a given return period at each location.
    
    Parameters:
    -----------
    gev_params : xarray.Dataset
        Dataset with GEV parameters
    return_period : int
        Return period in years
    
    Returns:
    --------
    xarray.DataArray
        Return levels at each location
    """
    # Extract GEV parameters
    shape = gev_params.shape.values
    loc = gev_params.location.values
    scale = gev_params.scale.values
    
    # Calculate return levels
    p = 1 - 1 / return_period
    return_levels = np.zeros_like(shape)
    
    # For shape ≈ 0 (Gumbel)
    gumbel_mask = np.abs(shape) < 1e-6
    return_levels[gumbel_mask] = loc[gumbel_mask] - scale[gumbel_mask] * np.log(-np.log(p))
    
    # For shape ≠ 0 (Fréchet or Weibull)
    nongumbel_mask = ~gumbel_mask
    return_levels[nongumbel_mask] = loc[nongumbel_mask] + scale[nongumbel_mask] * ((-np.log(p)) ** (-shape[nongumbel_mask]) - 1) / shape[nongumbel_mask]
    
    # Create DataArray
    return_level_da = xr.DataArray(
        return_levels,
        dims=['latitude', 'longitude'],
        coords={
            'latitude': gev_params.latitude.values,
            'longitude': gev_params.longitude.values
        },
        attrs={
            'return_period': f"{return_period} years",
            'description': 'Return levels based on GEV distribution'
        }
    )
    
    return return_level_da

# Main function to run the max-stable process analysis
def analyze_max_stable_process(data, variable_name='tp', block_size=24, return_periods=[10, 50, 100]):
    """
    Main function to perform max-stable process analysis on climate data.
    
    Parameters:
    -----------
    data : xarray.Dataset
        The dataset containing climate variables
    variable_name : str
        Name of the variable to analyze
    block_size : int
        Size of temporal blocks for maxima calculation
    return_periods : list
        List of return periods (in years) to calculate
        
    Returns:
    --------
    dict
        Dictionary containing results of the analysis
    """
    print(f"Analyzing {variable_name} using max-stable processes...")
    
    # Extract the variable data
    variable_data = data[variable_name]
    
    # Step 1: Fit GEV distribution to block maxima
    print("Fitting GEV distribution to block maxima...")
    gev_params, block_maxima = fit_gev(variable_data, block_size)
    
    # Step 2: Estimate spatial dependence
    print("Estimating spatial dependence with extremal coefficients...")
    distance_matrix, extremal_coefs = estimate_extremal_coefficients(block_maxima)
    
    # Step 3: Fit Smith model
    print("Fitting Smith max-stable process model...")
    sigma = fit_smith_model(distance_matrix, extremal_coefs)
    print(f"Estimated range parameter (sigma): {sigma:.2f}")
    
    # Step 4: Calculate return levels
    return_level_dict = {}
    for period in return_periods:
        print(f"Calculating {period}-year return levels...")
        return_level = calculate_return_levels(gev_params, period)
        return_level_dict[period] = return_level
    
    # Compile results
    results = {
        'gev_parameters': gev_params,
        'block_maxima': block_maxima,
        'smith_sigma': sigma,
        'return_levels': return_level_dict
    }
    
    print("Max-stable process analysis complete.")
    return results

# Function to plot the results
def plot_max_stable_results(results, variable_name='tp'):
    """
    Creates visualization of max-stable process analysis results.
    
    Parameters:
    -----------
    results : dict
        Dictionary of results from analyze_max_stable_process
    variable_name : str
        Name of the variable analyzed
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot GEV parameters
    gev_params = results['gev_parameters']
    
    # Plot shape parameter
    ax1 = fig.add_subplot(321)
    gev_params.shape.plot(ax=ax1, cmap='RdBu_r')
    ax1.set_title('GEV Shape Parameter')
    
    # Plot location parameter
    ax2 = fig.add_subplot(322)
    gev_params.location.plot(ax=ax2, cmap='viridis')
    ax2.set_title('GEV Location Parameter')
    
    # Plot scale parameter
    ax3 = fig.add_subplot(323)
    gev_params.scale.plot(ax=ax3, cmap='plasma')
    ax3.set_title('GEV Scale Parameter')
    
    # Plot sample of block maxima
    ax4 = fig.add_subplot(324)
    results['block_maxima'].isel(time=0).plot(ax=ax4, cmap='YlOrRd')
    ax4.set_title('Sample of Block Maxima')
    
    # Plot return levels
    return_levels = results['return_levels']
    periods = list(return_levels.keys())
    
    ax5 = fig.add_subplot(325)
    return_levels[periods[0]].plot(ax=ax5, cmap='hot_r')
    ax5.set_title(f'{periods[0]}-year Return Level')
    
    ax6 = fig.add_subplot(326)
    if len(periods) > 1:
        return_levels[periods[-1]].plot(ax=ax6, cmap='hot_r')
        ax6.set_title(f'{periods[-1]}-year Return Level')
    
    plt.tight_layout()
    plt.suptitle(f'Max-Stable Process Analysis for {variable_name}', fontsize=16, y=1.05)
    
    return fig

# Example usage
if __name__ == "__main__":
    # Assuming 'data' is your xarray.Dataset loaded from the NetCDF file
    # data = xr.open_dataset('your_file.nc')
    
    # Example of how to use these functions
    # results = analyze_max_stable_process(data, variable_name='tp', block_size=24)
    # fig = plot_max_stable_results(results, variable_name='tp')
    # plt.show()
    
    print("To use this script with your data, call analyze_max_stable_process() with your xarray Dataset")
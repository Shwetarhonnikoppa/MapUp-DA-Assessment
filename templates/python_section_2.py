import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
     # Fill diagonal with 0 as the distance from a point to itself is 0
    np.fill_diagonal(df.values, 0)
    
    # Make the matrix symmetric (i.e., distance A to B equals B to A)
    df = df.where(df.notna(), df.T)

    # Apply cumulative distance between toll locations
    for k in df.columns:
        for i in df.columns:
            for j in df.columns:
                # Update the distance from i to j if passing through k is shorter
                if df.loc[i, j] > df.loc[i, k] + df.loc[k, j]:
                    df.loc[i, j] = df.loc[i, k] + df.loc[k, j]

    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
     unrolled_data = []
    # Iterate through the matrix
    for id_start in df.index:
        for id_end in df.columns:
            distance = df.loc[id_start, id_end]
            unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    
    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)
    
    return unrolled_df
    


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
     # Calculate the average distance for the reference ID
    reference_distances = df[df['id_start'] == reference_id]['distance']
    avg_distance = reference_distances.mean()

    # Define the 10% threshold (upper and lower bounds)
    lower_bound = avg_distance * 0.9
    upper_bound = avg_distance * 1.1

    # Find all other IDs whose average distance is within the 10% range of the reference ID's average distance
    id_grouped = df.groupby('id_start')['distance'].mean()
    ids_within_threshold = id_grouped[
        (id_grouped >= lower_bound) & (id_grouped <= upper_bound)
    ].index.tolist()

    # Return the sorted list of IDs
    return sorted(ids_within_threshold)                                                                        


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
     # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates by multiplying the distance with each rate coefficient
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    def apply_discount(row):
        start_day = row['start_day']
        start_time = datetime.datetime.strptime(row['start_time'], '%H:%M:%S').time()

        # Weekdays Monday to Friday
        if start_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            if datetime.time(0, 0, 0) <= start_time <= datetime.time(10, 0, 0):
                discount_factor = 0.8
            elif datetime.time(10, 0, 0) < start_time <= datetime.time(18, 0, 0):
                discount_factor = 1.2
            else:
                discount_factor = 0.8
        # Weekends Saturday, Sunday
        else:
            discount_factor = 0.7

        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            row[vehicle] = row[vehicle] * discount_factor
    return df

from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    for i in range(0, len(lst), n):
        end = min(i + n - 1, len(lst) - 1)
        left, right = i, end
        while left < right:
            # Swap elements at 'left' and 'right'
            lst[left], lst[right] = lst[right], lst[left]
            left += 1
            right -= 1
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}
    for string in lst:
        # Finding the length of the string
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        # Append the string to the list to its length
        length_dict[length].append(string)
    # Return the dictionary    
    return dict(sorted(length_dict.items()))
    

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def flatten(current_dict: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        # Initialize an empty dictionary to hold the flattened items
        items = {}
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.update(flatten(value, new_key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        items.update(flatten(item, f"{new_key}[{index}]"))
                    else:
                        items[f"{new_key}[{index}]"] = item
            else:
               # add them directly to the items dictionary
                items[new_key] = value
        # Return the flattened items
        return items
    return flatten(nested_dict)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start: int):
        # If the current permutation is complete, add it to the result
        if start == len(nums):
            result.append(nums[:])
            return
        # Use a set to track which numbers have been used at this position
        seen = set()
        for i in range(start, len(nums)):
            # Skip duplicates
            if nums[i] in seen:
                continue
            # Mark this number as seen
            seen.add(nums[i])
            # Swap the current number with the start index
            nums[start], nums[i] = nums[i], nums[start]
            # Recursively call backtrack for the next position
            backtrack(start + 1)
            # Backtrack to restore the original list
            nums[start], nums[i] = nums[i], nums[start]
    result = []  # List to hold the unique permutations
    nums.sort()  # Sort the list to ensure duplicates are adjacent
    backtrack(0)  # Start the backtracking process
    return result


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Regular expressions for the date formats
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]

    # Compile all patterns into a single regex pattern
    combined_pattern = re.compile('|'.join(date_patterns))

    # Find all matching dates in the text
    valid_dates = combined_pattern.findall(text)
    return valid_dates


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string to get a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate the distance using the Haversine formula
    distances = [0]  # Start with 0 for the first point
    for i in range(1, len(df)):
        dist = haversine(df.latitude[i-1], df.longitude[i-1], df.latitude[i], df.longitude[i])
        distances.append(dist)
    
    # Add the distances as a new column to the DataFrame
    df['distance'] = distances
    
    return df  # Return the DataFrame


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)  # Get the size of the matrix
    # Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Create a new matrix to hold the final values
    final_matrix = [[0] * n for _ in range(n)]
    
    # Calculate the sum of rows and columns
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  # Sum of the current row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the current column
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude the element itself
    
    return final_matrix  # Return the transformed matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    # Combine the date and time columns into a single datetime column
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])
    
    # Initialize a boolean series for results
    results = pd.Series(index=grouped.groups.keys(), dtype=bool)

    # Check each group for completeness
    for (id_val, id_2_val), group in grouped:
        # Check if we have all 7 days
        days_covered = group['start_datetime'].dt.dayofweek.unique()
        has_full_weeks = len(days_covered) == 7
        
        # Check if the timestamps cover a full 24-hour period
        time_span = group['end_datetime'].max() - group['start_datetime'].min()
        covers_full_day = time_span >= pd.Timedelta(days=1)
        
        # Mark the result
        results[(id_val, id_2_val)] = not (has_full_weeks and covers_full_day)
    
    return results  # Return the boolean series
 
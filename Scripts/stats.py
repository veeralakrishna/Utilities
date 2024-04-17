import numpy as np
import pandas as pd


def stats(data_frame, custom_aggregations=None):
    """
    Collect stats about all columns of a dataframe, their types,
    and descriptive statistics, return them in a Pandas DataFrame.

    :param data_frame: the Pandas DataFrame to show statistics for.
    :param custom_aggregations: a dictionary of custom aggregation functions
                                 for numeric columns {column_name: aggregation_function}.
    :return: a new Pandas DataFrame with the statistics data for the given DataFrame.
    """
    if custom_aggregations is None:
        custom_aggregations = {}
    
    stats_column_names = ('column', 'dtype', 'nan_cts', 'val_cts',
                          'min', 'max', 'mean', 'stdev', 'skew', 'kurtosis',
                          'mode', 'mode_freq', 'lower_bound', 'upper_bound')
    
    stats_array = []
    for column_name, col in data_frame.items():
        if pd.api.types.is_numeric_dtype(col):
            custom_aggregation = custom_aggregations.get(column_name, None)
            if custom_aggregation:
                stats_array.append([
                    column_name, col.dtype, col.isna().sum(), len(col.dropna().unique()),
                    col.min(), col.max(), custom_aggregation(col), col.std(), col.skew(),
                    col.kurtosis(), None, None,
                    col.quantile(0.25) - 1.5 * (col.quantile(0.75) - col.quantile(0.25)),
                    col.quantile(0.75) + 1.5 * (col.quantile(0.75) - col.quantile(0.25))
                ])
            else:
                stats_array.append([
                    column_name, col.dtype, col.isna().sum(), len(col.dropna().unique()),
                    col.min(), col.max(), col.mean(), col.std(), col.skew(),
                    col.kurtosis(), None, None,
                    col.quantile(0.25) - 1.5 * (col.quantile(0.75) - col.quantile(0.25)),
                    col.quantile(0.75) + 1.5 * (col.quantile(0.75) - col.quantile(0.25))
                ])
        elif pd.api.types.is_datetime64_any_dtype(col):
            stats_array.append([
                column_name, col.dtype, col.isna().sum(), len(col.dropna().unique()),
                None, None, None, None, None, None,
                col.mode().iloc[0], col.value_counts().iloc[0],
                None, None
            ])
        else:
            mode_value_counts = col.value_counts()
            stats_array.append([
                column_name, col.dtype, col.isna().sum(), len(col.dropna().unique()),
                None, None, None, None, None, None,
                mode_value_counts.index[0] if not mode_value_counts.empty else None,
                mode_value_counts.iloc[0] if not mode_value_counts.empty else None,
                None, None
            ])
    
    stats_df = pd.DataFrame(data=stats_array, columns=stats_column_names)
    return stats_df


def visualize_distribution(data_frame, column_name, bins=20):
    """
    Visualize the distribution of a numeric column using a histogram.

    :param data_frame: the Pandas DataFrame containing the column.
    :param column_name: the name of the column to visualize.
    :param bins: the number of bins for the histogram (default: 20).
    """
    if column_name not in data_frame.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data_frame[column_name], bins=bins, kde=True)
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()


# Unit Tests
# import unittest

# class TestStatsFunctions(unittest.TestCase):
#     def setUp(self):
#         self.df = pd.DataFrame({
#             'numeric_col': [1, 2, 3, 4, 5],
#             'string_col': ['a', 'b', 'c', 'd', 'e'],
#             'date_col': pd.date_range(start='2022-01-01', periods=5),
#             'nan_col': [1, np.nan, 3, np.nan, 5]
#         })

#     def test_stats(self):
#         stats_df = stats(self.df)
#         self.assertEqual(stats_df.shape[0], 4)
#         self.assertTrue('numeric_col' in stats_df['column'].values)
#         self.assertTrue('string_col' in stats_df['column'].values)
#         self.assertTrue('date_col' in stats_df['column'].values)
#         self.assertTrue('nan_col' in stats_df['column'].values)
    

# if __name__ == '__main__':
#     unittest.main()




# Example usage
data = pd.DataFrame({
    'text': ['example text', 'another example', 'yet another example'],
    'numeric1': [1, 2, 3],
    'numeric2': [4.5, 6.7, 8.9],
    'category': ['A', 'B', 'A']
})

# Collect statistics
stats_df = stats(data)
display(stats_df)


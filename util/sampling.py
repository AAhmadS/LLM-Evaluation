import pandas as pd
import random

def get_distinct_random_values(df, column_name, num_values=5):
  """
  Returns a list of distinct random values from a specified column in a DataFrame.

  Args:
    df: The input DataFrame.
    column_name: The name of the column to extract values from.
    num_values: The number of distinct random values to return.

  Returns:
    A list of distinct random values, or an empty list if the column doesn't exist
    or has fewer than num_values unique elements.
  """
  if column_name not in df.columns:
    raise ValueError("Specified column not found in the DataFrame.")

  unique_values = df[column_name].unique()
  if len(unique_values) < num_values:
    raise ValueError("Column has fewer unique values than num_values. The dataset as been possibly corrupted.")

  return random.sample(list(unique_values), num_values)


def sample_dataframe(df, context_column, values, quality_column):
    """
    Returns a sample of the DataFrame with twice the size of the values list,
    containing two pairs (highest and lowest from quality_column) for each value in context_column.
    """

    # Check if input columns exist in the DataFrame
    if context_column not in df.columns or quality_column not in df.columns:
        raise ValueError("Specified columns not found in the DataFrame.")

    sampled_df = pd.DataFrame()
    for val in values:
        # Filter the DataFrame for the current value
        subset = df[df[context_column] == val]

        if subset.empty:
          raise ValueError("Specified value not found in the DataFrame.")
        # Sort the subset by quality_column
        subset_sorted = subset.sort_values(by=quality_column)

        # Get the lowest and highest values from the subset
        lowest = subset_sorted.iloc[0]
        highest = subset_sorted.iloc[-1]

        # Append them to sampled_df
        sampled_df = pd.concat([sampled_df, pd.DataFrame([lowest, highest])], ignore_index=True)

    return sampled_df


def process_dataframe(df, context_function, column, use_response = True):
  """
  Processes a DataFrame, creates a list of dictionaries, and returns a string.

  Args:
    df: The input DataFrame.
    column: The name of the column to process.

  Returns:
    A string containing the concatenated outputs of key_stringer function.
  """

  def key_stringer(index, context, good, bad):
    # Replace this with your actual key_stringer function implementation
    if good:
      return f"[Sample_{index}]: "+ context+ "\n[Good_Response]"+good+"[/Good_Response]\n[Bad_Response]"+bad+"[/Bad_Response]\n"
    else:
      return f"[Sample_{index}]: "+ context+ "\n"

  output_string = "\nHere are some samples from the dataset:\n"
  values = df[column].unique()

  for index,value in enumerate(values):
      subset = df[df[column] == value]
      context = context_function(df.iloc[2*index])
      good = subset.iloc[1]["response"]
      bad = subset.iloc[0]["response"]
      index = subset.index[0]

      output_string += key_stringer(index,context, good, bad) if use_response else key_stringer(index,context, None, None)

  return output_string

def generate_samples_prompt(data, use_response, number_of_shots):
  distinct_values = get_distinct_random_values(data, "context", num_values = number_of_shots)
  sampled_dataframe = sample_dataframe(data, "context", distinct_values, "Overall")
  samples_prompt = process_dataframe(sampled_dataframe, lambda x: "[Context] "+ x["context"] + " [Context]\n[Facts] "+ x["fact"]+ " [Facts]", "context", use_response)
  return samples_prompt
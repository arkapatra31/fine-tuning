from SLM.fine_tuning_data.dataExtractorAndProcessor.load_dataset import df

"""
Ensure all the columns are in proper order and the data type is consistent across the records
There are although few ways to sanitize like - Drop the rows with invalid datatype or convert the datatype to string.
In case of NaN either drop them or replace with a fixed placeholder string
"""

# Ensure all review texts are strings
#df['review'] = df['review'].astype(str)

# Drop rows with NaN values in review_text
df.dropna(subset=['review'], inplace=True)
df.dropna(subset=['rating'], inplace=True)

# Or replace NaN values with a placeholder text
#df['review'].fillna('No review text provided', inplace=True)

# Also adjust by decrementing by 1 in order to start the labels from 0
df['rating'] = df['rating'] - 1

__all__ = [
    df
]
"""
FILTER
"""
def select_stations(df_column):
    pass


"""
REMOVE
"""
def replace_values(df, column_name, value_to_replace, replacement_value):
    """
    Remove values
    
    Parameters:
    @df
    @column_name: column where to delete a specified value
    @value_to_replace
    @replacement_value
    """
    df[column_name].replace({value_to_replace:replacement_value}, inplace=True)
    return df

def delete_bad_measures(df_column, operator, value):
    """
    Remove bad measures from a column
    
    Parameters : 
    @df-column
    @operator: <, <=, >=, >, ==
    @value (int)
    """
    pass

def get_filled_columns(row_series, columns):
    filled_columns = []
    for column in columns:
        if row_series[column] != []:
            if row_series[column][0] != 'none':
                filled_columns.append(column)
    return filled_columns
import pandas as pd
from sklearn.preprocessing import StandardScaler


def one_hot_dummies(df: pd.DataFrame, names: list, drop: bool = True) -> pd.DataFrame:
    """Function that creates dummies from a given list of keys
    and replace them into the position of key. 

    Args:
        df (pd.DataFrame): [frame with the data]
        names (list): [list of keys to create one-hot dummies]
        drop (bool): [True if you want to drop first dummy]
    Returns:
        [pd.DataFrame]: [return a modifed dataframe]
    """
    dfCopy = df.copy()
    for name in names:
        oneHotDummies = pd.get_dummies(dfCopy[name], drop_first=drop)
        # putting newly created dummies in place of name column
        indexOneHotPaste = dfCopy.columns.get_loc(name)
        for columns in oneHotDummies:
            dfCopy[columns] = oneHotDummies[columns]
        dfCopy = dfCopy.drop([name], axis=1)
        cols = (dfCopy.columns).tolist()
        cols = cols[0:indexOneHotPaste] + cols[-oneHotDummies.shape[1]:] + \
            cols[indexOneHotPaste:-oneHotDummies.shape[1]]
        dfCopy = dfCopy[cols]
    return dfCopy


def standardize(df: pd.DataFrame, colsToStandardize: list) -> pd.DataFrame:
    """Standardize columns with given names

    Args:
        df (pd.DataFrame): [all data ]
        colsToStandardize (list): [columns' names you want to standardize]

    Returns:
        pd.DataFrame: [return the same structure as df but with standardize data]
    """
    dfCopy = df.copy()
    dataToStandardize = df[colsToStandardize].copy()
    scaler = StandardScaler()
    scaler.fit(dataToStandardize)
    dataStandardized = scaler.transform(dataToStandardize)
    dataStandardized = pd.DataFrame(
        data=dataStandardized, columns=dataToStandardize.columns)
    # filling data with standardized values
    for column in dataStandardized.columns:
        dfCopy[column] = dataStandardized[column]
    return dfCopy

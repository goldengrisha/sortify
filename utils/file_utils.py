import pandas as pd


def get_tickets_as_df(path: str = "./data/tickets.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def get_cleaned_tickets_as_df(
    path: str = "./data/cleaned_tickets_v1.csv",
) -> pd.DataFrame:
    return pd.read_csv(path)

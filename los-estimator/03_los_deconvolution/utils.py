import pandas as pd

def date_to_day(date, start_day):\
    return (date - pd.Timestamp(start_day)).days
def day_to_date(day, start_day):
    return pd.Timestamp(start_day) + pd.Timedelta(days=day)

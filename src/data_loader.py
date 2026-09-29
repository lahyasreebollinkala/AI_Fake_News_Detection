import pandas as pd

from config import RAW_DATA


class DataLoader:

    @staticmethod
    def load():

        df = pd.read_csv(RAW_DATA)

        return df
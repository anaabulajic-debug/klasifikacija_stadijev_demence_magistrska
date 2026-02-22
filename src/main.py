from data_loading import load_dataset
import pandas as pd
from recoding import unify_dataset
from grafi import target_class_balance

def main():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    lethe, nacc = load_dataset()
    print(repr(lethe.columns.tolist()))
    print(repr(nacc.columns.tolist()))
    unify_dataset(lethe, nacc)
    print(repr(lethe.columns.tolist()))
    print(repr(nacc.columns.tolist()))
    target_class_balance(lethe, nacc)

if __name__ == '__main__':
    main()
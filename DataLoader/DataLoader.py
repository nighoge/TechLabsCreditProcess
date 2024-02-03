from fastai.tabular.all import *
import pandas as pd

def createTabularPandas():
    df = pd.read_csv('../credit_risk_dataset.csv')
    categorical_cols = ['person_home_ownership', 'loan_intent']
    continuous_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    binary_col  = 'cb_person_default_on_file'

    ordinal_col = 'loan_grade'
    custom_order = ['A', 'B', 'C', 'D', 'E']

    target = 'loan_status'

    class EncodeBinary(Transform):
        def encodes(self, df: pd.DataFrame):
            df[binary_col] = df[binary_col].map({'Y': 1, 'N': 0})

    class EncodeOrdinal(Transform):
        def encodes(self, df: pd.DataFrame):
            df[ordinal_col] = df[ordinal_col].map({val: i for i, val in enumerate(custom_order)})

    procs = [Categorify, FillMissing, EncodeBinary(), EncodeOrdinal()] # Normalize


    splits = RandomSplitter(valid_pct=0.2)(range_of(df))

    return TabularPandas(df, procs=procs, cat_names=categorical_cols, cont_names=continuous_cols, y_names=target, splits=splits)

def createDataLoader(bs=64):
    return createTabularPandas().dataloaders(bs=bs)



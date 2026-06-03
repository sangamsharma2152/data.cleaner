import numpy as np
import pandas as pd


def _train_test_split(df, test_size, random_state):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    shuffled = df.sample(frac=1, random_state=random_state)
    test_count = max(1, int(round(len(shuffled) * test_size)))
    test = shuffled.iloc[:test_count]
    train = shuffled.iloc[test_count:]

    if train.empty:
        raise ValueError("The train split is empty. Reduce test_size or add more rows.")

    return train, test


def _encode_train_test(train, test):
    train = train.copy()
    test = test.copy()
    encoded_columns = []

    categorical_columns = [
        col
        for col in train.columns
        if (
            pd.api.types.is_object_dtype(train[col])
            or pd.api.types.is_string_dtype(train[col])
            or isinstance(train[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(train[col])
        )
    ]

    for col in categorical_columns:
        values = train[col].astype(str).fillna("Unknown")
        categories = {value: index for index, value in enumerate(sorted(values.unique()))}

        train[col] = values.map(categories).astype(int)
        test[col] = test[col].astype(str).fillna("Unknown").map(categories).fillna(-1).astype(int)
        encoded_columns.append(col)

    return train, test, encoded_columns


def _drop_train_constants(train, test):
    constant_columns = [col for col in train.columns if train[col].nunique(dropna=False) <= 1]
    return (
        train.drop(columns=constant_columns),
        test.drop(columns=constant_columns, errors="ignore"),
        constant_columns,
    )


def _scale_train_test(train, test):
    train = train.copy()
    test = test.copy()
    numeric_columns = train.select_dtypes(include="number").columns.tolist()

    if not numeric_columns:
        return train, test, []

    means = train[numeric_columns].mean()
    stds = train[numeric_columns].std(ddof=0).replace(0, 1)

    train[numeric_columns] = (train[numeric_columns] - means) / stds
    test[numeric_columns] = (test[numeric_columns] - means) / stds
    train[numeric_columns] = train[numeric_columns].replace([np.inf, -np.inf], 0).fillna(0)
    test[numeric_columns] = test[numeric_columns].replace([np.inf, -np.inf], 0).fillna(0)
    return train, test, numeric_columns


def ml_ready(df, test_size=0.2, random_state=42):
    if df.empty:
        raise ValueError("Cannot prepare an empty dataset for machine learning.")

    clean_df = df.copy()
    clean_df = clean_df.dropna(axis=1, how="all")
    clean_df = clean_df.loc[:, ~clean_df.columns.duplicated()]

    if len(clean_df) < 2:
        raise ValueError("At least two rows are required for a train-test split.")

    train, test = _train_test_split(clean_df, test_size, random_state)
    train, test, encoded_columns = _encode_train_test(train, test)
    train, test, dropped_columns = _drop_train_constants(train, test)
    train, test, scaled_columns = _scale_train_test(train, test)

    report = {
        "encoded_columns": encoded_columns,
        "dropped_constant_columns": dropped_columns,
        "scaled_columns": scaled_columns,
        "train_shape": train.shape,
        "test_shape": test.shape,
    }

    return train.reset_index(drop=True), test.reset_index(drop=True), report

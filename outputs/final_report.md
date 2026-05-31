# Multi-Agent AutoML Execution Report

Generated: 2026-05-31 00:20:49

## [2026-05-30 22:07:18] PIPELINE
**Action:** Starting

Input: data/home-credit-default-risk/application_train_processed.csv | Target: TARGET | Domain: credit_risk | Model: binary_classification

---

## [2026-05-30 22:07:18] PIPELINE
**Action:** Stage 1

Initializing Data Cleaner Agent

---

## [2026-05-30 22:07:22] DataCleaner
**Action:** Process Start

Loading data from data/home-credit-default-risk/application_train_processed.csv

---

## [2026-05-30 22:08:14] DataCleaner
**Action:** Tool: inspect_metadata

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns]}

---

## [2026-05-30 22:08:53] DataCleaner
**Action:** Tool Result: inspect_metadata

Success: {
  "shape": [
    307511,
    557
  ],
  "columns": [
    "SK_ID_CURR",
    "TARGET",
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "

---

## [2026-05-30 22:08:53] DataCleaner
**Action:** Label

Using supplied target_column='TARGET'

---

## [2026-05-30 22:08:55] DataCleaner
**Action:** Tool: check_label_quality

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'label_col': 'TARGET'}

---

## [2026-05-30 22:08:56] DataCleaner
**Action:** Tool Result: check_label_quality

Success: {
  "label_col": "TARGET",
  "total_rows": 307511,
  "num_classes": 2,
  "class_distribution": {
    "0": 282686,
    "1": 24825
  },
  "null_labels": 0,
  "null_label_percentage": 0.0,
  "imbalance_r

---

## [2026-05-30 22:08:56] DataCleaner
**Action:** PK

Using supplied entity_id_col='SK_ID_CURR'

---

## [2026-05-30 22:08:56] DataCleaner
**Action:** PK

Using supplied composite_key_cols=['SK_ID_CURR']

---

## [2026-05-30 22:08:58] DataCleaner
**Action:** Tool: check_pk_uniqueness

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'entity_id_col': 'SK_ID_CURR', 'composite_key_cols': ['SK_ID_CURR'], 'label_col': 'TARGET', 'identity_cols': None}

---

## [2026-05-30 22:09:18] DataCleaner
**Action:** Tool Result: check_pk_uniqueness

Success: {
  "domain_context": "Banking / fraud ML dataset. Identity integrity is critical: composite key violations \u2192 double-count labels; shared device/phone \u2192 fraud ring leakage.",
  "total_rows":

---

## [2026-05-30 22:09:20] DataCleaner
**Action:** Tool: check_column_formats

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns]}

---

## [2026-05-30 22:09:56] DataCleaner
**Action:** Tool Result: check_column_formats

Success: {
  "format_issues_found": 27,
  "columns_with_issues": {
    "bur_bb_MONTHS_BALANCE_mean_mean": [
      {
        "issue": "unexpected_negative_values",
        "detail": "92046 negative values (29.9

---

## [2026-05-30 22:10:00] DataCleaner
**Action:** Tool: detect_outliers

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'SK_ID_CURR'}

---

## [2026-05-30 22:10:01] DataCleaner
**Action:** Tool Result: detect_outliers

Success: {
  "column": "SK_ID_CURR",
  "q1": 189145.5,
  "q3": 367142.5,
  "iqr": 177997.0,
  "lower_bound": -344845.5,
  "upper_bound": 901133.5,
  "outlier_count": 0,
  "outlier_percentage": 0.0
}

---

## [2026-05-30 22:10:03] DataCleaner
**Action:** Tool: detect_outliers

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'TARGET'}

---

## [2026-05-30 22:10:03] DataCleaner
**Action:** Tool Result: detect_outliers

Success: {
  "column": "TARGET",
  "q1": 0.0,
  "q3": 0.0,
  "iqr": 0.0,
  "lower_bound": 0.0,
  "upper_bound": 0.0,
  "outlier_count": 24825,
  "outlier_percentage": 8.07
}

---

## [2026-05-30 22:10:05] DataCleaner
**Action:** Tool: detect_outliers

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'CNT_CHILDREN'}

---

## [2026-05-30 22:10:06] DataCleaner
**Action:** Tool Result: detect_outliers

Success: {
  "column": "CNT_CHILDREN",
  "q1": 0.0,
  "q3": 1.0,
  "iqr": 1.0,
  "lower_bound": -3.0,
  "upper_bound": 4.0,
  "outlier_count": 126,
  "outlier_percentage": 0.04
}

---

## [2026-05-30 22:10:08] DataCleaner
**Action:** Tool: detect_outliers

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'AMT_INCOME_TOTAL'}

---

## [2026-05-30 22:10:08] DataCleaner
**Action:** Tool Result: detect_outliers

Success: {
  "column": "AMT_INCOME_TOTAL",
  "q1": 112500.0,
  "q3": 202500.0,
  "iqr": 90000.0,
  "lower_bound": -157500.0,
  "upper_bound": 472500.0,
  "outlier_count": 3014,
  "outlier_percentage": 0.98
}

---

## [2026-05-30 22:10:12] DataCleaner
**Action:** Tool: detect_outliers

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'AMT_CREDIT'}

---

## [2026-05-30 22:10:12] DataCleaner
**Action:** Tool Result: detect_outliers

Success: {
  "column": "AMT_CREDIT",
  "q1": 270000.0,
  "q3": 808650.0,
  "iqr": 538650.0,
  "lower_bound": -1345950.0,
  "upper_bound": 2424600.0,
  "outlier_count": 444,
  "outlier_percentage": 0.14
}

---

## [2026-05-30 22:10:16] DataCleaner
**Action:** Tool: detect_outliers

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'AMT_ANNUITY'}

---

## [2026-05-30 22:10:16] DataCleaner
**Action:** Tool Result: detect_outliers

Success: {
  "column": "AMT_ANNUITY",
  "q1": 16524.0,
  "q3": 34596.0,
  "iqr": 18072.0,
  "lower_bound": -37692.0,
  "upper_bound": 88812.0,
  "outlier_count": 1024,
  "outlier_percentage": 0.33
}

---

## [2026-05-30 22:10:20] DataCleaner
**Action:** Tool: detect_outliers

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'AMT_GOODS_PRICE'}

---

## [2026-05-30 22:10:21] DataCleaner
**Action:** Tool Result: detect_outliers

Success: {
  "column": "AMT_GOODS_PRICE",
  "q1": 238500.0,
  "q3": 679500.0,
  "iqr": 441000.0,
  "lower_bound": -1084500.0,
  "upper_bound": 2002500.0,
  "outlier_count": 1090,
  "outlier_percentage": 0.35
}

---

## [2026-05-30 22:10:24] DataCleaner
**Action:** Tool: detect_outliers

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'REGION_POPULATION_RELATIVE'}

---

## [2026-05-30 22:10:24] DataCleaner
**Action:** Tool Result: detect_outliers

Success: {
  "column": "REGION_POPULATION_RELATIVE",
  "q1": 0.010006,
  "q3": 0.028663,
  "iqr": 0.0187,
  "lower_bound": -0.046,
  "upper_bound": 0.0846,
  "outlier_count": 0,
  "outlier_percentage": 0.0
}

---

## [2026-05-30 22:10:26] DataCleaner
**Action:** Tool: detect_outliers

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'DAYS_BIRTH'}

---

## [2026-05-30 22:10:27] DataCleaner
**Action:** Tool Result: detect_outliers

Success: {
  "column": "DAYS_BIRTH",
  "q1": -19682.0,
  "q3": -12413.0,
  "iqr": 7269.0,
  "lower_bound": -41489.0,
  "upper_bound": 9394.0,
  "outlier_count": 0,
  "outlier_percentage": 0.0
}

---

## [2026-05-30 22:10:29] DataCleaner
**Action:** Tool: detect_outliers

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'DAYS_EMPLOYED'}

---

## [2026-05-30 22:10:29] DataCleaner
**Action:** Tool Result: detect_outliers

Success: {
  "column": "DAYS_EMPLOYED",
  "q1": -2760.0,
  "q3": -289.0,
  "iqr": 2471.0,
  "lower_bound": -10173.0,
  "upper_bound": 7124.0,
  "outlier_count": 59624,
  "outlier_percentage": 19.39
}

---

## [2026-05-30 22:10:31] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_ENDDATE_mean'}

---

## [2026-05-30 22:10:32] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_ENDDATE_mean",
  "min_date": "1969-12-31 23:59:59.999958125",
  "max_date": "1970-01-01 00:00:00.000031198",
  "date_range_days": 0,
  "distinct_dates": 15526,
  "null

---

## [2026-05-30 22:10:34] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_ENDDATE_sum'}

---

## [2026-05-30 22:10:35] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_ENDDATE_sum",
  "min_date": "1969-12-31 23:59:59.999844729",
  "max_date": "1970-01-01 00:00:00.000214193",
  "date_range_days": 0,
  "distinct_dates": 45664,
  "null_

---

## [2026-05-30 22:10:37] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_ENDDATE_max'}

---

## [2026-05-30 22:10:38] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_ENDDATE_max",
  "min_date": "1969-12-31 23:59:59.999958125",
  "max_date": "1970-01-01 00:00:00.000031199",
  "date_range_days": 0,
  "distinct_dates": 13030,
  "null_

---

## [2026-05-30 22:10:40] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_ENDDATE_min'}

---

## [2026-05-30 22:10:41] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_ENDDATE_min",
  "min_date": "1969-12-31 23:59:59.999957940",
  "max_date": "1970-01-01 00:00:00.000031198",
  "date_range_days": 0,
  "distinct_dates": 6828,
  "null_d

---

## [2026-05-30 22:10:43] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_ENDDATE_std'}

---

## [2026-05-30 22:10:43] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_ENDDATE_std",
  "min_date": "1970-01-01 00:00:00",
  "max_date": "1970-01-01 00:00:00.000038718",
  "date_range_days": 0,
  "distinct_dates": 17934,
  "null_dates": 84

---

## [2026-05-30 22:10:45] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_ENDDATE_count'}

---

## [2026-05-30 22:10:46] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_ENDDATE_count",
  "min_date": "1970-01-01 00:00:00",
  "max_date": "1970-01-01 00:00:00.000000107",
  "date_range_days": 0,
  "distinct_dates": 59,
  "null_dates": 440

---

## [2026-05-30 22:10:48] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_ENDDATE_FACT_mean'}

---

## [2026-05-30 22:10:48] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_ENDDATE_FACT_mean",
  "min_date": "1969-12-31 23:59:59.999991624",
  "max_date": "1970-01-01 00:00:00",
  "date_range_days": 0,
  "distinct_dates": 2812,
  "null_dates": 7715

---

## [2026-05-30 22:10:51] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_ENDDATE_FACT_sum'}

---

## [2026-05-30 22:10:51] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_ENDDATE_FACT_sum",
  "min_date": "1969-12-31 23:59:59.999925354",
  "max_date": "1970-01-01 00:00:00",
  "date_range_days": 0,
  "distinct_dates": 18826,
  "null_dates": 4402

---

## [2026-05-30 22:10:53] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_ENDDATE_FACT_max'}

---

## [2026-05-30 22:10:54] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_ENDDATE_FACT_max",
  "min_date": "1969-12-31 23:59:59.999997113",
  "max_date": "1970-01-01 00:00:00",
  "date_range_days": 0,
  "distinct_dates": 2807,
  "null_dates": 77156

---

## [2026-05-30 22:10:56] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_ENDDATE_FACT_min'}

---

## [2026-05-30 22:10:56] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_ENDDATE_FACT_min",
  "min_date": "1969-12-31 23:59:59.999957977",
  "max_date": "1970-01-01 00:00:00",
  "date_range_days": 0,
  "distinct_dates": 2916,
  "null_dates": 77156

---

## [2026-05-30 22:10:59] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_ENDDATE_FACT_std'}

---

## [2026-05-30 22:10:59] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_ENDDATE_FACT_std",
  "min_date": "1970-01-01 00:00:00",
  "max_date": "1970-01-01 00:00:00.000016511",
  "date_range_days": 0,
  "distinct_dates": 1807,
  "null_dates": 13052

---

## [2026-05-30 22:11:01] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_ENDDATE_FACT_count'}

---

## [2026-05-30 22:11:02] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_ENDDATE_FACT_count",
  "min_date": "1970-01-01 00:00:00",
  "max_date": "1970-01-01 00:00:00.000000108",
  "date_range_days": 0,
  "distinct_dates": 54,
  "null_dates": 44020

---

## [2026-05-30 22:11:04] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_UPDATE_mean'}

---

## [2026-05-30 22:11:04] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_UPDATE_mean",
  "min_date": "1969-12-31 23:59:59.999958110",
  "max_date": "1970-01-01 00:00:00.000000014",
  "date_range_days": 0,
  "distinct_dates": 2762,
  "null_d

---

## [2026-05-30 22:11:07] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_UPDATE_sum'}

---

## [2026-05-30 22:11:07] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_UPDATE_sum",
  "min_date": "1969-12-31 23:59:59.999909935",
  "max_date": "1970-01-01 00:00:00.000000014",
  "date_range_days": 0,
  "distinct_dates": 17724,
  "null_d

---

## [2026-05-30 22:11:09] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_UPDATE_max'}

---

## [2026-05-30 22:11:10] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_UPDATE_max",
  "min_date": "1969-12-31 23:59:59.999958110",
  "max_date": "1970-01-01 00:00:00.000000372",
  "date_range_days": 0,
  "distinct_dates": 2663,
  "null_da

---

## [2026-05-30 22:11:12] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_UPDATE_min'}

---

## [2026-05-30 22:11:12] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_UPDATE_min",
  "min_date": "1969-12-31 23:59:59.999958053",
  "max_date": "1970-01-01 00:00:00.000000014",
  "date_range_days": 0,
  "distinct_dates": 2969,
  "null_da

---

## [2026-05-30 22:11:14] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_UPDATE_std'}

---

## [2026-05-30 22:11:15] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_UPDATE_std",
  "min_date": "1970-01-01 00:00:00",
  "max_date": "1970-01-01 00:00:00.000029612",
  "date_range_days": 0,
  "distinct_dates": 1916,
  "null_dates": 8009

---

## [2026-05-30 22:11:17] DataCleaner
**Action:** Tool: check_temporal

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'date_col': 'bur_DAYS_CREDIT_UPDATE_count'}

---

## [2026-05-30 22:11:18] DataCleaner
**Action:** Tool Result: check_temporal

Success: {
  "date_col": "bur_DAYS_CREDIT_UPDATE_count",
  "min_date": "1970-01-01 00:00:00.000000001",
  "max_date": "1970-01-01 00:00:00.000000116",
  "date_range_days": 0,
  "distinct_dates": 60,
  "null_da

---

## [2026-05-30 22:11:18] DataCleaner
**Action:** LLM Call

model=cloud-model | prompt_len=23410 chars | temp=0.2 | top_p=0.95 | max_tokens=4000 | json_mode=True

---

## [2026-05-30 22:11:33] DataCleaner
**Action:** ERROR

Cannot reach LiteLLM proxy at http://localhost:4000: Connection error.

---

## [2026-05-30 22:11:34] DataCleaner
**Action:** LLM Fallback

Trying Claude (claude-opus-4-7) as last resort...

---

## [2026-05-30 22:12:03] DataCleaner
**Action:** LLM Tokens

model=claude-opus-4-7 | input=15033 | output=2358 | total=17391

---

## [2026-05-30 22:12:03] DataCleaner
**Action:** LLM Response

Claude (claude-opus-4-7) | 4654 chars | fallback=claude

---

## [2026-05-30 22:12:03] DataCleaner
**Action:** LLM Decision

Parsing decisions from LLM response

---

## [2026-05-30 22:12:03] DataCleaner
**Action:** LLM Reasoning

Dataset has 557 columns with 36 columns having >80% nulls (drop candidates) and 2 constant columns (cc_SK_DPD_min, cc_SK_DPD_DEF_min). No exact duplicates, no PK violations, no composite key issues. Label imbalance ratio 11.39 is below 20 threshold (acceptable). Negative values in MONTHS_BALANCE columns are EXPECTED (Home Credit dataset encodes time relative to application as negative days/months) — not errors, should NOT be clipped. DAYS_EMPLOYED 19% outliers are the known 365243 sentinel issue but clipping may distort; leaving as-is since it's a known domain artifact. Temporal leakage flag for pos_CNT_INSTALMENT_FUTURE_* noted but these are aggregated historical features, flagging only. No format casting needed — all numerics properly typed.

---

## [2026-05-30 22:12:03] DataCleaner
**Action:** Action: drop_column

Column: cc_SK_DPD_min, Reason: Constant column — single unique value

---

## [2026-05-30 22:12:05] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 557 columns], 'col': 'cc_SK_DPD_min'}

---

## [2026-05-30 22:12:10] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DE

---

## [2026-05-30 22:12:10] DataCleaner
**Action:** Action: drop_column

Column: cc_SK_DPD_DEF_min, Reason: Constant column — single unique value

---

## [2026-05-30 22:12:12] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_min  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...                 0.0                0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...                 ...                ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...                 NaN                NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...                 NaN                NaN                NaN                NaN                NaN                 NaN

[307511 rows x 556 columns], 'col': 'cc_SK_DPD_DEF_min'}

---

## [2026-05-30 22:12:16] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:12:17] DataCleaner
**Action:** Action: drop_column

Column: prev_RATE_INTEREST_PRIMARY_mean, Reason: 98.5% null

---

## [2026-05-30 22:12:19] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 555 columns], 'col': 'prev_RATE_INTEREST_PRIMARY_mean'}

---

## [2026-05-30 22:12:23] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:12:24] DataCleaner
**Action:** Action: drop_column

Column: prev_RATE_INTEREST_PRIMARY_max, Reason: 98.5% null

---

## [2026-05-30 22:12:26] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 554 columns], 'col': 'prev_RATE_INTEREST_PRIMARY_max'}

---

## [2026-05-30 22:12:30] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:12:31] DataCleaner
**Action:** Action: drop_column

Column: prev_RATE_INTEREST_PRIMARY_min, Reason: 98.5% null

---

## [2026-05-30 22:12:33] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 553 columns], 'col': 'prev_RATE_INTEREST_PRIMARY_min'}

---

## [2026-05-30 22:12:38] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:12:38] DataCleaner
**Action:** Action: drop_column

Column: prev_RATE_INTEREST_PRIMARY_std, Reason: 99.94% null

---

## [2026-05-30 22:12:40] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 552 columns], 'col': 'prev_RATE_INTEREST_PRIMARY_std'}

---

## [2026-05-30 22:12:45] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:12:46] DataCleaner
**Action:** Action: drop_column

Column: prev_RATE_INTEREST_PRIVILEGED_mean, Reason: 98.5% null

---

## [2026-05-30 22:12:48] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 551 columns], 'col': 'prev_RATE_INTEREST_PRIVILEGED_mean'}

---

## [2026-05-30 22:12:53] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:12:54] DataCleaner
**Action:** Action: drop_column

Column: prev_RATE_INTEREST_PRIVILEGED_max, Reason: 98.5% null

---

## [2026-05-30 22:12:56] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 550 columns], 'col': 'prev_RATE_INTEREST_PRIVILEGED_max'}

---

## [2026-05-30 22:13:01] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:13:01] DataCleaner
**Action:** Action: drop_column

Column: prev_RATE_INTEREST_PRIVILEGED_min, Reason: 98.5% null

---

## [2026-05-30 22:13:04] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 549 columns], 'col': 'prev_RATE_INTEREST_PRIVILEGED_min'}

---

## [2026-05-30 22:13:09] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:13:09] DataCleaner
**Action:** Action: drop_column

Column: prev_RATE_INTEREST_PRIVILEGED_std, Reason: 99.94% null

---

## [2026-05-30 22:13:12] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 548 columns], 'col': 'prev_RATE_INTEREST_PRIVILEGED_std'}

---

## [2026-05-30 22:13:16] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:13:16] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_ATM_CURRENT_mean, Reason: 80.12% null

---

## [2026-05-30 22:13:18] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 547 columns], 'col': 'cc_AMT_DRAWINGS_ATM_CURRENT_mean'}

---

## [2026-05-30 22:13:22] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:13:23] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_ATM_CURRENT_max, Reason: 80.12% null

---

## [2026-05-30 22:13:25] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 546 columns], 'col': 'cc_AMT_DRAWINGS_ATM_CURRENT_max'}

---

## [2026-05-30 22:13:28] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:13:28] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_ATM_CURRENT_min, Reason: 80.12% null

---

## [2026-05-30 22:13:31] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 545 columns], 'col': 'cc_AMT_DRAWINGS_ATM_CURRENT_min'}

---

## [2026-05-30 22:13:34] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:13:35] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_ATM_CURRENT_std, Reason: 80.26% null

---

## [2026-05-30 22:13:37] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 544 columns], 'col': 'cc_AMT_DRAWINGS_ATM_CURRENT_std'}

---

## [2026-05-30 22:13:41] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:13:42] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_OTHER_CURRENT_mean, Reason: 80.12% null

---

## [2026-05-30 22:13:44] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 543 columns], 'col': 'cc_AMT_DRAWINGS_OTHER_CURRENT_mean'}

---

## [2026-05-30 22:13:48] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:13:48] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_OTHER_CURRENT_max, Reason: 80.12% null

---

## [2026-05-30 22:13:51] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 542 columns], 'col': 'cc_AMT_DRAWINGS_OTHER_CURRENT_max'}

---

## [2026-05-30 22:13:54] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:13:55] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_OTHER_CURRENT_min, Reason: 80.12% null

---

## [2026-05-30 22:13:57] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 541 columns], 'col': 'cc_AMT_DRAWINGS_OTHER_CURRENT_min'}

---

## [2026-05-30 22:14:01] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:14:01] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_OTHER_CURRENT_std, Reason: 80.26% null

---

## [2026-05-30 22:14:03] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 540 columns], 'col': 'cc_AMT_DRAWINGS_OTHER_CURRENT_std'}

---

## [2026-05-30 22:14:08] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:14:08] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_POS_CURRENT_mean, Reason: 80.12% null

---

## [2026-05-30 22:14:11] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 539 columns], 'col': 'cc_AMT_DRAWINGS_POS_CURRENT_mean'}

---

## [2026-05-30 22:14:15] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:14:16] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_POS_CURRENT_max, Reason: 80.12% null

---

## [2026-05-30 22:14:18] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 538 columns], 'col': 'cc_AMT_DRAWINGS_POS_CURRENT_max'}

---

## [2026-05-30 22:14:22] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:14:23] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_POS_CURRENT_min, Reason: 80.12% null

---

## [2026-05-30 22:14:26] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 537 columns], 'col': 'cc_AMT_DRAWINGS_POS_CURRENT_min'}

---

## [2026-05-30 22:14:29] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:14:30] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_DRAWINGS_POS_CURRENT_std, Reason: 80.26% null

---

## [2026-05-30 22:14:32] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 536 columns], 'col': 'cc_AMT_DRAWINGS_POS_CURRENT_std'}

---

## [2026-05-30 22:14:36] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:14:36] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_PAYMENT_CURRENT_mean, Reason: 80.14% null

---

## [2026-05-30 22:14:39] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 535 columns], 'col': 'cc_AMT_PAYMENT_CURRENT_mean'}

---

## [2026-05-30 22:14:43] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:14:44] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_PAYMENT_CURRENT_max, Reason: 80.14% null

---

## [2026-05-30 22:14:46] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 534 columns], 'col': 'cc_AMT_PAYMENT_CURRENT_max'}

---

## [2026-05-30 22:14:51] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:14:51] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_PAYMENT_CURRENT_min, Reason: 80.14% null

---

## [2026-05-30 22:14:53] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 533 columns], 'col': 'cc_AMT_PAYMENT_CURRENT_min'}

---

## [2026-05-30 22:14:57] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:14:58] DataCleaner
**Action:** Action: drop_column

Column: cc_AMT_PAYMENT_CURRENT_std, Reason: 80.29% null

---

## [2026-05-30 22:15:00] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 532 columns], 'col': 'cc_AMT_PAYMENT_CURRENT_std'}

---

## [2026-05-30 22:15:05] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:15:05] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_ATM_CURRENT_mean, Reason: 80.12% null

---

## [2026-05-30 22:15:07] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 531 columns], 'col': 'cc_CNT_DRAWINGS_ATM_CURRENT_mean'}

---

## [2026-05-30 22:15:11] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:15:12] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_ATM_CURRENT_max, Reason: 80.12% null

---

## [2026-05-30 22:15:14] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 530 columns], 'col': 'cc_CNT_DRAWINGS_ATM_CURRENT_max'}

---

## [2026-05-30 22:15:19] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:15:20] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_ATM_CURRENT_min, Reason: 80.12% null

---

## [2026-05-30 22:15:22] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 529 columns], 'col': 'cc_CNT_DRAWINGS_ATM_CURRENT_min'}

---

## [2026-05-30 22:15:26] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:15:27] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_ATM_CURRENT_std, Reason: 80.26% null

---

## [2026-05-30 22:15:29] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 528 columns], 'col': 'cc_CNT_DRAWINGS_ATM_CURRENT_std'}

---

## [2026-05-30 22:15:34] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:15:34] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_OTHER_CURRENT_mean, Reason: 80.12% null

---

## [2026-05-30 22:15:37] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 527 columns], 'col': 'cc_CNT_DRAWINGS_OTHER_CURRENT_mean'}

---

## [2026-05-30 22:15:41] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:15:42] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_OTHER_CURRENT_max, Reason: 80.12% null

---

## [2026-05-30 22:15:44] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 526 columns], 'col': 'cc_CNT_DRAWINGS_OTHER_CURRENT_max'}

---

## [2026-05-30 22:15:49] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:15:50] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_OTHER_CURRENT_min, Reason: 80.12% null

---

## [2026-05-30 22:15:53] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 525 columns], 'col': 'cc_CNT_DRAWINGS_OTHER_CURRENT_min'}

---

## [2026-05-30 22:15:56] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:15:57] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_OTHER_CURRENT_std, Reason: 80.26% null

---

## [2026-05-30 22:15:59] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 524 columns], 'col': 'cc_CNT_DRAWINGS_OTHER_CURRENT_std'}

---

## [2026-05-30 22:16:03] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:16:03] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_POS_CURRENT_mean, Reason: 80.12% null

---

## [2026-05-30 22:16:05] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 523 columns], 'col': 'cc_CNT_DRAWINGS_POS_CURRENT_mean'}

---

## [2026-05-30 22:16:10] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:16:10] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_POS_CURRENT_max, Reason: 80.12% null

---

## [2026-05-30 22:16:13] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 522 columns], 'col': 'cc_CNT_DRAWINGS_POS_CURRENT_max'}

---

## [2026-05-30 22:16:17] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:16:17] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_POS_CURRENT_min, Reason: 80.12% null

---

## [2026-05-30 22:16:19] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 521 columns], 'col': 'cc_CNT_DRAWINGS_POS_CURRENT_min'}

---

## [2026-05-30 22:16:23] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:16:23] DataCleaner
**Action:** Action: drop_column

Column: cc_CNT_DRAWINGS_POS_CURRENT_std, Reason: 80.26% null

---

## [2026-05-30 22:16:26] DataCleaner
**Action:** Tool: drop_column

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 520 columns], 'col': 'cc_CNT_DRAWINGS_POS_CURRENT_std'}

---

## [2026-05-30 22:16:29] DataCleaner
**Action:** Tool Result: drop_column

Success:         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_

---

## [2026-05-30 22:27:27] DataCleaner
**Action:** Data Saved

Cleaned data saved to outputs/clean_data.csv

---

## [2026-05-30 22:27:27] DataCleaner
**Action:** Report Saved

Saved to outputs/data_cleaner_report.json

---

## [2026-05-30 22:27:27] DataCleaner
**Action:** Process Complete

Shape: (307511, 557) -> (307511, 519)

---

## [2026-05-30 22:27:28] HANDOFF
**Action:** Data Transfer from DataCleaner

Data: outputs/clean_data.csv, Report keys: ['agent', 'original_shape', 'final_shape', 'actions_taken', 'summary', 'columns_remaining', 'entity_id_col', 'composite_key_cols', 'target_column']

---

## [2026-05-30 22:27:28] PIPELINE
**Action:** Stage 2

Initializing Feature Engineer Agent

---

## [2026-05-30 22:27:30] FeatureEngineer
**Action:** Col Descriptions Loaded

Loaded 196 descriptions | name_field='Row' desc_field='Description' group_field='Table'

---

## [2026-05-30 22:29:03] FeatureEngineer
**Action:** Process Start

Received clean data with shape (307511, 519)

---

## [2026-05-30 22:29:05] FeatureEngineer
**Action:** Protected Cols

Skipping feature engineering for: ['SK_ID_CURR']

---

## [2026-05-30 22:29:05] FeatureEngineer
**Action:** Previous Agent Summary

Performed 38 cleaning actions: Dropped column 'cc_SK_DPD_min': Constant column — single unique value; Dropped column 'cc_SK_DPD_DEF_min': Constant column — single unique value; Dropped column 'prev_RATE_INTEREST_PRIMARY_mean': 98.5% null; and 35 more actions.

---

## [2026-05-30 22:29:28] FeatureEngineer
**Action:** LLM Call

model=cloud-model | prompt_len=73793 chars | temp=0.2 | top_p=0.95 | max_tokens=4000 | json_mode=True

---

## [2026-05-30 22:29:41] FeatureEngineer
**Action:** ERROR

Cannot reach LiteLLM proxy at http://localhost:4000: Connection error.

---

## [2026-05-30 22:29:41] FeatureEngineer
**Action:** LLM Fallback

Trying Claude (claude-opus-4-7) as last resort...

---

## [2026-05-30 22:30:06] FeatureEngineer
**Action:** LLM Tokens

model=claude-opus-4-7 | input=35437 | output=2060 | total=37497

---

## [2026-05-30 22:30:06] FeatureEngineer
**Action:** LLM Response

Claude (claude-opus-4-7) | 3925 chars | fallback=claude

---

## [2026-05-30 22:30:06] FeatureEngineer
**Action:** LLM Decision

Parsing feature engineering decisions

---

## [2026-05-30 22:30:06] FeatureEngineer
**Action:** LLM Reasoning

Encode all categoricals first. Create domain-driven interactions: credit/income, annuity/income, credit/goods, employment ratios, EXT_SOURCE combinations (strongest predictors), bureau debt ratios, overdue rates. Then correlation analysis and select top 350 features (70% cap).

---

## [2026-05-30 22:30:06] FeatureEngineer
**Action:** Action: encode_all_categorical

Encode 16 categorical columns before interactions and selection

---

## [2026-05-30 22:30:09] FeatureEngineer
**Action:** Tool: encode_all_categorical

Parameters: {'df':         SK_ID_CURR  TARGET NAME_CONTRACT_TYPE CODE_GENDER FLAG_OWN_CAR FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std cc_SK_DPD_DEF_count
0           100002       1         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
1           100003       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
2           100004       0    Revolving loans           M            Y               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
3           100006       0         Cash loans           F            N               Y  ...              6.0                 0.0                0.0                0.0                0.0                 6.0
4           100007       0         Cash loans           M            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
...            ...     ...                ...         ...          ...             ...  ...              ...                 ...                ...                ...                ...                 ...
307506      456251       0         Cash loans           M            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307507      456252       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307508      456253       0         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307509      456254       1         Cash loans           F            N               Y  ...              NaN                 NaN                NaN                NaN                NaN                 NaN
307510      456255       0         Cash loans           F            N               N  ...              NaN                 NaN                NaN                NaN                NaN                 NaN

[307511 rows x 519 columns], 'method': 'label'}

---

## [2026-05-30 22:30:20] FeatureEngineer
**Action:** encode_all_categorical

Encoded 16 categorical columns (requested method='label')

---

## [2026-05-30 22:30:23] FeatureEngineer
**Action:** Tool Result: encode_all_categorical

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std  cc_SK_DPD

---

## [2026-05-30 22:30:23] FeatureEngineer
**Action:** Action: create_interaction

Debt burden: credit amount vs income

---

## [2026-05-30 22:30:26] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_count  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std  cc_SK_DPD_DEF_count
0           100002       1                   0            1             0                1  ...              NaN                 NaN                NaN                NaN                NaN                  NaN
1           100003       0                   0            0             0                0  ...              NaN                 NaN                NaN                NaN                NaN                  NaN
2           100004       0                   1            1             1                1  ...              NaN                 NaN                NaN                NaN                NaN                  NaN
3           100006       0                   0            0             0                1  ...              6.0                 0.0                0.0                0.0                0.0                  6.0
4           100007       0                   0            1             0                1  ...              NaN                 NaN                NaN                NaN                NaN                  NaN
...            ...     ...                 ...          ...           ...              ...  ...              ...                 ...                ...                ...                ...                  ...
307506      456251       0                   0            1             0                0  ...              NaN                 NaN                NaN                NaN                NaN                  NaN
307507      456252       0                   0            0             0                1  ...              NaN                 NaN                NaN                NaN                NaN                  NaN
307508      456253       0                   0            0             0                1  ...              NaN                 NaN                NaN                NaN                NaN                  NaN
307509      456254       1                   0            0             0                1  ...              NaN                 NaN                NaN                NaN                NaN                  NaN
307510      456255       0                   0            0             0                0  ...              NaN                 NaN                NaN                NaN                NaN                  NaN

[307511 rows x 519 columns], 'new_col': 'CREDIT_INCOME_RATIO', 'expression': "df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)"}

---

## [2026-05-30 22:30:30] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std  cc_SK_DPD_DEF_count  CREDI

---

## [2026-05-30 22:30:30] FeatureEngineer
**Action:** Action: create_interaction

Installment burden vs income

---

## [2026-05-30 22:30:35] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_mean  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std  cc_SK_DPD_DEF_count  CREDIT_INCOME_RATIO
0           100002       1                   0            1             0                1  ...                 NaN                NaN                NaN                NaN                  NaN             2.007879
1           100003       0                   0            0             0                0  ...                 NaN                NaN                NaN                NaN                  NaN             4.790732
2           100004       0                   1            1             1                1  ...                 NaN                NaN                NaN                NaN                  NaN             1.999970
3           100006       0                   0            0             0                1  ...                 0.0                0.0                0.0                0.0                  6.0             2.316150
4           100007       0                   0            1             0                1  ...                 NaN                NaN                NaN                NaN                  NaN             4.222187
...            ...     ...                 ...          ...           ...              ...  ...                 ...                ...                ...                ...                  ...                  ...
307506      456251       0                   0            1             0                0  ...                 NaN                NaN                NaN                NaN                  NaN             1.617133
307507      456252       0                   0            0             0                1  ...                 NaN                NaN                NaN                NaN                  NaN             3.743698
307508      456253       0                   0            0             0                1  ...                 NaN                NaN                NaN                NaN                  NaN             4.429148
307509      456254       1                   0            0             0                1  ...                 NaN                NaN                NaN                NaN                  NaN             2.164356
307510      456255       0                   0            0             0                0  ...                 NaN                NaN                NaN                NaN                  NaN             4.285687

[307511 rows x 520 columns], 'new_col': 'ANNUITY_INCOME_RATIO', 'expression': "df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)"}

---

## [2026-05-30 22:30:39] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std  cc_SK_DPD_DEF_count  CREDIT_INCOME_RATIO  ANNU

---

## [2026-05-30 22:30:39] FeatureEngineer
**Action:** Action: create_interaction

Loan-to-value proxy

---

## [2026-05-30 22:30:44] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_sum  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std  cc_SK_DPD_DEF_count  CREDIT_INCOME_RATIO  ANNUITY_INCOME_RATIO
0           100002       1                   0            1             0                1  ...                NaN                NaN                NaN                  NaN             2.007879              0.121977
1           100003       0                   0            0             0                0  ...                NaN                NaN                NaN                  NaN             4.790732              0.132216
2           100004       0                   1            1             1                1  ...                NaN                NaN                NaN                  NaN             1.999970              0.099999
3           100006       0                   0            0             0                1  ...                0.0                0.0                0.0                  6.0             2.316150              0.219898
4           100007       0                   0            1             0                1  ...                NaN                NaN                NaN                  NaN             4.222187              0.179961
...            ...     ...                 ...          ...           ...              ...  ...                ...                ...                ...                  ...                  ...                   ...
307506      456251       0                   0            1             0                0  ...                NaN                NaN                NaN                  NaN             1.617133              0.174970
307507      456252       0                   0            0             0                1  ...                NaN                NaN                NaN                  NaN             3.743698              0.166685
307508      456253       0                   0            0             0                1  ...                NaN                NaN                NaN                  NaN             4.429148              0.195940
307509      456254       1                   0            0             0                1  ...                NaN                NaN                NaN                  NaN             2.164356              0.118157
307510      456255       0                   0            0             0                0  ...                NaN                NaN                NaN                  NaN             4.285687              0.311855

[307511 rows x 521 columns], 'new_col': 'CREDIT_GOODS_RATIO', 'expression': "df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)"}

---

## [2026-05-30 22:30:50] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std  cc_SK_DPD_DEF_count  CREDIT_INCOME_RATIO  ANNUITY_INCOME_RATIO  C

---

## [2026-05-30 22:30:50] FeatureEngineer
**Action:** Action: create_interaction

Effective loan term inverse

---

## [2026-05-30 22:30:54] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_max  cc_SK_DPD_DEF_std  cc_SK_DPD_DEF_count  CREDIT_INCOME_RATIO  ANNUITY_INCOME_RATIO  CREDIT_GOODS_RATIO
0           100002       1                   0            1             0                1  ...                NaN                NaN                  NaN             2.007879              0.121977            1.158394
1           100003       0                   0            0             0                0  ...                NaN                NaN                  NaN             4.790732              0.132216            1.145198
2           100004       0                   1            1             1                1  ...                NaN                NaN                  NaN             1.999970              0.099999            0.999993
3           100006       0                   0            0             0                1  ...                0.0                0.0                  6.0             2.316150              0.219898            1.052799
4           100007       0                   0            1             0                1  ...                NaN                NaN                  NaN             4.222187              0.179961            0.999998
...            ...     ...                 ...          ...           ...              ...  ...                ...                ...                  ...                  ...                   ...                 ...
307506      456251       0                   0            1             0                0  ...                NaN                NaN                  NaN             1.617133              0.174970            1.131995
307507      456252       0                   0            0             0                1  ...                NaN                NaN                  NaN             3.743698              0.166685            1.197995
307508      456253       0                   0            0             0                1  ...                NaN                NaN                  NaN             4.429148              0.195940            1.158398
307509      456254       1                   0            0             0                1  ...                NaN                NaN                  NaN             2.164356              0.118157            1.158391
307510      456255       0                   0            0             0                0  ...                NaN                NaN                  NaN             4.285687              0.311855            0.999999

[307511 rows x 522 columns], 'new_col': 'ANNUITY_CREDIT_RATIO', 'expression': "df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)"}

---

## [2026-05-30 22:30:58] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_std  cc_SK_DPD_DEF_count  CREDIT_INCOME_RATIO  ANNUITY_INCOME_RATIO  CREDIT_GOODS_RATIO  

---

## [2026-05-30 22:30:58] FeatureEngineer
**Action:** Action: create_interaction

Per-capita income

---

## [2026-05-30 22:31:03] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_std  cc_SK_DPD_DEF_count  CREDIT_INCOME_RATIO  ANNUITY_INCOME_RATIO  CREDIT_GOODS_RATIO  ANNUITY_CREDIT_RATIO
0           100002       1                   0            1             0                1  ...                NaN                  NaN             2.007879              0.121977            1.158394              0.060749
1           100003       0                   0            0             0                0  ...                NaN                  NaN             4.790732              0.132216            1.145198              0.027598
2           100004       0                   1            1             1                1  ...                NaN                  NaN             1.999970              0.099999            0.999993              0.050000
3           100006       0                   0            0             0                1  ...                0.0                  6.0             2.316150              0.219898            1.052799              0.094941
4           100007       0                   0            1             0                1  ...                NaN                  NaN             4.222187              0.179961            0.999998              0.042623
...            ...     ...                 ...          ...           ...              ...  ...                ...                  ...                  ...                   ...                 ...                   ...
307506      456251       0                   0            1             0                0  ...                NaN                  NaN             1.617133              0.174970            1.131995              0.108197
307507      456252       0                   0            0             0                1  ...                NaN                  NaN             3.743698              0.166685            1.197995              0.044524
307508      456253       0                   0            0             0                1  ...                NaN                  NaN             4.429148              0.195940            1.158398              0.044239
307509      456254       1                   0            0             0                1  ...                NaN                  NaN             2.164356              0.118157            1.158391              0.054592
307510      456255       0                   0            0             0                0  ...                NaN                  NaN             4.285687              0.311855            0.999999              0.072767

[307511 rows x 523 columns], 'new_col': 'INCOME_PER_FAM', 'expression': "df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)"}

---

## [2026-05-30 22:31:07] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_count  CREDIT_INCOME_RATIO  ANNUITY_INCOME_RATIO  CREDIT_GOODS_RATIO  ANNUITY_CREDIT_RATI

---

## [2026-05-30 22:31:07] FeatureEngineer
**Action:** Action: create_interaction

Employment tenure relative to age

---

## [2026-05-30 22:31:11] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  cc_SK_DPD_DEF_count  CREDIT_INCOME_RATIO  ANNUITY_INCOME_RATIO  CREDIT_GOODS_RATIO  ANNUITY_CREDIT_RATIO  INCOME_PER_FAM
0           100002       1                   0            1             0                1  ...                  NaN             2.007879              0.121977            1.158394              0.060749        101250.0
1           100003       0                   0            0             0                0  ...                  NaN             4.790732              0.132216            1.145198              0.027598         90000.0
2           100004       0                   1            1             1                1  ...                  NaN             1.999970              0.099999            0.999993              0.050000         33750.0
3           100006       0                   0            0             0                1  ...                  6.0             2.316150              0.219898            1.052799              0.094941         45000.0
4           100007       0                   0            1             0                1  ...                  NaN             4.222187              0.179961            0.999998              0.042623         60750.0
...            ...     ...                 ...          ...           ...              ...  ...                  ...                  ...                   ...                 ...                   ...             ...
307506      456251       0                   0            1             0                0  ...                  NaN             1.617133              0.174970            1.131995              0.108197         78750.0
307507      456252       0                   0            0             0                1  ...                  NaN             3.743698              0.166685            1.197995              0.044524         36000.0
307508      456253       0                   0            0             0                1  ...                  NaN             4.429148              0.195940            1.158398              0.044239         76500.0
307509      456254       1                   0            0             0                1  ...                  NaN             2.164356              0.118157            1.158391              0.054592         57000.0
307510      456255       0                   0            0             0                0  ...                  NaN             4.285687              0.311855            0.999999              0.072767         52500.0

[307511 rows x 524 columns], 'new_col': 'EMPLOYED_BIRTH_RATIO', 'expression': "df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] - 1)"}

---

## [2026-05-30 22:31:15] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  CREDIT_INCOME_RATIO  ANNUITY_INCOME_RATIO  CREDIT_GOODS_RATIO  ANNUITY_CREDIT_RATIO  INCOME_PER_FAM  EM

---

## [2026-05-30 22:31:15] FeatureEngineer
**Action:** Action: create_interaction

Combined external risk score - strongest predictor

---

## [2026-05-30 22:31:20] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  CREDIT_INCOME_RATIO  ANNUITY_INCOME_RATIO  CREDIT_GOODS_RATIO  ANNUITY_CREDIT_RATIO  INCOME_PER_FAM  EMPLOYED_BIRTH_RATIO
0           100002       1                   0            1             0                1  ...             2.007879              0.121977            1.158394              0.060749        101250.0              0.067322
1           100003       0                   0            0             0                0  ...             4.790732              0.132216            1.145198              0.027598         90000.0              0.070858
2           100004       0                   1            1             1                1  ...             1.999970              0.099999            0.999993              0.050000         33750.0              0.011813
3           100006       0                   0            0             0                1  ...             2.316150              0.219898            1.052799              0.094941         45000.0              0.159897
4           100007       0                   0            1             0                1  ...             4.222187              0.179961            0.999998              0.042623         60750.0              0.152411
...            ...     ...                 ...          ...           ...              ...  ...                  ...                   ...                 ...                   ...             ...                   ...
307506      456251       0                   0            1             0                0  ...             1.617133              0.174970            1.131995              0.108197         78750.0              0.025300
307507      456252       0                   0            0             0                1  ...             3.743698              0.166685            1.197995              0.044524         36000.0            -17.580044
307508      456253       0                   0            0             0                1  ...             4.429148              0.195940            1.158398              0.044239         76500.0              0.529231
307509      456254       1                   0            0             0                1  ...             2.164356              0.118157            1.158391              0.054592         57000.0              0.400100
307510      456255       0                   0            0             0                0  ...             4.285687              0.311855            0.999999              0.072767         52500.0              0.074865

[307511 rows x 525 columns], 'new_col': 'EXT_SOURCES_MEAN', 'expression': "(df['EXT_SOURCE_1'].fillna(0.5) + df['EXT_SOURCE_2'].fillna(0.5) + df['EXT_SOURCE_3'].fillna(0.5)) / 3"}

---

## [2026-05-30 22:31:24] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  ANNUITY_INCOME_RATIO  CREDIT_GOODS_RATIO  ANNUITY_CREDIT_RATIO  INCOME_PER_FAM  EMPLOYED_BIRTH_RATIO  E

---

## [2026-05-30 22:31:24] FeatureEngineer
**Action:** Action: create_interaction

Interaction of external scores

---

## [2026-05-30 22:31:28] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  ANNUITY_INCOME_RATIO  CREDIT_GOODS_RATIO  ANNUITY_CREDIT_RATIO  INCOME_PER_FAM  EMPLOYED_BIRTH_RATIO  EXT_SOURCES_MEAN
0           100002       1                   0            1             0                1  ...              0.121977            1.158394              0.060749        101250.0              0.067322          0.161787
1           100003       0                   0            0             0                0  ...              0.132216            1.145198              0.027598         90000.0              0.070858          0.477838
2           100004       0                   1            1             1                1  ...              0.099999            0.999993              0.050000         33750.0              0.011813          0.595160
3           100006       0                   0            0             0                1  ...              0.219898            1.052799              0.094941         45000.0              0.159897          0.550147
4           100007       0                   0            1             0                1  ...              0.179961            0.999998              0.042623         60750.0              0.152411          0.440913
...            ...     ...                 ...          ...           ...              ...  ...                   ...                 ...                   ...             ...                   ...               ...
307506      456251       0                   0            1             0                0  ...              0.174970            1.131995              0.108197         78750.0              0.025300          0.442401
307507      456252       0                   0            0             0                1  ...              0.166685            1.197995              0.044524         36000.0            -17.580044          0.371997
307508      456253       0                   0            0             0                1  ...              0.195940            1.158398              0.044239         76500.0              0.529231          0.499536
307509      456254       1                   0            0             0                1  ...              0.118157            1.158391              0.054592         57000.0              0.400100          0.558395
307510      456255       0                   0            0             0                0  ...              0.311855            0.999999              0.072767         52500.0              0.074865          0.518984

[307511 rows x 526 columns], 'new_col': 'EXT_SOURCES_PROD', 'expression': "df['EXT_SOURCE_1'].fillna(0.5) * df['EXT_SOURCE_2'].fillna(0.5) * df['EXT_SOURCE_3'].fillna(0.5)"}

---

## [2026-05-30 22:31:32] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  CREDIT_GOODS_RATIO  ANNUITY_CREDIT_RATIO  INCOME_PER_FAM  EMPLOYED_BIRTH_RATIO  EXT_SOURCES_MEAN  EXT_S

---

## [2026-05-30 22:31:32] FeatureEngineer
**Action:** Action: create_interaction

Top two external scores combined

---

## [2026-05-30 22:31:38] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  CREDIT_GOODS_RATIO  ANNUITY_CREDIT_RATIO  INCOME_PER_FAM  EMPLOYED_BIRTH_RATIO  EXT_SOURCES_MEAN  EXT_SOURCES_PROD
0           100002       1                   0            1             0                1  ...            1.158394              0.060749        101250.0              0.067322          0.161787          0.003043
1           100003       0                   0            0             0                0  ...            1.145198              0.027598         90000.0              0.070858          0.477838          0.096842
2           100004       0                   1            1             1                1  ...            0.999993              0.050000         33750.0              0.011813          0.595160          0.202787
3           100006       0                   0            0             0                1  ...            1.052799              0.094941         45000.0              0.159897          0.550147          0.162610
4           100007       0                   0            1             0                1  ...            0.999998              0.042623         60750.0              0.152411          0.440913          0.080685
...            ...     ...                 ...          ...           ...              ...  ...                 ...                   ...             ...                   ...               ...               ...
307506      456251       0                   0            1             0                0  ...            1.131995              0.108197         78750.0              0.025300          0.442401          0.049613
307507      456252       0                   0            0             0                1  ...            1.197995              0.044524         36000.0            -17.580044          0.371997          0.028998
307508      456253       0                   0            0             0                1  ...            1.158398              0.044239         76500.0              0.529231          0.499536          0.087235
307509      456254       1                   0            0             0                1  ...            1.158391              0.054592         57000.0              0.400100          0.558395          0.169937
307510      456255       0                   0            0             0                0  ...            0.999999              0.072767         52500.0              0.074865          0.518984          0.059287

[307511 rows x 527 columns], 'new_col': 'EXT2_EXT3', 'expression': "df['EXT_SOURCE_2'].fillna(0.5) * df['EXT_SOURCE_3'].fillna(0.5)"}

---

## [2026-05-30 22:31:43] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  ANNUITY_CREDIT_RATIO  INCOME_PER_FAM  EMPLOYED_BIRTH_RATIO  EXT_SOURCES_MEAN  EXT_SOURCES_PROD  EXT2_EX

---

## [2026-05-30 22:31:43] FeatureEngineer
**Action:** Action: create_interaction

Bureau debt utilization

---

## [2026-05-30 22:31:46] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  ANNUITY_CREDIT_RATIO  INCOME_PER_FAM  EMPLOYED_BIRTH_RATIO  EXT_SOURCES_MEAN  EXT_SOURCES_PROD  EXT2_EXT3
0           100002       1                   0            1             0                1  ...              0.060749        101250.0              0.067322          0.161787          0.003043   0.036649
1           100003       0                   0            0             0                0  ...              0.027598         90000.0              0.070858          0.477838          0.096842   0.311123
2           100004       0                   1            1             1                1  ...              0.050000         33750.0              0.011813          0.595160          0.202787   0.405575
3           100006       0                   0            0             0                1  ...              0.094941         45000.0              0.159897          0.550147          0.162610   0.325221
4           100007       0                   0            1             0                1  ...              0.042623         60750.0              0.152411          0.440913          0.080685   0.161369
...            ...     ...                 ...          ...           ...              ...  ...                   ...             ...                   ...               ...               ...        ...
307506      456251       0                   0            1             0                0  ...              0.108197         78750.0              0.025300          0.442401          0.049613   0.340816
307507      456252       0                   0            0             0                1  ...              0.044524         36000.0            -17.580044          0.371997          0.028998   0.057996
307508      456253       0                   0            0             0                1  ...              0.044239         76500.0              0.529231          0.499536          0.087235   0.117248
307509      456254       1                   0            0             0                1  ...              0.054592         57000.0              0.400100          0.558395          0.169937   0.339874
307510      456255       0                   0            0             0                0  ...              0.072767         52500.0              0.074865          0.518984          0.059287   0.080722

[307511 rows x 528 columns], 'new_col': 'BUREAU_DEBT_CREDIT_RATIO', 'expression': "df['bur_AMT_CREDIT_SUM_DEBT_sum'] / (df['bur_AMT_CREDIT_SUM_sum'] + 1)"}

---

## [2026-05-30 22:31:50] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  INCOME_PER_FAM  EMPLOYED_BIRTH_RATIO  EXT_SOURCES_MEAN  EXT_SOURCES_PROD  EXT2_EXT3  BUREAU_DEBT_CREDIT

---

## [2026-05-30 22:31:50] FeatureEngineer
**Action:** Action: create_interaction

Cross-group: bureau debt vs income

---

## [2026-05-30 22:31:54] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  INCOME_PER_FAM  EMPLOYED_BIRTH_RATIO  EXT_SOURCES_MEAN  EXT_SOURCES_PROD  EXT2_EXT3  BUREAU_DEBT_CREDIT_RATIO
0           100002       1                   0            1             0                1  ...        101250.0              0.067322          0.161787          0.003043   0.036649                  0.284121
1           100003       0                   0            0             0                0  ...         90000.0              0.070858          0.477838          0.096842   0.311123                  0.000000
2           100004       0                   1            1             1                1  ...         33750.0              0.011813          0.595160          0.202787   0.405575                  0.000000
3           100006       0                   0            0             0                1  ...         45000.0              0.159897          0.550147          0.162610   0.325221                  0.207140
4           100007       0                   0            1             0                1  ...         60750.0              0.152411          0.440913          0.080685   0.161369                  0.000000
...            ...     ...                 ...          ...           ...              ...  ...             ...                   ...               ...               ...        ...                       ...
307506      456251       0                   0            1             0                0  ...         78750.0              0.025300          0.442401          0.049613   0.340816                  0.207140
307507      456252       0                   0            0             0                1  ...         36000.0            -17.580044          0.371997          0.028998   0.057996                  0.207140
307508      456253       0                   0            0             0                1  ...         76500.0              0.529231          0.499536          0.087235   0.117248                  0.453493
307509      456254       1                   0            0             0                1  ...         57000.0              0.400100          0.558395          0.169937   0.339874                  0.000000
307510      456255       0                   0            0             0                0  ...         52500.0              0.074865          0.518984          0.059287   0.080722                  0.403720

[307511 rows x 529 columns], 'new_col': 'BUREAU_DEBT_INCOME_RATIO', 'expression': "df['bur_AMT_CREDIT_SUM_DEBT_sum'] / (df['AMT_INCOME_TOTAL'] + 1)"}

---

## [2026-05-30 22:31:59] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  EMPLOYED_BIRTH_RATIO  EXT_SOURCES_MEAN  EXT_SOURCES_PROD  EXT2_EXT3  BUREAU_DEBT_CREDIT_RATIO  BUREAU_D

---

## [2026-05-30 22:31:59] FeatureEngineer
**Action:** Action: create_interaction

Bureau delinquency rate

---

## [2026-05-30 22:32:04] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  EMPLOYED_BIRTH_RATIO  EXT_SOURCES_MEAN  EXT_SOURCES_PROD  EXT2_EXT3  BUREAU_DEBT_CREDIT_RATIO  BUREAU_DEBT_INCOME_RATIO
0           100002       1                   0            1             0                1  ...              0.067322          0.161787          0.003043   0.036649                  0.284121                  1.213727
1           100003       0                   0            0             0                0  ...              0.070858          0.477838          0.096842   0.311123                  0.000000                  0.000000
2           100004       0                   1            1             1                1  ...              0.011813          0.595160          0.202787   0.405575                  0.000000                  0.000000
3           100006       0                   0            0             0                1  ...              0.159897          0.550147          0.162610   0.325221                  0.207140                  1.175429
4           100007       0                   0            1             0                1  ...              0.152411          0.440913          0.080685   0.161369                  0.000000                  0.000000
...            ...     ...                 ...          ...           ...              ...  ...                   ...               ...               ...        ...                       ...                       ...
307506      456251       0                   0            1             0                0  ...              0.025300          0.442401          0.049613   0.340816                  0.207140                  1.175429
307507      456252       0                   0            0             0                1  ...            -17.580044          0.371997          0.028998   0.057996                  0.207140                  1.175429
307508      456253       0                   0            0             0                1  ...              0.529231          0.499536          0.087235   0.117248                  0.453493                 11.737394
307509      456254       1                   0            0             0                1  ...              0.400100          0.558395          0.169937   0.339874                  0.000000                  0.000000
307510      456255       0                   0            0             0                0  ...              0.074865          0.518984          0.059287   0.080722                  0.403720                  9.745418

[307511 rows x 530 columns], 'new_col': 'BUREAU_OVERDUE_DEBT_RATIO', 'expression': "df['bur_AMT_CREDIT_SUM_OVERDUE_sum'] / (df['bur_AMT_CREDIT_SUM_DEBT_sum'].abs() + 1)"}

---

## [2026-05-30 22:32:08] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  EXT_SOURCES_MEAN  EXT_SOURCES_PROD  EXT2_EXT3  BUREAU_DEBT_CREDIT_RATIO  BUREAU_DEBT_INCOME_RATIO  BURE

---

## [2026-05-30 22:32:08] FeatureEngineer
**Action:** Action: create_interaction

Social circle default rate

---

## [2026-05-30 22:32:12] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  EXT_SOURCES_MEAN  EXT_SOURCES_PROD  EXT2_EXT3  BUREAU_DEBT_CREDIT_RATIO  BUREAU_DEBT_INCOME_RATIO  BUREAU_OVERDUE_DEBT_RATIO
0           100002       1                   0            1             0                1  ...          0.161787          0.003043   0.036649                  0.284121                  1.213727                        0.0
1           100003       0                   0            0             0                0  ...          0.477838          0.096842   0.311123                  0.000000                  0.000000                        0.0
2           100004       0                   1            1             1                1  ...          0.595160          0.202787   0.405575                  0.000000                  0.000000                        0.0
3           100006       0                   0            0             0                1  ...          0.550147          0.162610   0.325221                  0.207140                  1.175429                        0.0
4           100007       0                   0            1             0                1  ...          0.440913          0.080685   0.161369                  0.000000                  0.000000                        0.0
...            ...     ...                 ...          ...           ...              ...  ...               ...               ...        ...                       ...                       ...                        ...
307506      456251       0                   0            1             0                0  ...          0.442401          0.049613   0.340816                  0.207140                  1.175429                        0.0
307507      456252       0                   0            0             0                1  ...          0.371997          0.028998   0.057996                  0.207140                  1.175429                        0.0
307508      456253       0                   0            0             0                1  ...          0.499536          0.087235   0.117248                  0.453493                 11.737394                        0.0
307509      456254       1                   0            0             0                1  ...          0.558395          0.169937   0.339874                  0.000000                  0.000000                        0.0
307510      456255       0                   0            0             0                0  ...          0.518984          0.059287   0.080722                  0.403720                  9.745418                        0.0

[307511 rows x 531 columns], 'new_col': 'SOCIAL_DEF_RATIO', 'expression': "df['DEF_30_CNT_SOCIAL_CIRCLE'] / (df['OBS_30_CNT_SOCIAL_CIRCLE'] + 1)"}

---

## [2026-05-30 22:32:17] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  EXT_SOURCES_PROD  EXT2_EXT3  BUREAU_DEBT_CREDIT_RATIO  BUREAU_DEBT_INCOME_RATIO  BUREAU_OVERDUE_DEBT_RA

---

## [2026-05-30 22:32:17] FeatureEngineer
**Action:** Action: create_interaction

Clean employment ratio

---

## [2026-05-30 22:32:23] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  EXT_SOURCES_PROD  EXT2_EXT3  BUREAU_DEBT_CREDIT_RATIO  BUREAU_DEBT_INCOME_RATIO  BUREAU_OVERDUE_DEBT_RATIO  SOCIAL_DEF_RATIO
0           100002       1                   0            1             0                1  ...          0.003043   0.036649                  0.284121                  1.213727                        0.0          0.666667
1           100003       0                   0            0             0                0  ...          0.096842   0.311123                  0.000000                  0.000000                        0.0          0.000000
2           100004       0                   1            1             1                1  ...          0.202787   0.405575                  0.000000                  0.000000                        0.0          0.000000
3           100006       0                   0            0             0                1  ...          0.162610   0.325221                  0.207140                  1.175429                        0.0          0.000000
4           100007       0                   0            1             0                1  ...          0.080685   0.161369                  0.000000                  0.000000                        0.0          0.000000
...            ...     ...                 ...          ...           ...              ...  ...               ...        ...                       ...                       ...                        ...               ...
307506      456251       0                   0            1             0                0  ...          0.049613   0.340816                  0.207140                  1.175429                        0.0          0.000000
307507      456252       0                   0            0             0                1  ...          0.028998   0.057996                  0.207140                  1.175429                        0.0          0.000000
307508      456253       0                   0            0             0                1  ...          0.087235   0.117248                  0.453493                 11.737394                        0.0          0.000000
307509      456254       1                   0            0             0                1  ...          0.169937   0.339874                  0.000000                  0.000000                        0.0          0.000000
307510      456255       0                   0            0             0                0  ...          0.059287   0.080722                  0.403720                  9.745418                        0.0          0.000000

[307511 rows x 532 columns], 'new_col': 'DAYS_EMPLOYED_PCT', 'expression': "df['DAYS_EMPLOYED'].replace(365243, 0) / (df['DAYS_BIRTH'] - 1)"}

---

## [2026-05-30 22:32:29] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  ...  BUREAU_DEBT_CREDIT_RATIO  BUREAU_DEBT_INCOME_RATIO  BUREAU_OVERDUE_DEBT_RATIO  SOCIAL_DEF_RATIO  DAYS_EMPLOYED_PCT
0    

---

## [2026-05-30 22:32:29] FeatureEngineer
**Action:** Action: create_interaction

Income-to-credit ratio

---

## [2026-05-30 22:32:34] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  ...  BUREAU_DEBT_CREDIT_RATIO  BUREAU_DEBT_INCOME_RATIO  BUREAU_OVERDUE_DEBT_RATIO  SOCIAL_DEF_RATIO  DAYS_EMPLOYED_PCT
0           100002       1                   0            1             0  ...                  0.284121                  1.213727                        0.0          0.666667           0.067322
1           100003       0                   0            0             0  ...                  0.000000                  0.000000                        0.0          0.000000           0.070858
2           100004       0                   1            1             1  ...                  0.000000                  0.000000                        0.0          0.000000           0.011813
3           100006       0                   0            0             0  ...                  0.207140                  1.175429                        0.0          0.000000           0.159897
4           100007       0                   0            1             0  ...                  0.000000                  0.000000                        0.0          0.000000           0.152411
...            ...     ...                 ...          ...           ...  ...                       ...                       ...                        ...               ...                ...
307506      456251       0                   0            1             0  ...                  0.207140                  1.175429                        0.0          0.000000           0.025300
307507      456252       0                   0            0             0  ...                  0.207140                  1.175429                        0.0          0.000000          -0.000000
307508      456253       0                   0            0             0  ...                  0.453493                 11.737394                        0.0          0.000000           0.529231
307509      456254       1                   0            0             0  ...                  0.000000                  0.000000                        0.0          0.000000           0.400100
307510      456255       0                   0            0             0  ...                  0.403720                  9.745418                        0.0          0.000000           0.074865

[307511 rows x 533 columns], 'new_col': 'INCOME_CREDIT_PCT', 'expression': "df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + 1)"}

---

## [2026-05-30 22:32:38] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  ...  BUREAU_DEBT_INCOME_RATIO  BUREAU_OVERDUE_DEBT_RATIO  SOCIAL_DEF_RATIO  DAYS_EMPLOYED_PCT  INCOME_CREDIT_PCT
0           

---

## [2026-05-30 22:32:38] FeatureEngineer
**Action:** Action: create_interaction

Car age relative to client age

---

## [2026-05-30 22:32:42] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  ...  BUREAU_DEBT_INCOME_RATIO  BUREAU_OVERDUE_DEBT_RATIO  SOCIAL_DEF_RATIO  DAYS_EMPLOYED_PCT  INCOME_CREDIT_PCT
0           100002       1                   0            1             0  ...                  1.213727                        0.0          0.666667           0.067322           0.498034
1           100003       0                   0            0             0  ...                  0.000000                        0.0          0.000000           0.070858           0.208735
2           100004       0                   1            1             1  ...                  0.000000                        0.0          0.000000           0.011813           0.499996
3           100006       0                   0            0             0  ...                  1.175429                        0.0          0.000000           0.159897           0.431746
4           100007       0                   0            1             0  ...                  0.000000                        0.0          0.000000           0.152411           0.236842
...            ...     ...                 ...          ...           ...  ...                       ...                        ...               ...                ...                ...
307506      456251       0                   0            1             0  ...                  1.175429                        0.0          0.000000           0.025300           0.618372
307507      456252       0                   0            0             0  ...                  1.175429                        0.0          0.000000          -0.000000           0.267111
307508      456253       0                   0            0             0  ...                 11.737394                        0.0          0.000000           0.529231           0.225775
307509      456254       1                   0            0             0  ...                  0.000000                        0.0          0.000000           0.400100           0.462027
307510      456255       0                   0            0             0  ...                  9.745418                        0.0          0.000000           0.074865           0.233333

[307511 rows x 534 columns], 'new_col': 'CAR_TO_BIRTH_RATIO', 'expression': "df['OWN_CAR_AGE'] / (-df['DAYS_BIRTH']/365 + 1)"}

---

## [2026-05-30 22:32:47] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  ...  BUREAU_OVERDUE_DEBT_RATIO  SOCIAL_DEF_RATIO  DAYS_EMPLOYED_PCT  INCOME_CREDIT_PCT  CAR_TO_BIRTH_RATIO
0           100002

---

## [2026-05-30 22:32:47] FeatureEngineer
**Action:** Action: create_interaction

Phone change recency vs age

---

## [2026-05-30 22:32:50] FeatureEngineer
**Action:** Tool: create_interaction

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  ...  BUREAU_OVERDUE_DEBT_RATIO  SOCIAL_DEF_RATIO  DAYS_EMPLOYED_PCT  INCOME_CREDIT_PCT  CAR_TO_BIRTH_RATIO
0           100002       1                   0            1             0  ...                        0.0          0.666667           0.067322           0.498034            0.223366
1           100003       0                   0            0             0  ...                        0.0          0.000000           0.070858           0.208735            0.223366
2           100004       0                   1            1             1  ...                        0.0          0.000000           0.011813           0.499996            0.488898
3           100006       0                   0            0             0  ...                        0.0          0.000000           0.159897           0.431746            0.223366
4           100007       0                   0            1             0  ...                        0.0          0.000000           0.152411           0.236842            0.223366
...            ...     ...                 ...          ...           ...  ...                        ...               ...                ...                ...                 ...
307506      456251       0                   0            1             0  ...                        0.0          0.000000           0.025300           0.618372            0.223366
307507      456252       0                   0            0             0  ...                        0.0          0.000000          -0.000000           0.267111            0.223366
307508      456253       0                   0            0             0  ...                        0.0          0.000000           0.529231           0.225775            0.223366
307509      456254       1                   0            0             0  ...                        0.0          0.000000           0.400100           0.462027            0.223366
307510      456255       0                   0            0             0  ...                        0.0          0.000000           0.074865           0.233333            0.223366

[307511 rows x 535 columns], 'new_col': 'PHONE_TO_BIRTH_RATIO', 'expression': "df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH'] - 1)"}

---

## [2026-05-30 22:32:55] FeatureEngineer
**Action:** Tool Result: create_interaction

Success:         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  BUREAU_OVERDUE_DEBT_RATIO  SOCIAL_DEF_RATIO  DAYS_EMPLOYED_PCT  INCOME_CREDIT_PCT  CAR_TO_BIRTH_RATIO  

---

## [2026-05-30 22:32:55] FeatureEngineer
**Action:** Action: correlation_analysis

Validate features against TARGET

---

## [2026-05-30 22:33:00] FeatureEngineer
**Action:** Tool: correlation_analysis

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  BUREAU_OVERDUE_DEBT_RATIO  SOCIAL_DEF_RATIO  DAYS_EMPLOYED_PCT  INCOME_CREDIT_PCT  CAR_TO_BIRTH_RATIO  PHONE_TO_BIRTH_RATIO
0           100002       1                   0            1             0                1  ...                        0.0          0.666667           0.067322           0.498034            0.223366              0.119848
1           100003       0                   0            0             0                0  ...                        0.0          0.000000           0.070858           0.208735            0.223366              0.049386
2           100004       0                   1            1             1                1  ...                        0.0          0.000000           0.011813           0.499996            0.488898              0.042789
3           100006       0                   0            0             0                1  ...                        0.0          0.000000           0.159897           0.431746            0.223366              0.032463
4           100007       0                   0            1             0                1  ...                        0.0          0.000000           0.152411           0.236842            0.223366              0.055486
...            ...     ...                 ...          ...           ...              ...  ...                        ...               ...                ...                ...                 ...                   ...
307506      456251       0                   0            1             0                0  ...                        0.0          0.000000           0.025300           0.618372            0.223366              0.029267
307507      456252       0                   0            0             0                1  ...                        0.0          0.000000          -0.000000           0.267111            0.223366             -0.000000
307508      456253       0                   0            0             0                1  ...                        0.0          0.000000           0.529231           0.225775            0.223366              0.127547
307509      456254       1                   0            0             0                1  ...                        0.0          0.000000           0.400100           0.462027            0.223366              0.026919
307510      456255       0                   0            0             0                0  ...                        0.0          0.000000           0.074865           0.233333            0.223366              0.046687

[307511 rows x 536 columns], 'target': 'TARGET'}

---

## [2026-05-30 22:33:33] FeatureEngineer
**Action:** Tool Result: correlation_analysis

Success: {
  "correlations": {
    "EXT_SOURCES_MEAN": -0.2217,
    "EXT2_EXT3": -0.1943,
    "EXT_SOURCES_PROD": -0.1895,
    "EXT_SOURCE_3": -0.1789,
    "EXT_SOURCE_2": -0.1605,
    "EXT_SOURCE_1": -0.1553,

---

## [2026-05-30 22:33:33] FeatureEngineer
**Action:** Correlation Results

{
  "correlations": {
    "EXT_SOURCES_MEAN": -0.2217,
    "EXT2_EXT3": -0.1943,
    "EXT_SOURCES_PROD": -0.1895,
    "EXT_SOURCE_3": -0.1789,
    "EXT_SOURCE_2": -0.1605,
    "EXT_SOURCE_1": -0.1553,
    "cc_CNT_DRAWINGS_CURRENT_std": 0.1089,
    "cc_CNT_DRAWINGS_CURRENT_max": 0.1014,
    "bur_DAYS_CREDIT_mean": 0.0897,
    "bur_bb_MONTHS_BALANCE_min_mean": 0.089,
    "cc_AMT_BALANCE_mean": 0.0872,
    "cc_AMT_RECIVABLE_mean": 0.0865,
    "cc_AMT_TOTAL_RECEIVABLE_mean": 0.0865,
    "cc_AMT_RECE

---

## [2026-05-30 22:33:33] FeatureEngineer
**Action:** Action: select_top_features

Keep top 350 features (70% cap) to reduce noise

---

## [2026-05-30 22:33:37] FeatureEngineer
**Action:** Tool: select_top_features

Parameters: {'df':         SK_ID_CURR  TARGET  NAME_CONTRACT_TYPE  CODE_GENDER  FLAG_OWN_CAR  FLAG_OWN_REALTY  ...  BUREAU_OVERDUE_DEBT_RATIO  SOCIAL_DEF_RATIO  DAYS_EMPLOYED_PCT  INCOME_CREDIT_PCT  CAR_TO_BIRTH_RATIO  PHONE_TO_BIRTH_RATIO
0           100002       1                   0            1             0                1  ...                        0.0          0.666667           0.067322           0.498034            0.223366              0.119848
1           100003       0                   0            0             0                0  ...                        0.0          0.000000           0.070858           0.208735            0.223366              0.049386
2           100004       0                   1            1             1                1  ...                        0.0          0.000000           0.011813           0.499996            0.488898              0.042789
3           100006       0                   0            0             0                1  ...                        0.0          0.000000           0.159897           0.431746            0.223366              0.032463
4           100007       0                   0            1             0                1  ...                        0.0          0.000000           0.152411           0.236842            0.223366              0.055486
...            ...     ...                 ...          ...           ...              ...  ...                        ...               ...                ...                ...                 ...                   ...
307506      456251       0                   0            1             0                0  ...                        0.0          0.000000           0.025300           0.618372            0.223366              0.029267
307507      456252       0                   0            0             0                1  ...                        0.0          0.000000          -0.000000           0.267111            0.223366             -0.000000
307508      456253       0                   0            0             0                1  ...                        0.0          0.000000           0.529231           0.225775            0.223366              0.127547
307509      456254       1                   0            0             0                1  ...                        0.0          0.000000           0.400100           0.462027            0.223366              0.026919
307510      456255       0                   0            0             0                0  ...                        0.0          0.000000           0.074865           0.233333            0.223366              0.046687

[307511 rows x 536 columns], 'target': 'TARGET', 'k': 350}

---

## [2026-05-30 22:34:37] FeatureEngineer
**Action:** Tool Result: select_top_features

Success:         EXT_SOURCES_MEAN  EXT2_EXT3  EXT_SOURCES_PROD  EXT_SOURCE_2  EXT_SOURCE_3  ...  cc_CNT_INSTALMENT_MATURE_CUM_mean  prev_AMT_CREDIT_sum  prev_AMT_GOODS_PRICE_count  pos_SK_DPD_DEF_max  TARGET
0

---

## [2026-05-30 22:34:38] FeatureEngineer
**Action:** Protected Col Restored

Re-added 'SK_ID_CURR' (composite key / entity ID) after feature selection

---

## [2026-05-30 22:42:49] FeatureEngineer
**Action:** Data Saved

Engineered data saved to outputs/engineered_data.csv

---

## [2026-05-30 22:42:49] FeatureEngineer
**Action:** Report Saved

Saved to outputs/feature_engineer_report.json

---

## [2026-05-30 22:42:49] FeatureEngineer
**Action:** Process Complete

Shape: (307511, 519) -> (307511, 352)

---

## [2026-05-30 22:42:50] HANDOFF
**Action:** Data Transfer from FeatureEngineer

Data: outputs/engineered_data.csv, Report keys: ['agent', 'original_shape', 'final_shape', 'actions_taken', 'summary', 'final_features', 'entity_id_col', 'composite_key_cols', 'target_column']

---

## [2026-05-30 22:42:50] PIPELINE
**Action:** Stage 3

Initializing Train Model Agent

---

## [2026-05-30 22:44:08] TrainModel
**Action:** Process Start

Shape=(307511, 352)

---

## [2026-05-30 22:44:08] TrainModel
**Action:** Previous Agent Summary

Performed 20 feature engineering actions: Encoded all categorical columns with label: Encode 16 categorical columns before interactions and selection; Created feature 'CREDIT_INCOME_RATIO': Debt burden: credit amount vs income; Created feature 'ANNUITY_INCOME_RATIO': Installment burden vs income; and 17 more actions.

---

## [2026-05-30 22:44:11] TrainModel
**Action:** Config

target=TARGET | date_col=DAYS_BIRTH | id_col=SK_ID_CURR | n_features=349 | gpu=NO (CPU only)

---

## [2026-05-30 22:44:12] TrainModel
**Action:** WARN

Only 1 distinct month(s) in 'DAYS_BIRTH' — cannot create OOT split, falling back to simple split

---

## [2026-05-30 22:44:21] TrainModel
**Action:** Split (simple)

train=184506 | valid=61502 | test=61503 | no OOT

---

## [2026-05-30 22:44:29] TrainModel
**Action:** Time Budget

rows=307511 | cols=349 | flaml=446s | optuna=446s

---

## [2026-05-30 22:44:29] TrainModel
**Action:** FLAML Start

time_budget=446s | estimators=xgboost,lgbm,catboost,rf,extra_tree

---

## [2026-05-30 22:52:16] TrainModel
**Action:** FLAML

best=xgboost | CV_loss=0.2329 | valid_auc=0.7671

---

## [2026-05-30 22:52:16] TrainModel
**Action:** Optuna Start

n_trials=50 | timeout=446s

---

## [2026-05-30 23:05:09] TrainModel
**Action:** Optuna

best_AUC=0.7801 | params={"n_estimators": 750, "learning_rate": 0.010725209743171996, "max_depth": 10, "min_child_weight": 25, "subsample": 0.5274034664069657, "colsample_bytree": 0.5090949803242604, "gamma": 3.93940226136269

---

## [2026-05-30 23:05:09] TrainModel
**Action:** RFE Start

target=100 | rfecv=False

---

## [2026-05-30 23:50:53] TrainModel
**Action:** RFE

349 → 100 features

---

## [2026-05-30 23:50:53] TrainModel
**Action:** PSI

No OOT data → skip

---

## [2026-05-30 23:50:57] TrainModel
**Action:** Stability

Only 1 months < min=6 → skip

---

## [2026-05-30 23:51:52] TrainModel
**Action:** SHAP+PSI Prune

start=100 | baseline_valid_auc=0.7612 | min_features=10 | psi_available=False

---

## [2026-05-30 23:55:38] TrainModel
**Action:** SHAP+PSI

step=1 remove 'cc_AMT_RECIVABLE_max' → auc dropped 0.7612→0.7609 [no_improve=1/2]

---

## [2026-05-30 23:58:33] TrainModel
**Action:** SHAP+PSI

step=2 removed 'cc_AMT_DRAWINGS_CURRENT_max' | n=98 | auc=0.7616

---

## [2026-05-31 00:00:07] TrainModel
**Action:** SHAP+PSI

step=3 remove 'cc_AMT_PAYMENT_TOTAL_CURRENT_std' → auc dropped 0.7616→0.7609 [no_improve=1/2]

---

## [2026-05-31 00:01:36] TrainModel
**Action:** SHAP+PSI

step=4 removed 'cc_AMT_RECIVABLE_std' | n=96 | auc=0.7619

---

## [2026-05-31 00:02:54] TrainModel
**Action:** SHAP+PSI

step=5 remove 'prev_NFLAG_INSURED_ON_APPROVAL_min' → auc dropped 0.7619→0.7609 [no_improve=1/2]

---

## [2026-05-31 00:04:11] TrainModel
**Action:** SHAP+PSI

step=6 remove 'cc_AMT_DRAWINGS_ATM_CURRENT_count' → auc dropped 0.7619→0.7615 [no_improve=2/2]

---

## [2026-05-31 00:04:11] TrainModel
**Action:** SHAP+PSI Done

100 → 96 features | best_valid_auc=0.7619 | steps=6

---

## [2026-05-31 00:04:11] TrainModel
**Action:** Feature Pipeline

init=349 → RFE=100 → PSI=100 → Stability=100 → SHAP+PSI=96

---

## [2026-05-31 00:20:15] TrainModel
**Action:** CV AUC

0.7792 ± 0.0045

---

## [2026-05-31 00:20:19] TrainModel
**Action:** valid AUC

0.7785

---

## [2026-05-31 00:20:22] TrainModel
**Action:** test AUC

0.7816

---

## [2026-05-31 00:20:22] TrainModel
**Action:** Model Saved

outputs/final_model.pkl

---

## [2026-05-31 00:20:22] TrainModel
**Action:** Model Code Saved

outputs/final_model_code.py

---

## [2026-05-31 00:20:22] TrainModel
**Action:** LLM Call

model=local-model | prompt_len=756 chars | temp=0.2 | top_p=0.95 | max_tokens=2000 | json_mode=False

---

## [2026-05-31 00:20:35] TrainModel
**Action:** ERROR

Cannot reach LiteLLM proxy at http://localhost:4000: Connection error.

---

## [2026-05-31 00:20:35] TrainModel
**Action:** LLM Fallback

Trying Claude (claude-opus-4-7) as last resort...

---

## [2026-05-31 00:20:49] TrainModel
**Action:** LLM Tokens

model=claude-opus-4-7 | input=415 | output=341 | total=756

---

## [2026-05-31 00:20:49] TrainModel
**Action:** LLM Response

Claude (claude-opus-4-7) | 963 chars | fallback=claude

---

## [2026-05-31 00:20:49] TrainModel
**Action:** Report Saved

Saved to outputs/model_trainer_report.json

---

## [2026-05-31 00:20:49] TrainModel
**Action:** Process Complete

best=xgboost | features=96 | oot_auc=N/A | valid_auc=0.7785

---


# Final Summary

## Agent 1: Data Cleaner
- Actions: Performed 38 cleaning actions: Dropped column 'cc_SK_DPD_min': Constant column — single unique value; Dropped column 'cc_SK_DPD_DEF_min': Constant column — single unique value; Dropped column 'prev_RATE_INTEREST_PRIMARY_mean': 98.5% null; and 35 more actions.

## Agent 2: Feature Engineer
- Strategy: Performed 20 feature engineering actions: Encoded all categorical columns with label: Encode 16 categorical columns before interactions and selection; Created feature 'CREDIT_INCOME_RATIO': Debt burden: credit amount vs income; Created feature 'ANNUITY_INCOME_RATIO': Installment burden vs income; and 17 more actions.

## Agent 3: Model Trainer
- Final Metrics: {'best_model': 'xgboost', 'best_params': {'n_estimators': 750, 'learning_rate': 0.010725209743171996, 'max_depth': 10, 'min_child_weight': 25, 'subsample': 0.5274034664069657, 'colsample_bytree': 0.5090949803242604, 'gamma': 3.939402261362697e-07, 'reg_alpha': 5.472429642032198e-06, 'reg_lambda': 0.00052821153945323, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1, 'verbosity': 0}, 'cv_auc_mean': 0.7792, 'cv_auc_std': 0.0045, 'valid_auc': 0.7785, 'test_auc': 0.7816}


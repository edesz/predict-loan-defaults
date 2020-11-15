#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import os
import shlex
import subprocess
from datetime import datetime
from typing import Dict, List

import papermill as pm

PROJ_ROOT_DIR = os.path.abspath(os.getcwd())
data_dir = os.path.join(PROJ_ROOT_DIR, "data")
output_notebook_dir = os.path.join(PROJ_ROOT_DIR, "executed_notebooks")

raw_data_path = os.path.join(data_dir, "raw", "lending_club_loans.csv")
three_four_dict = dict(
    cloud_storage="yes",
    nan_threshold=0.5,
    non_useful_cols=["url", "desc"],
    datetime_cols=["issue_d", "last_pymnt_d"],
    cols_one_eighteen=[
        "id",
        "member_id",
        "funded_amnt",
        "funded_amnt_inv",
        "grade",
        "sub_grade",
        "emp_title",
    ],
    cols_eighteen_thirtysix=[
        "zip_code",
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
    ],
    cols_thirtyseven_end=[
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_amnt",
    ],
    loan_status=["Fully Paid", "Charged Off"],
    mapping_dictionary_labels={
        "loan_status": {"Fully Paid": 1, "Charged Off": 0}
    },
    four_or_less_value_columns=["pymnt_plan"],
    more_than_one_pct_missing_columns=["pub_rec_bankruptcies"],
    datetime_cols_v2=["last_credit_pull_d", "earliest_cr_line"],
    high_cardinality_cols=["addr_state"],
    mapping_dict_emp_length={
        "emp_length": {
            "10+ years": 10,
            "9 years": 9,
            "8 years": 8,
            "7 years": 7,
            "6 years": 6,
            "5 years": 5,
            "4 years": 4,
            "3 years": 3,
            "2 years": 2,
            "1 year": 1,
            "< 1 year": 0,
            "n/a": 0,
        }
    },
    nominal_columns=[
        "home_ownership",
        "verification_status",
        "purpose",
        "term",
    ],
    repeated_data_cols=["title"],
    pct_to_numeric_cols=["int_rate", "revol_util"],
)

one_dict_nb_name = "1_feature_reduction.ipynb"
two_dict_nb_name = "2_feature_processing.ipynb"
three_dict_nb_name = "3_exploratory_data_analysis.ipynb"
four_dict_nb_name = "4_experiments_in_classification.ipynb"

one_dict = dict(
    raw_data_path=raw_data_path,
    cloud_storage="yes",
)
two_dict = dict(
    raw_data_path=raw_data_path,
    cloud_storage="yes",
)
three_dict = copy.deepcopy(three_four_dict)
three_dict.update(
    dict(
        raw_data_path=raw_data_path,
    )
)
# Append the inputs required by 4_*.ipynb
three_four_dict.update(
    dict(
        raw_data_file_path=raw_data_path,
        correlated_features=[
            "total_acc",
            "installment",
            "fico_range_low",
            "fico_range_high",
        ],
        look_ahead_features=["last_fico_range_low", "last_fico_range_high"],
        raw_labels=["loan_status"],
        new_labels=["is_default"],
        cols_to_show=[
            "preprocessor_type",
            "resamplers",
            "clf",
            "threshold",
            "params",
            "mean_test_recall_binary",
            "mean_test_fpr",
            "mean_test_auc",
            "mean_train_recall_binary",
            "mean_train_fpr",
            "mean_train_auc",
            "mean_fit_time",
            "std_train_recall_binary",
            "std_test_recall_binary",
            "std_train_fpr",
            "std_test_fpr",
            "mean_score_time",
            "clf_params",
        ],
    )
)


def run_cmd(cmd: str) -> None:
    print(cmd)
    process = subprocess.Popen(
        shlex.split(cmd), shell=False, stdout=subprocess.PIPE
    )
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(str(output.strip(), "utf-8"))
    _ = process.poll()


def papermill_run_notebook(
    nb_dict: Dict, output_notebook_dir: str = "executed_notebooks"
) -> None:
    """Execute notebook with papermill"""
    for notebook, nb_params in nb_dict.items():
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_nb = os.path.basename(notebook).replace(
            ".ipynb", f"-{now}.ipynb"
        )
        print(
            f"\nInput notebook path: {notebook}",
            f"Output notebook path: {output_notebook_dir}/{output_nb} ",
            sep="\n",
        )
        for key, val in nb_params.items():
            print(key, val, sep=": ")
        pm.execute_notebook(
            input_path=notebook,
            output_path=f"{output_notebook_dir}/{output_nb}",
            parameters=nb_params,
        )


def run_notebooks(
    notebook_list: List, output_notebook_dir: str = "executed_notebooks"
) -> None:
    """Execute notebooks from CLI.
    Parameters
    ----------
    nb_dict : List
        list of notebooks to be executed
    Usage
    -----
    > import os
    > PROJ_ROOT_DIR = os.path.abspath(os.getcwd())
    > one_dict_nb_name = "a.ipynb
    > one_dict = {"a": 1}
    > run_notebook(
          notebook_list=[
              {os.path.join(PROJ_ROOT_DIR, one_dict_nb_name): one_dict}
          ]
      )
    """
    for nb in notebook_list:
        papermill_run_notebook(
            nb_dict=nb, output_notebook_dir=output_notebook_dir
        )


if __name__ == "__main__":
    PROJ_ROOT_DIR = os.getcwd()
    nb_dict_list = [
        one_dict,
        two_dict,
        three_dict,
        three_four_dict,
    ]
    nb_name_list = [
        one_dict_nb_name,
        two_dict_nb_name,
        three_dict_nb_name,
        four_dict_nb_name,
    ]
    notebook_list = [
        {os.path.join(PROJ_ROOT_DIR, nb_name): nb_dict}
        for nb_dict, nb_name in zip(nb_dict_list, nb_name_list)
    ]
    run_notebooks(
        notebook_list=notebook_list, output_notebook_dir=output_notebook_dir
    )

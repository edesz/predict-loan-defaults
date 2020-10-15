# [Loan Default Prediction](#loan-default-prediction)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edesz/predict-loan-defaults) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edesz/predict-loan-defaults/master/1_feature_reduction.ipynb) ![CI](https://github.com/edesz/predict-loan-defaults/workflows/CI/badge.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/mit)

## [Table of Contents](#table-of-contents)
1. [Project Idea](#project-idea)
   * [Project Overview](#project-overview)
   * [Motivation](#motivation)
2. [Data acquisition](#data-acquisition)
   * [Primary data source](#primary-data-source)
   * [Supplementary data sources](#supplementary-data-sources)
   * [Data file creation](#data-file-creation)
3. [Analysis](#analysis)
4. [Usage](#usage)
5. [Project Organization](#project-organization)

## [Project Idea](#project-idea)

### [Project Overview](#project-overview)

The objective of this project is to use data released by the [P2P lending](https://en.wikipedia.org/wiki/Peer-to-peer_lending) company Lending Club to determine whether a newly approved loan will be paid back in full on time or result in a default.

### [Motivation](#motivation)
[Up until the end of 2020](https://www.fool.com/the-ascent/personal-loans/articles/lendingclub-ending-its-p2p-lending-platform-now-what/), the Lending Club platform connected loan borrowers with lenders (investors). An investor would lend money and earn a monthly rate of interest as the loan was paid back and the requested (loan) amount if the loan was fully paid back. Interest payments started as soon as a loan application was approved.

Data on approved loans were available from the [Statistics section](https://www.lendingclub.com/info/statistics.action) of the Lending Club platform webpage. This also included data for loans that were not paid back on time, resulting in a [default](https://en.wikipedia.org/wiki/Default_(finance)).

The goal of this project is to use historical Lending Club data to predict whether a new loan would be paid back in full on time or result in a default. The end user for this project is a conservative investor who was looking to determine whether to fund new loan applications on the platform (while it was still active).

## [Data acquisition](#data-acquisition)
### [Primary data source](#primary-data-source)
Data was taken from the [Lending Club Data platform](https://www.lendingclub.com/auth/login?login_url=%2Fstatistics%2Fadditional-statistics%3F) for the years 2007-2011. Since data on rejected loans is not useful to a prospective lender, only data on approved loans was used.

### [Supplementary data sources](#supplementary-data-sources)
None.

### [Data file creation](#data-file-creation)
None. For each step in the analysis, the raw data is loaded and processed from scratch. Intermediate files are not created after one or more steps of processing.

## [Analysis](#anlysis)
Analysis is performed using machine learning approaches in Python. Technical details are included in the various notebook (`*.ipynb`) files.

### [Evaluation Metrics](#evaluation-metrics)
In determining the [evaluation metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Offline_metrics)) to be considered for this business use case, keep in mind that the objective is to fund loans that do not default and do not fund loans loans that do default. With this in mind, the end user here, a conservative investor, wants to minimize their risk when it comes to funding loans, and so wants to
- maximize true positives
  - this is a loan that
    - was correctly predicted to result in a default (assigned label `1`) and indeed did result in a default
      - investor did not fund such a loan and it was not paid off on time
        - investor did not fund risky loans
  - the true positive rate (`TPR`, or [recall](https://en.wikipedia.org/wiki/Precision_and_recall#Introduction)) is the percent of loans that default that the investor would not fund
- minimize false positives
  - this is a loan that
    - is incorrectly predicted to result in a default (`1`) but was paid off on time (`0`)
      - investor did not fund such a loan but it would have been paid off on time
      - investor missed an opportunity to earn returns
  - the false positive rate (`FPR`) is the percent of loans that do not default that the investor would not fund

A case can be made that maximizing the true positive rate, correctly predicting loans that default (and minimizing money lost funding risky loans), is preferred to minimizing the false positive rate (incorrectly predicting loans that do not default, and minimizing missed opportunities to earn a return on lending money to these borrowers) for a conservative investor. In reality, this tradeoff should be discussed with the investor. For the current use case, both metrics will be used as much as possible, but the best model found during the analysis phase will be chosen based on the `TPR` (or recall).

## [Usage](#usage)
1. Clone this repository
   ```bash
   $ git clone https://github.com/edesz/predict-loan-defaults.git
   ```
2. Create Python virtual environment, install packages and launch intreactive Python platform
   ```bash
   $ make build
   ```
3. Run `*.ipynb` notebooks in numerical order of the numerical filename prefix (`1_`, `2_`, etc.).
   - `1_feature_reduction.ipynb`
     - initial reduction of number of columns in data, due to [lookahead bias](https://corporatefinanceinstitute.com/resources/knowledge/finance/look-ahead-bias/), [high cardinality](https://en.wikipedia.org/wiki/Cardinality_(SQL_statements)) or lack of usefulness for further analysis
   - `2_feature_processing.ipynb`
     - further removal of columns from data, cleaning of leftover columns
   - `3_exploratory_data_analysis.ipynb`
     - visual exploration of data for filtering and transforming columns (in preparation for analysis), removal of correlated columns, further removal of columns that look ahead in time
   - `4_experiments_in_classification.ipynb`
     - cost-sensitive machine learning experiments on imbalanced data
     - hyperparameter optimization
     - discrimination threshold adjustments
     - evaluation of performance of chosen algorithm

## [Project Organization](#project-organization)

    ├── LICENSE
    ├── .gitignore                    <- files and folders to be ignored by version control system
    ├── .pre-commit-config.yaml       <- configuration file for pre-commit hooks
    ├── .github
    │   ├── workflows
    │       └── integrate.yml         <- configuration file for CI build on Github Actions
    │       └── codeql-analysis.yml   <- configuration file for security scanning on Github Actions
    ├── LICENSE
    ├── environment.yml               <- configuration file to create environment to run project on Binder
    ├── Makefile                      <- Makefile with commands like `make lint` or `make build`
    ├── README.md                     <- The top-level README for developers using this project.
    ├── data
    │   ├── raw                       <- raw data retrieved from open data portal (using optional notebook)
    |   └── processed                 <- merged and filtered data, sampled at daily frequency
    ├── *.ipynb                       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                    and a short `-` delimited description
    │
    ├── requirements.txt              <- packages required to execute all Jupyter notebooks interactively (not from CI)
    ├── setup.py                      <- makes project pip installable (pip install -e .) so `src` can be imported
    ├── src                           <- Source code for use in this project.
    │   ├── __init__.py               <- Makes src a Python module
    │   └── *.py                      <- Scripts to use in analysis for pre-processing, training, etc.
    ├── papermill_runner.py           <- Python functions that execute system shell commands.
    └── tox.ini                       <- tox file with settings for running tox; see tox.testrun.org

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

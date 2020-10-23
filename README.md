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
   * [Evaluation Considerations](#evaluation-considerations)
   * [Metrics Used](#metrics-used)
4. [Usage](#usage)
5. [Project Organization](#project-organization)

## [Project Idea](#project-idea)

### [Project Overview](#project-overview)

The objective of this project is to use data released by the [P2P lending](https://en.wikipedia.org/wiki/Peer-to-peer_lending) company Lending Club to determine whether a newly approved loan will be paid back in full on time or result in a default.

### [Motivation](#motivation)
[Up until the end of 2020](https://www.fool.com/the-ascent/personal-loans/articles/lendingclub-ending-its-p2p-lending-platform-now-what/), the Lending Club platform connected loan borrowers with lenders (investors). An investor would lend money and earn a monthly rate of interest as the loan was paid back and the requested (loan) amount if the loan was fully paid back. Interest payments started as soon as a loan application was approved.

Data on approved loans were available from the [Statistics section](https://www.lendingclub.com/info/statistics.action) of the Lending Club platform webpage. This also included data for loans that were not paid back on time, resulting in a [default](https://en.wikipedia.org/wiki/Default_(finance)).

The goal of this project is to use historical Lending Club data to predict whether a new loan would be paid back in full on time or result in a default. The end user for this project is a cautious (risk-averse) investor who was looking to determine whether to fund new loan applications on the platform (while it was still active).

## [Data acquisition](#data-acquisition)
### [Primary data source](#primary-data-source)
Data was taken from the [Lending Club Data platform](https://www.lendingclub.com/auth/login?login_url=%2Fstatistics%2Fadditional-statistics%3F) for the years 2007-2011. Since data on rejected loans is not useful to a prospective lender, only data on approved loans was used.

### [Supplementary data sources](#supplementary-data-sources)
None.

### [Data file creation](#data-file-creation)
None. For each step in the analysis, the raw data is loaded and processed from scratch. Intermediate files are not created after one or more steps of processing.

## [Analysis](#anlysis)
Analysis is performed using machine learning approaches in Python. Technical details are included in the various notebook (`*.ipynb`) files.

### [Evaluation Considerations](#evaluation-considerations)

The following are the types of errors to be considered

| Actual | Predicted | Type of Error |
|--------|-----------|---------------|
| 0      | 1         | FP            |
| 1      | 1         | TP            |
| 0      | 0         | TN            |
| 1      | 0         | FN            |

The objective is to focus on the two types of misclassifications (errors) made by the quantitative analysis performed here - false negatives and false positives.

False negatives (`FN`) are the biggest concern since they incorrectly predict that a loan won't defaut but it actually does result in a default. The loan is defaulted but the investor provides funding and with no possibility to earn a return. This costs the investor money since risky loans are funded that result in a loss of money. `FN` should be minimized.

False positives (`FP`) predict  that a loan will default but it does not default and is paid off on time. This is a prospective loss of returns to the investor who has not funded the loan which would actually have been paid off. The investor could have effectively invested in these loans since they would have delivered returns. Since this was not done, returns are lost but principal is not lost, so that's not so bad. `FP` should be minimized.

A risk-averse investor would prefer to miss funding opportunities for loans that are paid off on time (false positives) than providing funding to risky loans that result in a default and take away their returns and principal (false negatives).

True negatives are when the model correctly identifies a loan paid off on time. The investor has funded such loans and is happily earning returns. This should not be used in punishing errors in the analysis.

True positives are when the model correctly predicts a loan that has defaulted. Again, the investor is happy here since such loans were not funded. This too should not be used in punishing errors in the analysis.

### [Metrics Used](#metrics-used)
[Evaluation metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Offline_metrics) that make use the `FP` and `FN` will be chosen for quantitatively assessing numerical analysis
- `TPR = TP / FN + TP`
  - called [Sensitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity), or [recall](https://en.wikipedia.org/wiki/Precision_and_recall#Introduction)
  - this minimizes the `FN` as required
  - since `FN` appears in the denominator, this metric should be maximized
  - number of predicted defaults that did default, divided by the number of loans that did default
  - this is the fraction of loans that shouldn't be funded (because their true outcome was a default) that were not funded since the investor has followed the predictions (default) made here
- `FPR = FP / TN + FP`
  - called [fall out](https://en.wikipedia.org/wiki/False_positive_rate)
  - this minimizes `FP` as required
  - since `FP` appears in the numerator, this metric should be minimized
  - number of predicted defaults that did not default, divided by the number of loans did not default
  - this is the fraction of loans that should be funded (because the true outcome was the loan being paid off on time) that were not funded since the investor followed the predictions (default) generated by analysis performed here

Both metrics will be used as much as possible, but the best model found during the analysis phase will be chosen based on the `TPR` (or recall).


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

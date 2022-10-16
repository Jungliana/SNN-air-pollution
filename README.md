SNN-pollution
==============================

A short description of the project.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump. 
    │
    ├── code      <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── notebooks      <- Jupyter notebooks.

--------


Setting up the environment
------------

1. Install `poetry`: https://python-poetry.org/docs/#installation
2. Create an environment with `poetry install`
3. Run `poetry shell`
4. To add a new package run `poetry add <package>`. Don't forget to commit the lockfile.
5. To run unit tests for your service use `poetry run pytest` or simply `pytest` within `poetry shell`.

<p><small>Project partially based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

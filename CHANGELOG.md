# CHANGELOG

### 2021-05-26 !4

- update `README` how to join competition
- git lfs track zarr: `git lfs track "**/*.zarr/**"`
- add notebooks:
    - create renku datasets: `renku_datasets_biweekly.ipynb`
    - RPSS verification: `RPSS_verification.ipynb`
    - ML train and predict based on weatherbench: `ML_train_and_predict.ipynb`
    - mean bias reduction: `mean_bias_reduction.ipynb`
    - template for training and predictions: `ML_forecast_template.ipynb`
- add renku dataset `s2s-ai-challenge` with files:
    - `hindcast-like-observations_2000-2019_biweekly_deterministic.zarr`
    - `forecast-like-observations_2020_biweekly_deterministic.zarr`
    - `hindcast-like-observations_2000-2019_biweekly_tercile-edges.nc`
    - `hindcast-like-observations_2000-2019_biweekly_terciled.zarr`
    - `forecast-like-observations_2020_biweekly_terciled.nc`
    - `ecmwf_forecast-input_2020_biweekly_deterministic.zarr`
    - `ecmwf_hindcast-input_2000-2019_biweekly_deterministic.zarr`
    - `ecmwf_recalibrated_benchmark_2020_biweekly_terciled.nc`
- add reproducibility section below in training
- how to deal with this dry mask? provide as renku dataset? now implicitly masked in categorized observations `obs_p`
- justify if training takes more than a week
- show RPS for all years ToDo: take RPSS



### 2021-05-10

- `git lfs track “submissions/*.nc”`, so submission netcdf files are not carried in `git` but `git lfs`
- add `notebooks/ML_prediction.ipynb` as a template for submission notebooks with safeguards.

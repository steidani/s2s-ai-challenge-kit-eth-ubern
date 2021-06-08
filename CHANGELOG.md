# CHANGELOG

### unreleased

- Add notebooks showcasing accessing output of different models from different sources: (!2, [Aaron Spring](https://renkulab.io/gitlab/aaron.spring))
    - S2S-Project models:
        - from from European Weather Cloud:
            - [`climetlab-s2s-ai-challenge`](https://github.com/ecmwf-lab/climetlab-s2s-ai-challenge/) [recommended], see [`climetlab-s2s-ai-challenge` notebooks](https://github.com/ecmwf-lab/climetlab-s2s-ai-challenge/tree/main/notebooks)
            - `curl` & `wget`, see [wget_curl.ipynb](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/blob/master/notebooks/data_access/wget_curl.ipynb)
            - `intake`, see [intake.ipynb](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/blob/master/notebooks/data_access/intake.ipynb)
        - `IRIDL` including overview, see [IRIDL.ipynb](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/blob/master/notebooks/data_access/IRIDL.ipynb)
    - SubX-Project models: `IRIDL` including overview, see [IRIDL.ipynb](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/blob/master/notebooks/data_access/IRIDL.ipynb)
    - How to access password-protected S2S-Project output from IRIDL with xarray? see [IRIDL.ipynb](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/blob/master/notebooks/data_access/IRIDL.ipynb)
- fix `netcdf4` version to `1.5.4` for `opendap` to work lazily with `xarray` (!2, !7, [Aaron Spring](https://renkulab.io/gitlab/aaron.spring))


### 2021-05-31: `v0.2` *release*

After this `v0.2` release, this CHANGELOG.md will describe all changes made in this template repository.

- update `README` how to join competition, please `git pull` if you forked before
- find status of your submission in `s2s-ai-competition-scoring-image` https://renkulab.io/gitlab/tasko.olevski/s2s-ai-competition-scoring-image/-/blob/master/README.md 
- calculate `RPSS` with respect to climatology (not ECMWF anymore) ([Aaron Spring](https://renkulab.io/gitlab/aaron.spring))
    - update `RPSS_verification.ipynb`
    - update `scorer`: https://renkulab.io/gitlab/tasko.olevski/s2s-ai-competition-scoring-image ([Tasko Olevski](https://renkulab.io/gitlab/tasko.olevski), [Aaron Spring](https://renkulab.io/gitlab/aaron.spring))
    - Averaged ECMWF RPSS skill value to beat at least: -0.0070


### 2021-05-26: `v0.1` *pre-release*

- update `README` how to join competition (!4, [Aaron Spring](https://renkulab.io/gitlab/aaron.spring))
- git lfs track zarr: `git lfs track "**/*.zarr/**"` (!4, [Aaron Spring](https://renkulab.io/gitlab/aaron.spring))
- add notebooks: (!4, [Aaron Spring](https://renkulab.io/gitlab/aaron.spring))
    - create renku datasets: `renku_datasets_biweekly.ipynb`
    - RPSS verification: `RPSS_verification.ipynb`
    - ML train and predict based on weatherbench: `ML_train_and_predict.ipynb`
    - mean bias reduction: `mean_bias_reduction.ipynb`
    - template for training and predictions: `ML_forecast_template.ipynb`
- add renku dataset `s2s-ai-challenge` with files: (!4, [Aaron Spring](https://renkulab.io/gitlab/aaron.spring))
    - `hindcast-like-observations_2000-2019_biweekly_deterministic.zarr`
    - `forecast-like-observations_2020_biweekly_deterministic.zarr`
    - `hindcast-like-observations_2000-2019_biweekly_tercile-edges.nc`
    - `hindcast-like-observations_2000-2019_biweekly_terciled.zarr`
    - `forecast-like-observations_2020_biweekly_terciled.nc`
    - `ecmwf_forecast-input_2020_biweekly_deterministic.zarr`
    - `ecmwf_hindcast-input_2000-2019_biweekly_deterministic.zarr`
    - `ecmwf_recalibrated_benchmark_2020_biweekly_terciled.nc`
- add reproducibility section below in training (!4, [Aaron Spring](https://renkulab.io/gitlab/aaron.spring))
- how to deal with this dry mask? provide as renku dataset? now implicitly masked in categorized observations `obs_p`
- justify if training takes more than a week (!4, [Aaron Spring](https://renkulab.io/gitlab/aaron.spring))
- show RPS for all years. ~~ToDo: take RPSS~~ (!4, [Aaron Spring](https://renkulab.io/gitlab/aaron.spring))



### 2021-05-10

- `git lfs track “submissions/*.nc”`, so submission netcdf files are not carried in `git` but `git lfs`
- add `notebooks/ML_prediction.ipynb` as a template for submission notebooks with safeguards.

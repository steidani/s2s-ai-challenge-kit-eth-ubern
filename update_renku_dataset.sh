
echo remove previous zarrs from renku dataset

renku dataset unlink s2s-ai-challenge --include "data/hindcast-like-observations_2000-2019_biweekly_deterministic.zarr/**/**" -y

renku dataset unlink s2s-ai-challenge --include "data/hindcast-like-observations_2000-2019_biweekly_terciled.zarr/**/**" -y

renku dataset unlink s2s-ai-challenge --include "data/ecmwf_hindcast-input_2000-2019_biweekly_deterministic.zarr/**/**" -y

renku dataset unlink s2s-ai-challenge --include "data/forecast-like-observations_2020_biweekly_deterministic.zarr/**/**" -y

renku dataset unlink s2s-ai-challenge --include "data/ecmwf_forecast-input_2020_biweekly_deterministic.zarr/**/**" -y

renku dataset unlink s2s-ai-challenge --include "data/**/**" -y


# observations and derived
renku dataset add s2s-ai-challenge data/hindcast-like-observations_2000-2019_biweekly_deterministic.zarr

renku dataset add s2s-ai-challenge data/forecast-like-observations_2020_biweekly_deterministic.zarr

renku dataset add s2s-ai-challenge -o data/hindcast-like-observations_2000-2019_biweekly_tercile-edges.nc

renku dataset add s2s-ai-challenge -o data/forecast-like-observations_2020_biweekly_terciled.nc

renku dataset add s2s-ai-challenge data/hindcast-like-observations_2000-2019_biweekly_terciled.zarr

# benchmark
renku dataset add s2s-ai-challenge -o data/ecmwf_recalibrated_benchmark_2020_biweekly_terciled.nc

# forecast / hindcast
for center in ecmwf; do
    renku dataset add s2s-ai-challenge data/${center}_hindcast-input_2000-2019_biweekly_deterministic.zarr

    renku dataset add s2s-ai-challenge data/${center}_forecast-input_2020_biweekly_deterministic.zarr

done

renku dataset ls-files s2s-ai-challenge

renku dataset ls-tags s2s-ai-challenge

echo consider new tag: renku dataset tag -d description s2s-ai-challenge 0.x

import itertools
from dataclasses import dataclass

import numpy as np
import xarray as xr
import xesmf as xe


@dataclass
class RomsVariable:
    name: str
    lon_name: str
    lat_name: str
    eta_name: str
    xi_name: str


def regrid(ds_grid, ds_data, parameter_name, lon_name, lat_name):
    ds_grid = ds_grid.rename({lon_name: "lon", lat_name: "lat"})
    ds_data = ds_data.rename({lon_name: "lon", lat_name: "lat"})
    regridder = xe.Regridder(ds_data, ds_grid, 'bilinear', unmapped_to_nan=True)
    return regridder(ds_data[parameter_name])


def fit_grid(ds_grid, da, eta_name, xi_name):
    da = da.interpolate_na(dim=eta_name, method="nearest", fill_value="extrapolate")
    da = da.interpolate_na(dim=xi_name, method="nearest", fill_value="extrapolate")
    return ds_grid.mask_rho * da


def fill_variables():
    variables = []
    variables.append(RomsVariable("temp", "lon_rho", "lat_rho", "eta_rho", "xi_rho"))
    return variables


def regrid_fit(ds_grid, ds_data, roms_variable):
    """
    Finds a roms_variable in ds_data and then
    regrids and interpolates + extrapolates it to lats and lons from ds_grid
    """
    da = regrid(ds_grid,
                ds_data,
                roms_variable.name,
                roms_variable.lon_name,
                roms_variable.lat_name
                )
    return fit_grid(ds_grid,
                    da,
                    roms_variable.eta_name,
                    roms_variable.xi_name
                    )


def get_slices(steps: int, num: int):
    samples = np.linspace(1, steps, num).astype(int)  # astype int rounds down
    return itertools.pairwise(samples)


def clim_of800to160():
    ds_of160_grid = xr.open_dataset('/cluster/projects/nn9297k/OF160/Grid/OF160_grid_v1.nc')
    ds_of800_clim = xr.open_dataset('/cluster/projects/nn9297k/OF800/data_clim/OF800_his_merged.nc')
    roms_variables = fill_variables()
    time_steps = ds_of800_clim.dims['ocean_time']
    for var in roms_variables:
        da = regrid_fit(ds_of160_grid, ds_of800_clim.isel(ocean_time=0), var)
        for i, time_slice in enumerate(get_slices(time_steps, 30)):
            da = xr.concat(
                [da, regrid_fit(ds_of160_grid, ds_of800_clim.isel(ocean_time=slice(*time_slice)), var)],
                dim="ocean_time",
                )
            xr.Dataset({
                var.name: da
            }).to_netcdf(
                f'/cluster/projects/nn9297k/OF160/Clm/OF160_clm_{var.name}_{i}.nc'
                )


if __name__ == "__main__":
    clim_of800to160()

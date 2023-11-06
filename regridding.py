import itertools
from multiprocessing import Pool
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
    mask_name: str


def regrid(ds_grid, ds_data, parameter_name, lon_name, lat_name):
    ds_grid = ds_grid.rename({lon_name: "lon", lat_name: "lat"})
    ds_data = ds_data.rename({lon_name: "lon", lat_name: "lat"})
    regridder = xe.Regridder(ds_data, ds_grid, 'bilinear', unmapped_to_nan=True)
    return regridder(ds_data[parameter_name])


def fit_grid(ds_grid, da, eta_name, xi_name, mask_name):
    da = da.interpolate_na(dim=eta_name, method="nearest", fill_value="extrapolate")
    da = da.interpolate_na(dim=xi_name, method="nearest", fill_value="extrapolate")
    return ds_grid[mask_name] * da


def fill_variables():
    variables = []
    variables.append(RomsVariable("temp", "lon_rho", "lat_rho", "eta_rho", "xi_rho", "mask_rho"))
    variables.append(RomsVariable("salt", "lon_rho", "lat_rho", "eta_rho", "xi_rho", "mask_rho"))
    variables.append(RomsVariable("u", "lon_u", "lat_u", "eta_u", "xi_u", "mask_u"))
    variables.append(RomsVariable("ubar", "lon_u", "lat_u", "eta_u", "xi_u", "mask_u"))
    variables.append(RomsVariable("v", "lon_v", "lat_v", "eta_v", "xi_v", "mask_v"))
    variables.append(RomsVariable("vbar", "lon_v", "lat_v", "eta_v", "xi_v", "mask_v"))
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
                    roms_variable.xi_name,
                    roms_variable.mask_name
                    )


def get_slices(steps: int, num: int):
    samples = np.linspace(0, steps, num).astype(int)  # astype int rounds down
    return itertools.pairwise(samples)


def f(ds_grid, ds_data, var):
    time_steps = ds_data.dims['ocean_time']
    for i, time_slice in enumerate(get_slices(time_steps, 100)):
        da = regrid_fit(ds_grid, ds_data.isel(ocean_time=slice(*time_slice)), var)
        xr.Dataset({var.name: da}).to_netcdf(
            f'/cluster/projects/nn9297k/OF160/Clm/{i:03d}_OF160_clm_{var.name}.nc'
            )
        print(f"Variable: {var.name} iteration {i:03d} saved")


if __name__ == "__main__":
    roms_variables = fill_variables()

    def wrapper(x):
        ds_of160_grid = xr.open_dataset('/cluster/projects/nn9297k/OF160/Grid/OF160_grid_v1.nc')
        ds_of800_clim = xr.open_dataset('/cluster/projects/nn9297k/OF800/data_clim/OF800_his_merged.nc')
        f(ds_of160_grid, ds_of800_clim, x)

    # lambda cannot be pickled (python3.10)
    with Pool(processes=len(roms_variables)) as p:
        p.map(wrapper, roms_variables)

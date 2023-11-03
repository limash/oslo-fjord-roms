import xesmf as xe

def regrid(ds_grid, ds_data, parameter_name, lon_name, lat_name):
    ds_grid = ds_grid.rename({lon_name: "lon", lat_name: "lat"})
    ds_data = ds_data.rename({lon_name: "lon", lat_name: "lat"})
    regridder = xe.Regridder(ds_data, ds_grid, 'bilinear', unmapped_to_nan=True)
    return regridder(ds_data[parameter_name])

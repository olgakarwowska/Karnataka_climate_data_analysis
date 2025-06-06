#%% 
### Loading the packages needed for this script

#pip install imdlib #do this once before running the code below
#pip install netCDF4 h5netcdf #do this once before running the code below ?

import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import os
import rasterio
import imdlib as imd

#%% 
### DOWNLOAD DATASET
#import imdlib as imd

# Downloading all years of rainfall, tmax or tmin data for India
# Variable data that can be downloaded are rain, tmax and tmin
#variable = 'rain'  
variable = 'tmax'
#variable = 'tmin'
#start_yr = 1901 # for rain, data is available from 1901
start_yr = 1951 # for tmax and tmin, data is available from 1951
end_yr =  2024    #2024 for rain, tmax and tmin
file_dir = 'C:\\Users\\Monie\\OneDrive - Wageningen University & Research\\Design of Climate Change Mitigation and Adaptation Strategies\\Data of project\\dataimd' # direct to this where you want to store the grd data files
"""
fn_format   : str or None
        fn_format represent filename format. 
        Default vales is None.  Which means filesnames are accoding to the IMD naming convention
        If we specify fn_format = 'yearwise', filenames are renamed like <year.grd>

file_dir   : str or None
        Directory for downloading the files.
        If None, the currently working directory is used.

sub_dir : bool
		True : if you need subdirectory for each variable type;
        False: Files will be saved directly under main directory
proxies : dict
        Give details in curly bracket as shown in the example below
        e.g. proxies = { 'http' : 'http://uname:password@ip:port'}
"""
data = imd.get_data(variable, start_yr, end_yr, fn_format='yearwise', file_dir=file_dir)



#%%
### RAIN DATA
### Convert .grd to .nc (output = one csv file for datapoints within the region for all years)
variable = 'rain'
start_yr = 1901
end_yr = 2024
file_format = 'yearwise' # other option (None), which will assume default imd naming convention
file_dir = 'C:\\Users\\Monie\\OneDrive - Wageningen University & Research\\Design of Climate Change Mitigation and Adaptation Strategies\\Data of project\\dataimd'   # file directory where you want to store the nc file you are going to create           #'rain' # other option is to keep this blank
data = imd.open_data(variable, start_yr, end_yr,'yearwise', file_dir=file_dir) #opens the grd files downloadeded earlier
data.to_netcdf('alldata_rain_India.nc', file_dir) #make one nc file of all downloaded grd files

## Raichur
### Convert .grd to .nc of rain data for Raichur (output = one csv file for datapoints within the region for all years)
raichur = xr.open_dataset ('alldata_rain_India.nc')
raichur = raichur.rio.write_crs('EPSG:4326') #set coordinate system
raichurshp = gpd.read_file('C:\\Users\\Monie\\OneDrive - Wageningen University & Research\\Design of Climate Change Mitigation and Adaptation Strategies\\Data of project\\dataimd\\Raichur_Tlab.shp') # reads the Raichur shapefile
raichurshp = raichurshp.to_crs('EPSG:4326')

#raichur.rio.bounds()                   #check coordinate system
#print(raichurshp.total_bounds)         #check coordinate system
#check if coordinate systems are the same

#clipping netcdf data of raichur to raichur district polygon
clipped_raichur = raichur.rio.clip(raichurshp.geometry,raichurshp.crs, drop = True)
print(clipped_raichur.data_vars)
clipped_raichur = clipped_raichur.to_dataframe().reset_index()
clipped_raichur.to_csv('clipped_raichur_rain.csv', index = False)

## Chikkaballapur
#### Convert .grd to .nc of rain data for Chikkaballapur (output = one csv file for datapoints within the region for all years)
chikkaballapur = xr.open_dataset ('alldata_rain_India.nc')
chikkaballapur = chikkaballapur.rio.write_crs('EPSG:4326') #set coordinate system
chikkaballapurshp = gpd.read_file('C:\\Users\\Monie\\OneDrive - Wageningen University & Research\\Design of Climate Change Mitigation and Adaptation Strategies\\Data of project\\dataimd\\Chikkaballapura_villages.shp') # reads the Chikkaballapur shapefile
chikkaballapurshp = chikkaballapurshp.to_crs('EPSG:4326')

#clipping netcdf data of chikkaballapur to chikkaballapur district polygon
clipped_chikkaballapur = chikkaballapur.rio.clip(chikkaballapurshp.geometry,chikkaballapur.rio.crs, drop = True)
print(clipped_chikkaballapur.data_vars)
clipped_chikkaballapur = clipped_chikkaballapur.to_dataframe().reset_index()
clipped_chikkaballapur.to_csv('clipped_chikkaballapur_rain.csv', index = False)



#%%
### TMAX DATA
### Convert .grd to .nc (output = one csv file for datapoints within the region for all years)
variable = 'tmax'
start_yr = 1951
end_yr = 2024
file_format = 'yearwise' # other option (None), which will assume deafult imd naming convention
file_dir = 'C:\\Users\\Monie\\OneDrive - Wageningen University & Research\\Design of Climate Change Mitigation and Adaptation Strategies\\Data of project\\dataimd'   # file directory where you want to store the nc file you are going to create           #'rain' # other option is to keep this blank
data = imd.open_data(variable, start_yr, end_yr,'yearwise', file_dir=file_dir) #opens the grd files downloaded earlier
data.to_netcdf('alldata_tmax_India.nc', file_dir) #make one nc file of all downloaded grd files

## Raichur
### Convert .grd to .nc of tmax data for Raichur (output = one csv file for datapoints within the region for all years)
raichur = xr.open_dataset ('alldata_tmax_India.nc')
raichur = raichur.rio.write_crs('EPSG:4326') #set coordinate system
raichurshp = gpd.read_file('C:\\Users\\Monie\\OneDrive - Wageningen University & Research\\Design of Climate Change Mitigation and Adaptation Strategies\\Data of project\\dataimd\\Raichur_Tlab.shp') # reads the Raichur shapefile
raichurshp = raichurshp.to_crs('EPSG:4326')

#clipping netcdf data of raichur to raichur district polygon
clipped_raichur = raichur.rio.clip(raichurshp.geometry,raichurshp.crs, drop = True)
print(clipped_raichur.data_vars)
clipped_raichur = clipped_raichur.to_dataframe().reset_index()
clipped_raichur.to_csv('clipped_raichur_tmax.csv', index = False)

## Chikkaballapur
#### Convert .grd to .nc of tmax data for Chikkaballapur (output = one csv file for datapoints within the region for all years)
chikkaballapur = xr.open_dataset ('alldata_tmax_India.nc')
chikkaballapur = chikkaballapur.rio.write_crs('EPSG:4326') #set coordinate system
chikkaballapurshp = gpd.read_file('C:\\Users\\Monie\\OneDrive - Wageningen University & Research\\Design of Climate Change Mitigation and Adaptation Strategies\\Data of project\\dataimd\\Chikkaballapura_villages.shp') # reads the Chikkaballapur shapefile
chikkaballapurshp = chikkaballapurshp.to_crs('EPSG:4326')

#clipping netcdf data of chikkaballapur to chikkaballapur district polygon
clipped_chikkaballapur = chikkaballapur.rio.clip(chikkaballapurshp.geometry,chikkaballapur.rio.crs, drop = True)
print(clipped_chikkaballapur.data_vars)
clipped_chikkaballapur = clipped_chikkaballapur.to_dataframe().reset_index()
clipped_chikkaballapur.to_csv('clipped_chikkaballapur_tmax.csv', index = False)



#%%
### TMIN DATA
### Convert .grd to .nc (output = one csv file for datapoints within the region for all years)
variable = 'tmin'
start_yr = 1951
end_yr = 2024
file_format = 'yearwise' # other option (None), which will assume deafult imd naming convention
file_dir = 'C:\\Users\\Monie\\OneDrive - Wageningen University & Research\\Design of Climate Change Mitigation and Adaptation Strategies\\Data of project\\dataimd'   # file directory where you want to store the nc file you are going to create           #'rain' # other option is to keep this blank
data = imd.open_data(variable, start_yr, end_yr,'yearwise', file_dir=file_dir) #opens the grd files downloaded earlier
data.to_netcdf('alldata_tmin_India.nc', file_dir) #make one nc file of all downloaded grd files

## Raichur
### Convert .grd to .nc of tmin data for Raichur (output = one csv file for datapoints within the region for all years)
raichur = xr.open_dataset ('alldata_tmin_India.nc')
raichur = raichur.rio.write_crs('EPSG:4326') #set coordinate system
raichurshp = gpd.read_file('C:\\Users\\Monie\\OneDrive - Wageningen University & Research\\Design of Climate Change Mitigation and Adaptation Strategies\\Data of project\\dataimd\\Raichur_Tlab.shp') # reads the Raichur shapefile
raichurshp = raichurshp.to_crs('EPSG:4326')

#clipping netcdf data of raichur to raichur district polygon
clipped_raichur = raichur.rio.clip(raichurshp.geometry,raichurshp.crs, drop = True)
print(clipped_raichur.data_vars)
clipped_raichur = clipped_raichur.to_dataframe().reset_index()
clipped_raichur.to_csv('clipped_raichur_tmin.csv', index = False)

## Chikkaballapur
#### Convert .grd to .nc of tmin data for Chikkaballapur (output = one csv file for datapoints within the region for all years)
chikkaballapur = xr.open_dataset ('alldata_tmin_India.nc')
chikkaballapur = chikkaballapur.rio.write_crs('EPSG:4326') #set coordinate system
chikkaballapurshp = gpd.read_file('C:\\Users\\Monie\\OneDrive - Wageningen University & Research\\Design of Climate Change Mitigation and Adaptation Strategies\\Data of project\\dataimd\\Chikkaballapura_villages.shp') # reads the Chikkaballapur shapefile
chikkaballapurshp = chikkaballapurshp.to_crs('EPSG:4326')

#clipping netcdf data of chikkaballapur to chikkaballapur district polygon
clipped_chikkaballapur = chikkaballapur.rio.clip(chikkaballapurshp.geometry,chikkaballapur.rio.crs, drop = True)
print(clipped_chikkaballapur.data_vars)
clipped_chikkaballapur = clipped_chikkaballapur.to_dataframe().reset_index()
clipped_chikkaballapur.to_csv('clipped_chikkaballapur_tmin.csv', index = False)







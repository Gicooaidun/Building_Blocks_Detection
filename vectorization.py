import os
import cv2
import numpy as np
from osgeo import gdal, ogr, osr
import geopandas as gpd



def vectorize(input_dir, output_dir):
    """
    Creates vector polygons from binary classification pixel image.
    :param input_dir: directory with PNG files of binary buliding block predictions.
    :param output_dir: directory to store Shape files of vectorized and generalized buliding block.
    """

    # logging
    print("start vectorization of prediction images")

    # get all image files for post processing
    files = os.listdir(input_dir)

    for file in files:
        # get sheet number and epoch
        sheet_number = file.split("_")[0]
        epoch_number = file.split("_")[-1].split(".")[0]

        # set paths
        file_in_path = f"{input_dir}/{file}"
        file_out_path_raw = f"{output_dir}/{sheet_number}_polygons_raw_epoch_{epoch_number}.shp"
        file_out_path_generalized = f"{output_dir}/{sheet_number}_polygons_generalized_epoch_{epoch_number}.shp"

        # read image
        image = gdal.Open(file_in_path)
        band = image.GetRasterBand(1)

        # vectorize image and save raw polygons
        drv = ogr.GetDriverByName("ESRI Shapefile")
        dst_ds = drv.CreateDataSource(file_out_path_raw)
        dst_layer = dst_ds.CreateLayer("polygons")
        gdal.Polygonize(band, band, dst_layer, -1, [], callback=None)
        dst_layer = None
        dst_ds = None

        # generalize and save generalized polygons
        buffer = 10
        gdf = gpd.read_file(file_out_path_raw)
        gdf["geometry"] = gdf["geometry"].simplify(buffer)
        gdf.to_file(file_out_path_generalized, driver="ESRI Shapefile")

        # logging
        print(f"sheet {sheet_number} of epoch {epoch_number} successfully vectorized and generalized")
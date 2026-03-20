#include "terrain_loader.h"

#include <cmath>
#include <stdexcept>

#include <gdal_priv.h>

namespace {

float sanitizeElevation(float value, bool hasNoData, double noDataValue) {
    if (!hasNoData) {
        return value;
    }
    if (std::isnan(noDataValue)) {
        return std::isnan(value) ? 0.0f : value;
    }
    return (std::fabs(static_cast<double>(value) - noDataValue) < 1e-6) ? 0.0f : value;
}

}  // namespace

DemData loadDemFromFile(const std::string& path) {
    GDALAllRegister();

    GDALDataset* dataset = static_cast<GDALDataset*>(GDALOpen(path.c_str(), GA_ReadOnly));
    if (!dataset) {
        throw std::runtime_error("Failed to open DEM file: " + path);
    }

    DemData dem;
    dem.width = dataset->GetRasterXSize();
    dem.height = dataset->GetRasterYSize();

    if (dem.width <= 0 || dem.height <= 0) {
        GDALClose(dataset);
        throw std::runtime_error("DEM has invalid raster size: " + path);
    }

    double geotransform[6] = {};
    if (dataset->GetGeoTransform(geotransform) != CE_None) {
        GDALClose(dataset);
        throw std::runtime_error("Failed to read DEM geotransform: " + path);
    }

    dem.originX = geotransform[0];
    dem.pixelSizeX = geotransform[1];
    dem.originY = geotransform[3];
    dem.pixelSizeY = std::fabs(geotransform[5]);

    GDALRasterBand* band = dataset->GetRasterBand(1);
    if (!band) {
        GDALClose(dataset);
        throw std::runtime_error("DEM is missing raster band 1: " + path);
    }

    int hasNoDataFlag = 0;
    dem.noDataValue = band->GetNoDataValue(&hasNoDataFlag);
    dem.hasNoData = hasNoDataFlag != 0;

    dem.elevation.resize(static_cast<size_t>(dem.width) * static_cast<size_t>(dem.height));
    const CPLErr ioStatus = band->RasterIO(
        GF_Read,
        0,
        0,
        dem.width,
        dem.height,
        dem.elevation.data(),
        dem.width,
        dem.height,
        GDT_Float32,
        0,
        0
    );
    GDALClose(dataset);

    if (ioStatus != CE_None) {
        throw std::runtime_error("Failed to read DEM pixels via GDAL RasterIO: " + path);
    }

    for (float& value : dem.elevation) {
        value = sanitizeElevation(value, dem.hasNoData, dem.noDataValue);
    }

    return dem;
}

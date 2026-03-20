#include "terrain_loader.h"

#include <cmath>
#include <stdexcept>

#include <gdal_priv.h>

DemInfo loadDemInfo(const std::string& demPath) {
    GDALAllRegister();

    GDALDataset* dataset = static_cast<GDALDataset*>(GDALOpen(demPath.c_str(), GA_ReadOnly));
    if (!dataset) {
        throw std::runtime_error("Failed to open DEM file: " + demPath);
    }

    DemInfo info{};
    info.width = dataset->GetRasterXSize();
    info.height = dataset->GetRasterYSize();

    double geoTransform[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (dataset->GetGeoTransform(geoTransform) != CE_None) {
        GDALClose(dataset);
        throw std::runtime_error("Failed to read DEM geotransform.");
    }

    info.dx = static_cast<float>(std::fabs(geoTransform[1]));
    info.dy = static_cast<float>(std::fabs(geoTransform[5]));

    GDALRasterBand* band = dataset->GetRasterBand(1);
    if (!band) {
        GDALClose(dataset);
        throw std::runtime_error("DEM does not contain band 1.");
    }

    int hasNoData = 0;
    const double noData = band->GetNoDataValue(&hasNoData);
    info.hasNoData = (hasNoData != 0);
    info.noDataValue = info.hasNoData ? static_cast<float>(noData) : 0.0f;

    info.elevation.resize(static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height));
    const CPLErr readErr = band->RasterIO(
        GF_Read,
        0,
        0,
        info.width,
        info.height,
        info.elevation.data(),
        info.width,
        info.height,
        GDT_Float32,
        0,
        0
    );
    GDALClose(dataset);

    if (readErr != CE_None) {
        throw std::runtime_error("Failed to read DEM raster values.");
    }

    return info;
}

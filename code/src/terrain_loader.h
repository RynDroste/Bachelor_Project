#pragma once

#include <string>
#include <vector>

struct DemData {
    int width = 0;
    int height = 0;
    double originX = 0.0;
    double originY = 0.0;
    double pixelSizeX = 0.0;
    double pixelSizeY = 0.0;
    bool hasNoData = false;
    double noDataValue = 0.0;
    std::vector<float> elevation;
};

DemData loadDemFromFile(const std::string& path);

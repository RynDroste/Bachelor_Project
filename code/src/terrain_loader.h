#pragma once

#include <string>
#include <vector>

struct DemInfo {
    int width;
    int height;
    float dx;
    float dy;
    float noDataValue;
    bool hasNoData;
    std::vector<float> elevation;
};

DemInfo loadDemInfo(const std::string& demPath);

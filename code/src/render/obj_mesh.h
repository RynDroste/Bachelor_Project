#pragma once
#include <string>
#include <vector>

struct ObjMesh {
    struct MeshGroup {
        int         startVertex = 0;
        int         vertexCount = 0;
        std::string diffuseTex; // absolute path to map_Kd, or empty
    };

    // interleaved: pos(3) + normal(3) + uv(2) = 8 floats per vertex, flat normals
    std::vector<float>     verts;
    int                    vertexCount = 0;
    std::vector<MeshGroup> groups;

    bool load(const std::string& path);
};

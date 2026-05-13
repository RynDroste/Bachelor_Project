#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "render/obj_mesh.h"

#include <glm/glm.hpp>

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

static std::string dirOf(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? "" : path.substr(0, pos + 1);
}

bool ObjMesh::load(const std::string& path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    const std::string dir = dirOf(path);
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          path.c_str(), dir.c_str(), /*triangulate=*/true)) {
        std::fprintf(stderr, "ObjMesh: failed to load '%s': %s\n", path.c_str(), err.c_str());
        return false;
    }
    if (!warn.empty())
        std::fprintf(stderr, "ObjMesh warn (%s): %s\n", path.c_str(), warn.c_str());
    if (attrib.vertices.empty()) {
        std::fprintf(stderr, "ObjMesh: no geometry in '%s'\n", path.c_str());
        return false;
    }

    // Collect indices per material id, preserving first-seen order across all shapes.
    std::vector<int> matOrder;
    std::unordered_map<int, std::vector<tinyobj::index_t>> matBuckets;

    for (const auto& shape : shapes) {
        size_t idxOff = 0;
        for (size_t fi = 0; fi < shape.mesh.num_face_vertices.size(); fi++) {
            int mid = shape.mesh.material_ids.empty() ? -1 : shape.mesh.material_ids[fi];
            if (matBuckets.find(mid) == matBuckets.end())
                matOrder.push_back(mid);
            auto& bucket = matBuckets[mid];
            int fv = shape.mesh.num_face_vertices[fi];
            for (int v = 0; v < fv; v++)
                bucket.push_back(shape.mesh.indices[idxOff + v]);
            idxOff += fv;
        }
    }

    constexpr int kStride = 8; // pos(3) + normal(3) + uv(2)
    verts.clear();
    groups.clear();
    vertexCount = 0;

    for (int mid : matOrder) {
        const auto& idxList = matBuckets[mid];
        const int triCount  = static_cast<int>(idxList.size()) / 3;
        if (triCount <= 0) continue;

        MeshGroup mg;
        mg.startVertex = vertexCount;
        mg.vertexCount = triCount * 3;
        if (mid >= 0 && mid < static_cast<int>(materials.size()) &&
                !materials[mid].diffuse_texname.empty())
            mg.diffuseTex = dir + materials[mid].diffuse_texname;
        groups.push_back(mg);

        verts.resize(verts.size() + triCount * 3 * kStride);
        float* dst = verts.data() + mg.startVertex * kStride;

        for (int ti = 0; ti < triCount; ti++) {
            const tinyobj::index_t* idx = &idxList[ti * 3];

            glm::vec3 pos[3];
            for (int k = 0; k < 3; k++) {
                int vi    = idx[k].vertex_index * 3;
                pos[k]    = {attrib.vertices[vi], attrib.vertices[vi+1], attrib.vertices[vi+2]};
            }
            // Flat normal as fallback; use per-vertex normals from file if present.
            glm::vec3 flatN = glm::cross(pos[1] - pos[0], pos[2] - pos[0]);
            float     len   = glm::length(flatN);
            if (len > 1e-10f) flatN /= len;

            for (int k = 0; k < 3; k++) {
                glm::vec3 n = flatN;
                if (idx[k].normal_index >= 0) {
                    int ni = idx[k].normal_index * 3;
                    n = {attrib.normals[ni], attrib.normals[ni+1], attrib.normals[ni+2]};
                }
                glm::vec2 uv{0.f, 0.f};
                if (idx[k].texcoord_index >= 0) {
                    int ti2 = idx[k].texcoord_index * 2;
                    uv = {attrib.texcoords[ti2], attrib.texcoords[ti2+1]};
                }
                *dst++ = pos[k].x; *dst++ = pos[k].y; *dst++ = pos[k].z;
                *dst++ = n.x;      *dst++ = n.y;      *dst++ = n.z;
                *dst++ = uv.x;     *dst++ = uv.y;
            }
        }
        vertexCount += triCount * 3;
    }

    std::printf("ObjMesh: loaded '%s' — %zu positions, %d triangles, %zu groups\n",
                path.c_str(), attrib.vertices.size() / 3,
                vertexCount / 3, groups.size());
    return !groups.empty();
}

#pragma once
#include <glm/glm.hpp>
#include <vector>

struct ObjMesh;

struct BVHHit {
    float t    = -1.f; // ray parameter; negative = no hit
    int triIdx = -1;   // triangle index into ObjMesh (vertexCount/3 triangles)
};

struct MeshBVH {
    struct Node {
        glm::vec3 aabbMin{ 1e30f,  1e30f,  1e30f};
        glm::vec3 aabbMax{-1e30f, -1e30f, -1e30f};
        int left     = -1; // internal: left child; -1 = leaf
        int right    = -1;
        int triStart =  0;
        int triCount =  0; // 0 = internal node
    };

    std::vector<Node>      nodes;
    std::vector<int>       triIndices;      // reordered triangle IDs
    std::vector<glm::vec3> triV0, triV1, triV2; // precomputed per-triangle verts

    // Build from an already-loaded ObjMesh. O(n log n).
    void build(const ObjMesh& mesh);

    // Closest ray hit in [tMin, tMax]. Returns hit.t < 0 on miss.
    BVHHit intersect(const glm::vec3& rayOrigin, const glm::vec3& rayDir,
                     float tMin = 1e-4f, float tMax = 1e30f) const;

private:
    int buildNode(int* begin, int* end, int depth);
};

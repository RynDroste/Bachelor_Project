#include "render/mesh_bvh.h"
#include "render/obj_mesh.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

// ── Ray tests ─────────────────────────────────────────────────────────────────

static bool rayAABB(const glm::vec3& ro, const glm::vec3& invRd,
                    const glm::vec3& bmin, const glm::vec3& bmax,
                    float tMin, float tMax) {
    for (int i = 0; i < 3; i++) {
        float t0 = (bmin[i] - ro[i]) * invRd[i];
        float t1 = (bmax[i] - ro[i]) * invRd[i];
        if (t0 > t1) std::swap(t0, t1);
        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);
        if (tMax < tMin) return false;
    }
    return true;
}

// Möller–Trumbore; returns t > 0 on hit, -1 on miss.
static float rayTri(const glm::vec3& ro, const glm::vec3& rd,
                    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
    const glm::vec3 e1 = v1 - v0;
    const glm::vec3 e2 = v2 - v0;
    const glm::vec3 h  = glm::cross(rd, e2);
    const float     a  = glm::dot(e1, h);
    if (std::abs(a) < 1e-10f) return -1.f;
    const float     f  = 1.f / a;
    const glm::vec3 s  = ro - v0;
    const float     u  = f * glm::dot(s, h);
    if (u < 0.f || u > 1.f) return -1.f;
    const glm::vec3 q  = glm::cross(s, e1);
    const float     v  = f * glm::dot(rd, q);
    if (v < 0.f || u + v > 1.f) return -1.f;
    return f * glm::dot(e2, q);
}

// ── Build ─────────────────────────────────────────────────────────────────────

void MeshBVH::build(const ObjMesh& mesh) {
    const int nTri = mesh.vertexCount / 3;
    if (nTri == 0) return;

    constexpr int kStride = 8; // pos(3)+normal(3)+uv(2)
    triV0.resize(nTri);
    triV1.resize(nTri);
    triV2.resize(nTri);
    for (int ti = 0; ti < nTri; ti++) {
        const float* p0 = mesh.verts.data() + (ti * 3 + 0) * kStride;
        const float* p1 = mesh.verts.data() + (ti * 3 + 1) * kStride;
        const float* p2 = mesh.verts.data() + (ti * 3 + 2) * kStride;
        triV0[ti] = {p0[0], p0[1], p0[2]};
        triV1[ti] = {p1[0], p1[1], p1[2]};
        triV2[ti] = {p2[0], p2[1], p2[2]};
    }

    triIndices.resize(nTri);
    for (int i = 0; i < nTri; i++) triIndices[i] = i;

    nodes.clear();
    nodes.reserve(nTri * 2); // upper bound: 2n-1 nodes
    buildNode(triIndices.data(), triIndices.data() + nTri, 0);

    std::printf("MeshBVH: built %d nodes for %d triangles\n",
                static_cast<int>(nodes.size()), nTri);
}

int MeshBVH::buildNode(int* begin, int* end, int depth) {
    constexpr int kLeafMax = 4;
    const int count = static_cast<int>(end - begin);

    // Compute node AABB
    glm::vec3 bmin( 1e30f), bmax(-1e30f);
    for (const int* p = begin; p != end; ++p) {
        const int ti = *p;
        bmin = glm::min(bmin, glm::min(triV0[ti], glm::min(triV1[ti], triV2[ti])));
        bmax = glm::max(bmax, glm::max(triV0[ti], glm::max(triV1[ti], triV2[ti])));
    }

    const int nodeIdx = static_cast<int>(nodes.size());
    nodes.push_back({});
    nodes[nodeIdx].aabbMin = bmin;
    nodes[nodeIdx].aabbMax = bmax;

    if (count <= kLeafMax || depth > 24) {
        nodes[nodeIdx].triStart = static_cast<int>(begin - triIndices.data());
        nodes[nodeIdx].triCount = count;
        return nodeIdx;
    }

    // Split on the longest axis at the centroid median
    const glm::vec3 extent = bmax - bmin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;

    int* mid = begin + count / 2;
    std::nth_element(begin, mid, end, [&](int a, int b) {
        const glm::vec3 ca = (triV0[a] + triV1[a] + triV2[a]) * (1.f / 3.f);
        const glm::vec3 cb = (triV0[b] + triV1[b] + triV2[b]) * (1.f / 3.f);
        return ca[axis] < cb[axis];
    });

    // nodes may reallocate during recursion; always access by index
    const int leftIdx  = buildNode(begin, mid, depth + 1);
    const int rightIdx = buildNode(mid,   end,  depth + 1);
    nodes[nodeIdx].left  = leftIdx;
    nodes[nodeIdx].right = rightIdx;
    return nodeIdx;
}

// ── Intersect ─────────────────────────────────────────────────────────────────

BVHHit MeshBVH::intersect(const glm::vec3& ro, const glm::vec3& rd,
                           float tMin, float tMax) const {
    BVHHit result;
    if (nodes.empty()) return result;

    const glm::vec3 invRd(1.f / rd.x, 1.f / rd.y, 1.f / rd.z);

    int  stack[64];
    int  top  = 0;
    float bestT = tMax;
    stack[top++] = 0;

    while (top > 0) {
        const Node& node = nodes[stack[--top]];
        if (!rayAABB(ro, invRd, node.aabbMin, node.aabbMax, tMin, bestT))
            continue;

        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; k++) {
                const int   ti = triIndices[node.triStart + k];
                const float t  = rayTri(ro, rd, triV0[ti], triV1[ti], triV2[ti]);
                if (t > tMin && t < bestT) {
                    bestT         = t;
                    result.t      = t;
                    result.triIdx = ti;
                }
            }
        } else {
            stack[top++] = node.left;
            stack[top++] = node.right;
        }
    }
    return result;
}

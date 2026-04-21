#include "render/clipmap_grid.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Build a (2N+1)*(2N+1) vertex grid of integer positions in [-N..N] for both
// axes and two index buffers: a full solid mesh, and a "ring" mesh with the
// central N x N block of quads removed.
void buildMeshes(int N,
                 std::vector<float>& verts,
                 std::vector<unsigned>& solidIdx,
                 std::vector<unsigned>& ringIdx) {
    const int side = 2 * N + 1;
    verts.clear();
    verts.reserve(static_cast<size_t>(side) * side * 2);
    for (int j = -N; j <= N; ++j) {
        for (int i = -N; i <= N; ++i) {
            verts.push_back(static_cast<float>(i));
            verts.push_back(static_cast<float>(j));
        }
    }

    auto vid = [side, N](int i, int j) {
        return static_cast<unsigned>((i + N) + (j + N) * side);
    };

    solidIdx.clear();
    solidIdx.reserve(static_cast<size_t>(2 * N) * (2 * N) * 6);
    ringIdx.clear();
    ringIdx.reserve(static_cast<size_t>(2 * N) * (2 * N) * 6);

    const int hole = N / 2;  // ring hole half-extent in quad units: [-hole, hole)
    for (int j = -N; j < N; ++j) {
        for (int i = -N; i < N; ++i) {
            const unsigned a = vid(i, j);
            const unsigned b = vid(i + 1, j);
            const unsigned c = vid(i, j + 1);
            const unsigned d = vid(i + 1, j + 1);

            solidIdx.push_back(a);
            solidIdx.push_back(c);
            solidIdx.push_back(b);
            solidIdx.push_back(b);
            solidIdx.push_back(c);
            solidIdx.push_back(d);

            const bool inHole = (i >= -hole) && (i < hole) && (j >= -hole) && (j < hole);
            if (!inHole) {
                ringIdx.push_back(a);
                ringIdx.push_back(c);
                ringIdx.push_back(b);
                ringIdx.push_back(b);
                ringIdx.push_back(c);
                ringIdx.push_back(d);
            }
        }
    }
}

GLuint makeVao(const std::vector<float>& verts,
               const std::vector<unsigned>& idx,
               GLuint& outVbo,
               GLuint& outIbo,
               GLsizei& outCount) {
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &outVbo);
    glGenBuffers(1, &outIbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, outVbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
                 verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, outIbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(idx.size() * sizeof(unsigned)),
                 idx.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float),
                          reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    outCount = static_cast<GLsizei>(idx.size());
    return vao;
}

}  // namespace

bool clipmapGridInit(ClipmapGrid& g, int N, int L, float baseSpacing) {
    if (N <= 0 || (N & 1) != 0 || L <= 0 || baseSpacing <= 0.f) {
        std::fprintf(stderr,
                     "clipmapGridInit: invalid params N=%d L=%d d0=%g (N must be even, L>0, d0>0)\n",
                     N, L, static_cast<double>(baseSpacing));
        return false;
    }
    g.N = N;
    g.L = L;
    g.baseSpacing = baseSpacing;

    std::vector<float>    verts;
    std::vector<unsigned> solidIdx;
    std::vector<unsigned> ringIdx;
    buildMeshes(N, verts, solidIdx, ringIdx);

    g.solidVao = makeVao(verts, solidIdx, g.solidVbo, g.solidIbo, g.solidCount);
    g.ringVao  = makeVao(verts, ringIdx, g.ringVbo, g.ringIbo, g.ringCount);
    return g.solidVao != 0 && g.ringVao != 0;
}

void clipmapGridShutdown(ClipmapGrid& g) {
    if (g.solidVao) glDeleteVertexArrays(1, &g.solidVao);
    if (g.solidVbo) glDeleteBuffers(1, &g.solidVbo);
    if (g.solidIbo) glDeleteBuffers(1, &g.solidIbo);
    if (g.ringVao)  glDeleteVertexArrays(1, &g.ringVao);
    if (g.ringVbo)  glDeleteBuffers(1, &g.ringVbo);
    if (g.ringIbo)  glDeleteBuffers(1, &g.ringIbo);
    g = ClipmapGrid{};
}

glm::vec2 clipmapLevelCenter(const glm::vec2& cameraXZ, float levelSpacing) {
    // Snap camera XZ to a 2*spacing lattice so that level edges always align
    // across frames (prevents per-vertex crawling). Adjacent levels may then
    // mis-snap by up to `spacing` at shared edges; that is the classic clipmap
    // seam and can be fixed later with an L-shaped trim.
    const float step = 2.f * levelSpacing;
    return glm::vec2(std::floor(cameraXZ.x / step) * step,
                     std::floor(cameraXZ.y / step) * step);
}

glm::vec2 clipmapSharedCenter(const glm::vec2& cameraXZ, const ClipmapGrid& g) {
    // Quantize to the outermost level's snap step. Because step_{l} = 2*d_l and
    // d_l = d_0 * 2^l, the outermost step is the biggest; any multiple of it is
    // also a multiple of every inner step (step_{l+1} = 2 * step_l). So the
    // resulting center is a valid snap point for ALL levels simultaneously, which
    // is exactly what we need to make adjacent ring edges coincide.
    const int   topLevel  = (g.L > 0) ? (g.L - 1) : 0;
    const float outerSpacing = g.baseSpacing * std::exp2(static_cast<float>(topLevel));
    const float step = 2.f * outerSpacing;
    return glm::vec2(std::floor(cameraXZ.x / step) * step,
                     std::floor(cameraXZ.y / step) * step);
}

float clipmapLevelSpacing(const ClipmapGrid& g, int level) {
    return g.baseSpacing * std::exp2(static_cast<float>(level));
}

void clipmapGridDrawLevel(const ClipmapGrid& g, int level) {
    const bool solid = (level == 0);
    glBindVertexArray(solid ? g.solidVao : g.ringVao);
    const GLsizei count = solid ? g.solidCount : g.ringCount;
    glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

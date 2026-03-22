#include "jw_transport.h"

#include "shallow_water_solver.h"
#include "wavedecomposer.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr float kDryEps = 1e-4f;

float faceH_X(const Grid& g, int i, int j) {
    const int il = std::max(0, i - 1);
    const int ir = std::min(g.NX - 1, i);
    return (g.QX(i, j) >= 0.f) ? g.H(il, j) : g.H(ir, j);
}

float faceH_Y(const Grid& g, int i, int j) {
    const int jd = std::max(0, j - 1);
    const int ju = std::min(g.NY - 1, j);
    return (g.QY(i, j) >= 0.f) ? g.H(i, jd) : g.H(i, ju);
}

float uFaceX(const Grid& g, int i, int j) {
    const float hf = faceH_X(g, i, j);
    return (hf < kDryEps) ? 0.f : g.QX(i, j) / hf;
}

float uFaceY(const Grid& g, int i, int j) {
    const float hf = faceH_Y(g, i, j);
    return (hf < kDryEps) ? 0.f : g.QY(i, j) / hf;
}

float uXmid(const Grid& a, const Grid& b, int i, int j) {
    return 0.5f * (uFaceX(a, i, j) + uFaceX(b, i, j));
}

float uYmid(const Grid& a, const Grid& b, int i, int j) {
    return 0.5f * (uFaceY(a, i, j) + uFaceY(b, i, j));
}

float divUmidCell(const Grid& a, const Grid& b, int i, int j, float dx) {
    const float uxR = uXmid(a, b, i + 1, j);
    const float uxL = uXmid(a, b, i, j);
    const float uyT = uYmid(a, b, i, j + 1);
    const float uyB = uYmid(a, b, i, j);
    return (uxR - uxL + uyT - uyB) / dx;
}

float divUCellBar1(const Grid& g, int i, int j, float dx) {
    const float uxR = uFaceX(g, i + 1, j);
    const float uxL = uFaceX(g, i, j);
    const float uyT = uFaceY(g, i, j + 1);
    const float uyB = uFaceY(g, i, j);
    return (uxR - uxL + uyT - uyB) / dx;
}

inline float GofDiv(float div, float gamma) {
    return std::min(-div, -gamma * div);
}

inline float clampExpArg(float x) {
    return std::clamp(x, -30.f, 30.f);
}

float sampleQx(const std::vector<float>& q, int nx, int ny, float fi, float fj) {
    fi = std::clamp(fi, 0.f, static_cast<float>(nx));
    fj = std::clamp(fj, 0.f, static_cast<float>(ny - 1) - 1e-5f);
    int i0 = static_cast<int>(std::floor(fi));
    int j0 = static_cast<int>(std::floor(fj));
    const float tx = fi - static_cast<float>(i0);
    const float ty = fj - static_cast<float>(j0);
    i0 = std::clamp(i0, 0, nx - 1);
    j0 = std::clamp(j0, 0, ny - 1);
    const int i1 = std::min(i0 + 1, nx);
    const int j1 = std::min(j0 + 1, ny - 1);
    auto at = [&](int ii, int jj) { return q[ii + jj * (nx + 1)]; };
    const float q00 = at(i0, j0);
    const float q10 = at(i1, j0);
    const float q01 = at(i0, j1);
    const float q11 = at(i1, j1);
    const float q0  = q00 + tx * (q10 - q00);
    const float q1  = q01 + tx * (q11 - q01);
    return q0 + ty * (q1 - q0);
}

float sampleQy(const std::vector<float>& q, int nx, int ny, float fi, float fj) {
    fi = std::clamp(fi, 0.f, static_cast<float>(nx - 1) - 1e-5f);
    fj = std::clamp(fj, 0.f, static_cast<float>(ny));
    int i0 = static_cast<int>(std::floor(fi));
    int j0 = static_cast<int>(std::floor(fj));
    const float tx = fi - static_cast<float>(i0);
    const float ty = fj - static_cast<float>(j0);
    i0 = std::clamp(i0, 0, nx - 2);
    j0 = std::clamp(j0, 0, ny - 1);
    const int i1 = std::min(i0 + 1, nx - 1);
    const int j1 = std::min(j0 + 1, ny);
    auto at = [&](int ii, int jj) { return q[ii + jj * nx]; };
    const float q00 = at(i0, j0);
    const float q10 = at(i1, j0);
    const float q01 = at(i0, j1);
    const float q11 = at(i1, j1);
    const float q0  = q00 + tx * (q10 - q00);
    const float q1  = q01 + tx * (q11 - q01);
    return q0 + ty * (q1 - q0);
}

float sampleHcell(const std::vector<float>& h, int nx, int ny, float fi, float fj) {
    fi = std::clamp(fi, 0.f, static_cast<float>(nx - 1) - 1e-5f);
    fj = std::clamp(fj, 0.f, static_cast<float>(ny - 1) - 1e-5f);
    int i0 = static_cast<int>(std::floor(fi));
    int j0 = static_cast<int>(std::floor(fj));
    const float tx = fi - static_cast<float>(i0);
    const float ty = fj - static_cast<float>(j0);
    i0 = std::clamp(i0, 0, nx - 2);
    j0 = std::clamp(j0, 0, ny - 2);
    auto at = [&](int ii, int jj) { return h[ii + jj * nx]; };
    const float q00 = at(i0, j0);
    const float q10 = at(i0 + 1, j0);
    const float q01 = at(i0, j0 + 1);
    const float q11 = at(i0 + 1, j0 + 1);
    const float q0  = q00 + tx * (q10 - q00);
    const float q1  = q01 + tx * (q11 - q01);
    return q0 + ty * (q1 - q0);
}

float uzAtQxFace(const Grid& a, const Grid& b, int i, int j, int nx, int ny) {
    const auto uyM = [&](int ii, int jj) {
        ii = std::clamp(ii, 0, nx - 1);
        jj = std::clamp(jj, 0, ny);
        return uYmid(a, b, ii, jj);
    };
    const int im = std::max(0, i - 1);
    const int ip = std::min(nx - 1, i);
    const int j0 = std::clamp(j, 0, ny - 1);
    const int j1 = std::min(j + 1, ny);
    return 0.25f * (uyM(im, j0) + uyM(ip, j0) + uyM(im, j1) + uyM(ip, j1));
}

float uxAtQyFace(const Grid& a, const Grid& b, int i, int j, int nx, int ny) {
    const auto uxM = [&](int ii, int jj) {
        ii = std::clamp(ii, 0, nx);
        jj = std::clamp(jj, 0, ny - 1);
        return uXmid(a, b, ii, jj);
    };
    const int jm = std::max(0, j - 1);
    const int jp = std::min(ny - 1, j);
    const int i0 = std::clamp(i, 0, nx - 1);
    const int i1 = std::min(i + 1, nx);
    return 0.25f * (uxM(i0, jm) + uxM(i1, jm) + uxM(i0, jp) + uxM(i1, jp));
}

} // namespace

void jwTransportSurface(WaveDecomposition& dec,
                       const Grid&       gBar0,
                       const Grid&       gBar1,
                       float               halfW,
                       float               halfD,
                       float               dt,
                       float               gamma) {
    const int   nx = gBar0.NX;
    const int   ny = gBar0.NY;
    const float dx = gBar0.dx;

    // ----- Alg. 3: tilde qx, qy -----
    std::vector<float> qxs = dec.qx_tilde;
    std::vector<float> qys = dec.qy_tilde;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            float divC;
            if (i <= 0)
                divC = divUmidCell(gBar0, gBar1, 0, j, dx);
            else if (i >= nx)
                divC = divUmidCell(gBar0, gBar1, nx - 1, j, dx);
            else
                divC = 0.5f * (divUmidCell(gBar0, gBar1, i - 1, j, dx) + divUmidCell(gBar0, gBar1, i, j, dx));
            const float G  = GofDiv(divC, gamma);
            const int   ix = i + j * (nx + 1);
            dec.qx_tilde[ix] = qxs[ix] * std::exp(clampExpArg(G * dt));
        }
    }

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            float divC;
            if (j <= 0)
                divC = divUmidCell(gBar0, gBar1, i, 0, dx);
            else if (j >= ny)
                divC = divUmidCell(gBar0, gBar1, i, ny - 1, dx);
            else
                divC = 0.5f * (divUmidCell(gBar0, gBar1, i, j - 1, dx) + divUmidCell(gBar0, gBar1, i, j, dx));
            const float G  = GofDiv(divC, gamma);
            const int   iy = i + j * nx;
            dec.qy_tilde[iy] = qys[iy] * std::exp(clampExpArg(G * dt));
        }
    }

    qxs = dec.qx_tilde;
    qys = dec.qy_tilde;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            const float x  = static_cast<float>(i) * dx - halfW;
            const float z  = (static_cast<float>(j) + 0.5f) * dx - halfD;
            const float ux = uXmid(gBar0, gBar1, i, j);
            const float uz = uzAtQxFace(gBar0, gBar1, i, j, nx, ny);
            const float xd = x - ux * dt;
            const float zd = z - uz * dt;
            const float fi = (xd + halfW) / dx;
            const float fj = (zd + halfD) / dx - 0.5f;
            dec.qx_tilde[i + j * (nx + 1)] = sampleQx(qxs, nx, ny, fi, fj);
        }
    }

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const float x  = (static_cast<float>(i) + 0.5f) * dx - halfW;
            const float z  = static_cast<float>(j) * dx - halfD;
            const float uz = uYmid(gBar0, gBar1, i, j);
            const float ux = uxAtQyFace(gBar0, gBar1, i, j, nx, ny);
            const float xd = x - ux * dt;
            const float zd = z - uz * dt;
            const float fi = (xd + halfW) / dx - 0.5f;
            const float fj = (zd + halfD) / dx;
            dec.qy_tilde[i + j * nx] = sampleQy(qys, nx, ny, fi, fj);
        }
    }

    // ----- Alg. 4: tilde h（仅用 bar^{t+Δt} 算 ∇·ū）-----
    std::vector<float> hs = dec.h_tilde;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const float divC = divUCellBar1(gBar1, i, j, dx);
            const float G    = GofDiv(divC, gamma);
            dec.h_tilde[i + j * nx] = hs[i + j * nx] * std::exp(clampExpArg(G * dt));
        }
    }
    hs = dec.h_tilde;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const float x = (static_cast<float>(i) + 0.5f) * dx - halfW;
            const float z = (static_cast<float>(j) + 0.5f) * dx - halfD;
            const float ux =
                0.5f * (uFaceX(gBar1, i, j) + uFaceX(gBar1, i + 1, j));
            const float uz =
                0.5f * (uFaceY(gBar1, i, j) + uFaceY(gBar1, i, j + 1));
            const float xd = x - ux * dt;
            const float zd = z - uz * dt;
            const float fi = (xd + halfW) / dx - 0.5f;
            const float fj = (zd + halfD) / dx - 0.5f;
            dec.h_tilde[i + j * nx] = sampleHcell(hs, nx, ny, fi, fj);
        }
    }
}

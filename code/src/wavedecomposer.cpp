#include "wavedecomposer.h"

#include "shallow_water_solver.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr float kEps = 1e-8f;

inline float harmonicMean(float a, float b) {
    return 2.f * a * b / (a + b + kEps);
}

void computeAlphaFromH(const Grid& g, float d_grad_penalty, std::vector<float>& alpha) {
    alpha.resize(g.NX * g.NY);
    const float dx = g.dx;
    for (int j = 0; j < g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i) {
            int im = std::max(0, i - 1);
            int ip = std::min(g.NX - 1, i + 1);
            int jm = std::max(0, j - 1);
            int jp = std::min(g.NY - 1, j + 1);
            float denom_x = float(ip - im) * dx;
            float denom_y = float(jp - jm) * dx;
            float grad_h_x = (denom_x > 0.f) ? (g.H(ip, j) - g.H(im, j)) / denom_x : 0.f;
            float grad_h_y = (denom_y > 0.f) ? (g.H(i, jp) - g.H(i, jm)) / denom_y : 0.f;
            float grad_h_sq = grad_h_x * grad_h_x + grad_h_y * grad_h_y;
            float h = g.H(i, j);
            alpha[i + j * g.NX] = (h * h / 64.f) * std::exp(-d_grad_penalty * grad_h_sq);
        }
    }
}

// Neumann (zero flux) at domain boundary. u and α same layout (nx * ny).
void diffuseScalar2d(const std::vector<float>& u_init,
                     const std::vector<float>& alpha,
                     int nx, int ny, float dx, int n_iter,
                     std::vector<float>& u_out) {
    std::vector<float> u = u_init;
    std::vector<float> u_new(nx * ny);
    const float dx2 = dx * dx;

    for (int iter = 0; iter < n_iter; ++iter) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int idx = i + j * nx;
                float aij = alpha[idx];

                int ip = std::min(i + 1, nx - 1);
                int im = std::max(i - 1, 0);
                int jp = std::min(j + 1, ny - 1);
                int jm = std::max(j - 1, 0);

                float aR = (i + 1 < nx) ? harmonicMean(aij, alpha[ip + j * nx]) : 0.f;
                float aL = (i - 1 >= 0) ? harmonicMean(alpha[im + j * nx], aij) : 0.f;
                float aT = (j + 1 < ny) ? harmonicMean(aij, alpha[i + jp * nx]) : 0.f;
                float aB = (j - 1 >= 0) ? harmonicMean(alpha[i + jm * nx], aij) : 0.f;

                float uC = u[idx];
                float uR = (i + 1 < nx) ? u[ip + j * nx] : uC;
                float uL = (i - 1 >= 0) ? u[im + j * nx] : uC;
                float uT = (j + 1 < ny) ? u[i + jp * nx] : uC;
                float uB = (j - 1 >= 0) ? u[i + jm * nx] : uC;

                float a_max = std::max(std::max(aR, aL), std::max(aT, aB));
                a_max = std::max(a_max, kEps);
                float dt_loc = std::min(0.25f, dx2 / (4.f * a_max + kEps));

                float lap = aR * (uR - uC) - aL * (uC - uL) + aT * (uT - uC) - aB * (uC - uB);
                lap /= dx2;
                u_new[idx] = uC + dt_loc * lap;
            }
        }
        u.swap(u_new);
    }
    u_out = std::move(u);
}

void alphaOnQXFaces(const Grid& g, const std::vector<float>& alpha_cell,
                    std::vector<float>& alpha_qx) {
    alpha_qx.resize((g.NX + 1) * g.NY);
    for (int j = 0; j < g.NY; ++j) {
        for (int i = 0; i <= g.NX; ++i) {
            float a;
            if (i == 0)
                a = alpha_cell[0 + j * g.NX];
            else if (i == g.NX)
                a = alpha_cell[(g.NX - 1) + j * g.NX];
            else
                a = harmonicMean(alpha_cell[(i - 1) + j * g.NX], alpha_cell[i + j * g.NX]);
            alpha_qx[i + j * (g.NX + 1)] = a;
        }
    }
}

void alphaOnQYFaces(const Grid& g, const std::vector<float>& alpha_cell,
                    std::vector<float>& alpha_qy) {
    alpha_qy.resize(g.NX * (g.NY + 1));
    for (int j = 0; j <= g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i) {
            float a;
            if (j == 0)
                a = alpha_cell[i + 0 * g.NX];
            else if (j == g.NY)
                a = alpha_cell[i + (g.NY - 1) * g.NX];
            else
                a = harmonicMean(alpha_cell[i + (j - 1) * g.NX], alpha_cell[i + j * g.NX]);
            alpha_qy[i + j * g.NX] = a;
        }
    }
}

} // namespace

void waveDecompose(const Grid& g, float d_grad_penalty, WaveDecomposition& out) {
    waveDecompose(g, d_grad_penalty, 128, out);
}

void waveDecompose(const Grid& g, float d_grad_penalty, int n_diffusion_iters, WaveDecomposition& out) {
    const int nIter = std::max(1, n_diffusion_iters);
    std::vector<float> alpha_cell;
    computeAlphaFromH(g, d_grad_penalty, alpha_cell);

    std::vector<float> H(g.NX * g.NY);
    for (int j = 0; j < g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i) {
            H[i + j * g.NX] = g.H(i, j) + g.B(i, j);
        }
    }

    std::vector<float> H_bar;
    diffuseScalar2d(H, alpha_cell, g.NX, g.NY, g.dx, nIter, H_bar);

    out.h_bar.resize(g.NX * g.NY);
    out.h_tilde.resize(g.NX * g.NY);
    for (int j = 0; j < g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i) {
            int ix = i + j * g.NX;
            out.h_bar[ix] = std::max(0.f, H_bar[ix] - g.B(i, j));
            out.h_tilde[ix] = g.H(i, j) - out.h_bar[ix];
        }
    }

    std::vector<float> alpha_qx, alpha_qy;
    alphaOnQXFaces(g, alpha_cell, alpha_qx);
    alphaOnQYFaces(g, alpha_cell, alpha_qy);

    diffuseScalar2d(g.qx, alpha_qx, g.NX + 1, g.NY, g.dx, nIter, out.qx_bar);
    out.qx_tilde.resize(g.qx.size());
    for (size_t k = 0; k < g.qx.size(); ++k)
        out.qx_tilde[k] = g.qx[k] - out.qx_bar[k];

    diffuseScalar2d(g.qy, alpha_qy, g.NX, g.NY + 1, g.dx, nIter, out.qy_bar);
    out.qy_tilde.resize(g.qy.size());
    for (size_t k = 0; k < g.qy.size(); ++k)
        out.qy_tilde[k] = g.qy[k] - out.qy_bar[k];
}

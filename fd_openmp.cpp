#include <vector>
#include <cmath>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

std::vector<float> compute_acc_fd(const std::vector<float>& p, int N, float L, float sigma, float epsilon) {
    std::vector<float> acc(N * 3, 0.0f);
    float sigma2 = sigma * sigma;

    std::vector<std::vector<float>> shifts;
    float n_values[3] = { -L, 0.0f, L };
    for (float nx : n_values)
        for (float ny : n_values)
            for (float nz : n_values)
                shifts.push_back({nx, ny, nz});

    int num_threads = 4;
    omp_set_num_threads(num_threads);
    int M = static_cast<int>(std::sqrt(num_threads));
    if (M < 1) M = 1;
    int block_size = (N + M - 1) / M;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int row_block = 0; row_block < M; ++row_block) {
        for (int col_block = 0; col_block < M; ++col_block) {
            int i_start = row_block * block_size;
            int i_end = std::min(i_start + block_size, N);
            int j_start = col_block * block_size;
            int j_end = std::min(j_start + block_size, N);

            for (int i = i_start; i < i_end; ++i) {
                for (int j = j_start; j < j_end; ++j) {
                    if (i == j) continue;

                    float dx_init = p[i * 3 + 0] - p[j * 3 + 0];
                    float dy_init = p[i * 3 + 1] - p[j * 3 + 1];
                    float dz_init = p[i * 3 + 2] - p[j * 3 + 2];
                    
                    float min_dist_sq = -1.0f;
                    float best_dx = 0, best_dy = 0, best_dz = 0;

                    for (const auto& s : shifts) {
                        float dx = dx_init + s[0];
                        float dy = dy_init + s[1];
                        float dz = dz_init + s[2];
                        float d2 = dx * dx + dy * dy + dz * dz;

                        if (min_dist_sq < 0 || d2 < min_dist_sq) {
                            min_dist_sq = d2;
                            best_dx = dx; best_dy = dy; best_dz = dz;
                        }
                    }
                    
                    if (min_dist_sq < 0.001f) min_dist_sq = 0.001f; // Защита от деления на 0

                    float r2_inv = 1.0f / min_dist_sq;
                    float r6_inv = (sigma2 * r2_inv) * (sigma2 * r2_inv) * (sigma2 * r2_inv);
                    float r12_inv = r6_inv * r6_inv;
                    float force_scalar = 24.0f * epsilon * (2 * r12_inv - r6_inv) * r2_inv;

#pragma omp atomic
                    acc[i * 3 + 0] += force_scalar * best_dx;
#pragma omp atomic
                    acc[i * 3 + 1] += force_scalar * best_dy;
#pragma omp atomic
                    acc[i * 3 + 2] += force_scalar * best_dz;
                }
            }
        }
    }
    return acc;
}

PYBIND11_MODULE(fd_openmp, m) {
    m.def("compute_acc_fd", &compute_acc_fd, "Compute accelerations using Force Decomposition");
}
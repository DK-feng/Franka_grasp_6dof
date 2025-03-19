#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel
__global__ void fps_kernel(int b, int n, int m, const float *points, float *temp, int *idxs) {
    int bs_idx = blockIdx.x;
    points += bs_idx * n * 3;
    temp += bs_idx * n;
    idxs += bs_idx * m;

    int first_idx = 0;
    idxs[0] = first_idx;

    for (int j = 1; j < m; j++) {
        int best_idx = 0;
        float best_dist = -1;

        float x1 = points[idxs[j - 1] * 3 + 0];
        float y1 = points[idxs[j - 1] * 3 + 1];
        float z1 = points[idxs[j - 1] * 3 + 2];

        for (int k = 0; k < n; k++) {
            float x2 = points[k * 3 + 0];
            float y2 = points[k * 3 + 1];
            float z2 = points[k * 3 + 2];

            float dx = x2 - x1;
            float dy = y2 - y1;
            float dz = z2 - z1;

            float dist2 = dx * dx + dy * dy + dz * dz;
            float d = min(dist2, temp[k]);
            temp[k] = d;

            if (d > best_dist) {
                best_dist = d;
                best_idx = k;
            }
        }

        idxs[j] = best_idx;
    }
}

// launcher 函数，供 cpp 文件调用
void fps_launcher(int b, int n, int m, const float *points, float *temp, int *idxs) {
    fps_kernel<<<b, 1>>>(b, n, m, points, temp, idxs);
}

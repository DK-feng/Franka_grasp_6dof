#include <torch/extension.h>
#include <vector>


void fps_launcher(int b, int n, int m, const float *points, float *temp, int *idxs);

torch::Tensor farthest_point_sampling(torch::Tensor points, int m) {
    const auto b = points.size(0);
    const auto n = points.size(1);

    auto idxs = torch::zeros({b, m}, torch::dtype(torch::kInt32).device(points.device()));
    auto temp = torch::full({b, n}, 1e10, torch::dtype(torch::kFloat32).device(points.device()));

    fps_launcher(b, n, m,
        points.data_ptr<float>(),
        temp.data_ptr<float>(),
        idxs.data_ptr<int>());

    return idxs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sampling", &farthest_point_sampling, "Farthest Point Sampling (CUDA)");
}

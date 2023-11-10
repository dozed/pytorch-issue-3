#include <iostream>
#include <torch/torch.h>
#include <torch/linalg.h>
#include <torch/nn/modules/conv.h>

// source: https://github.com/pytorch/pytorch/blob/main/c10/core/MemoryFormat.h#L128
template <typename T>
inline bool is_channels_last_strides_2d_s4_1(
        const at::ArrayRef<T> sizes,
        const at::ArrayRef<T> strides) {
    T min = 0;
    // special case for trivial C dimension. default to NCHW
    if (strides[1] == 0) {
        return false;
    }
    // loop strides indices
    for (auto& d : {1, 3, 2, 0}) {
        if (sizes[d] == 0) {
            return false;
        }
        if (strides[d] < min) {
            return false;
        }
        // Fallback to NCHW as default layout for ambiguous cases
        // This is the flaw of implicit memory_format from strides.
        // N111 tensor with identical strides for size 1 dimension;
        // Two cases could lead us here:
        // a. N111 contiguous Tensor ([N,1,1,1]@[1,1,1,1])
        // b. N11W contiguous Tensor sliced on the W-dimension.
        // ([N,1,1,1]@[W,W,W,W])
        if (d == 0 && min == strides[1]) {
            return false;
        }
        // This is necessary to:
        // 1. distinguish the memory_format of N1H1;
        //     [H, 1, 1, 1] channels_last stride
        //     [H, H, 1, 1] contiguous stride
        // 2. permutation of 1C1W:
        //     [1, C, 1, H]@[HC, H, H, 1] transpose(1, 3)
        //     [1, H, 1, C]@[HC, 1, H, H] shouldn't be identified as channels_last
        min = strides[d];
        if (sizes[d] > 1) {
            min *= sizes[d];
        }
    }
    return true;
}

void print_tensor_info(const std::string& label, const at::Tensor &x) {
    c10::TensorImpl* tensor_impl = x.unsafeGetTensorImpl();
    c10::Storage storage = tensor_impl->unsafe_storage();

    std::cout << "---[ tensor info : " << label << " ]---" << std::endl;
    std::cout << "dtype : " << x.dtype() << std::endl;
    std::cout << "layout : " << x.layout() << std::endl;
    std::cout << "nbytes : " << x.nbytes() << std::endl;
    std::cout << "numel : " << x.numel() << std::endl;
    std::cout << "sizes : " << x.sizes() << std::endl;
    std::cout << "strides : " << x.strides() << std::endl;
    std::cout << "is_contiguous(Contiguous) : " << x.is_contiguous(at::MemoryFormat::Contiguous) << std::endl;
    std::cout << "is_contiguous(ChannelsLast) : " << x.is_contiguous(at::MemoryFormat::ChannelsLast) << std::endl;
    std::cout << "is_strides_like_channels_last : " << tensor_impl->is_strides_like(at::MemoryFormat::ChannelsLast) << std::endl;
    std::cout << "is_channels_last_strides_2d_s4_1 : " << is_channels_last_strides_2d_s4_1(x.sizes(), x.strides()) << std::endl;
    std::cout << "suggested_memory_format : " << x.suggest_memory_format() << std::endl;
    std::cout << "tensor_impl : " << tensor_impl << std::endl;
    std::cout << "data_ptr : " << (&storage.data_ptr()) << std::endl;
    std::cout << std::endl;
}

void test_conv(const at::Tensor& x, const at::Tensor& y) {
    torch::manual_seed(42);

    torch::nn::Conv2d conv = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(3, 2, 3).stride(1).padding(1).bias(true)
    );

    at::Tensor input1 = x.to(torch::kFloat32);
    at::Tensor input2 = y.to(torch::kFloat32);

    at::Tensor res1 = conv->forward(input1);
    at::Tensor res2 = conv->forward(input2);

    print_tensor_info("res1", res1);
    print_tensor_info("res2", res2);
    torch::Tensor norm = torch::sqrt(torch::sum(torch::pow(res1 - res2, 2)));
    std::cout << "norm(res1, res2) : " << norm << std::endl;  // 1.19209e-07
    std::cout << "torch::allclose(res1, res2) : " << torch::allclose(res1, res2) << std::endl;
    std::cout << "torch::equal(res1, res2)    : " << torch::equal(res1, res2) << std::endl;
    std::cout << "torch::eq(res1, res2)    : " << torch::eq(res1, res2) << std::endl;
}

int main() {
    // 1) create (1, 2, 3) ~ HWC Tensor
    at::Tensor orig =
        torch::tensor({
            {
                {1, 2, 3},
                {4, 5, 6},
            },
        }, torch::dtype(torch::kUInt8));

    // 2) permute tensor
    at::Tensor permuted = orig.permute({2, 0, 1});
    print_tensor_info("permuted", permuted);

    // 3) unsqueeze tensor and (optionally) fix strides
    at::Tensor permuted_unsqueezed_pre = permuted.unsqueeze(0);
    print_tensor_info("permuted_unsqueezed_pre", permuted_unsqueezed_pre);

    // fix strides as computed by `unsqueeze`
    at::IntArrayRef strides_fixed = at::IntArrayRef{6, 1, 6, 3};
    at::Tensor permuted_unsqueezed = permuted_unsqueezed_pre.as_strided(permuted_unsqueezed_pre.sizes(), strides_fixed);

    // or: take strides as computed by `unsqueeze`
    // at::Tensor permuted_unsqueezed = permuted_unsqueezed_pre;

    print_tensor_info("permuted_unsqueezed", permuted_unsqueezed);

    // 4) add zeros
    at::Tensor zeros = torch::zeros_like(permuted_unsqueezed);
    // at::Tensor zeros = torch::zeros(permuted_unsqueezed.sizes(), permuted_unsqueezed.dtype());
    at::Tensor added_zeros = permuted_unsqueezed.add(zeros);
    print_tensor_info("added_zeros", added_zeros);

    // 5) compute forward passes for the two tensors through the same Conv2d layer, the results should be equal
    test_conv(permuted_unsqueezed, added_zeros);

}

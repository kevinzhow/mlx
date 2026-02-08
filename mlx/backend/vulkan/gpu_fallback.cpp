// Copyright © 2026 MLX Vulkan Backend

#include "mlx/allocator.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/gpu/slicing.h"

#include <numeric>
#include <stdexcept>

namespace mlx::core {

namespace {

inline Stream cpu_fallback_stream() {
  return default_stream(Device::cpu);
}

inline void synchronize_if_gpu_stream(const Stream& s) {
  if (s.device == Device::gpu) {
    gpu::synchronize(s);
  }
}

template <typename T>
int64_t dynamic_offset_impl(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes) {
  auto* ptr = indices.data<T>();
  int64_t offset = 0;
  for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
    int ax = axes[i] < 0 ? static_cast<int>(strides.size()) + axes[i] : axes[i];
    if (ax < 0 || ax >= static_cast<int>(strides.size())) {
      throw std::runtime_error("[compute_dynamic_offset] Invalid axis.");
    }
    offset += static_cast<int64_t>(ptr[i]) * strides[ax];
  }
  return offset;
}

} // namespace

void copy_gpu(
    const array& src,
    array& out,
    CopyType ctype,
    const Stream& s) {
  synchronize_if_gpu_stream(s);
  auto cpu_s = cpu_fallback_stream();
  copy_cpu(src, out, ctype, cpu_s);
  synchronize(cpu_s);
}

void copy_gpu_inplace(
    const array& src,
    array& out,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_i_offset,
    std::optional<array> dynamic_o_offset) {
  synchronize_if_gpu_stream(s);
  auto cpu_s = cpu_fallback_stream();
  copy_cpu_inplace(
      src,
      out,
      data_shape,
      i_strides,
      o_strides,
      i_offset,
      o_offset,
      ctype,
      cpu_s,
      dynamic_i_offset,
      dynamic_o_offset);
  synchronize(cpu_s);
}

void fill_gpu(const array& val, array& out, const Stream& s) {
  copy_gpu(val, out, CopyType::Scalar, s);
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu(in, out, CopyType::General, s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  std::vector<int> sizes;
  sizes.reserve(inputs.size() + 1);
  sizes.push_back(0);
  for (auto& in : inputs) {
    sizes.push_back(in.shape(axis));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  out.set_data(allocator::malloc(out.nbytes()));

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;

  for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    copy_gpu_inplace(inputs[i], out_slice, CopyType::GeneralGeneral, s);
  }
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  synchronize_if_gpu_stream(s);

  array offset({1}, int64, nullptr, {});
  bool donate = indices.is_donatable() &&
      (indices.data_size() * indices.itemsize()) >= offset.itemsize();
  if (donate) {
    offset.copy_shared_buffer(indices);
  } else {
    offset.set_data(allocator::malloc(offset.itemsize()));
  }

  int64_t out_offset = 0;
  switch (indices.dtype()) {
    case int8:
      out_offset = dynamic_offset_impl<int8_t>(indices, strides, axes);
      break;
    case uint8:
      out_offset = dynamic_offset_impl<uint8_t>(indices, strides, axes);
      break;
    case int16:
      out_offset = dynamic_offset_impl<int16_t>(indices, strides, axes);
      break;
    case uint16:
      out_offset = dynamic_offset_impl<uint16_t>(indices, strides, axes);
      break;
    case int32:
      out_offset = dynamic_offset_impl<int32_t>(indices, strides, axes);
      break;
    case uint32:
      out_offset = dynamic_offset_impl<uint32_t>(indices, strides, axes);
      break;
    case int64:
      out_offset = dynamic_offset_impl<int64_t>(indices, strides, axes);
      break;
    case uint64:
      out_offset = dynamic_offset_impl<uint64_t>(indices, strides, axes);
      break;
    default:
      throw std::runtime_error(
          "[compute_dynamic_offset] Invalid index dtype for dynamic offset.");
  }

  offset.data<int64_t>()[0] = out_offset;
  return offset;
}

} // namespace mlx::core

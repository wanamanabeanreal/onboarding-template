#pragma once

#include <cstddef>
#include <vector>
#include <thread>
#include <cstring>
#include <experimental/simd> //17?
// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.

namespace stdx = std::experimental;

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;

  //flat
  std::vector<double> data_;

public:
  Grid(std::size_t rows, std::size_t cols):
  rows_(rows), cols_(cols), data_(rows*cols,0.0) {}

  Grid(const Grid&) = delete;
  Grid& operator=(const Grid&) = delete;

  //move over copy
  Grid(Grid&&other) noexcept = default;
  Grid& operator=(Grid&& other) noexcept = default;

  ~Grid() = default;


  double& operator()(std::size_t i, std::size_t j) {
    return data_[i*cols_ +j];
  }
  double  operator()(std::size_t i, std::size_t j) const {
    return data_[i*cols_ +j];
  }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  double* data() { return &data_[0]; }
  const double* data() const { return &data_[0]; }
};

void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  if (rows <= 2 || cols <= 2) {
    std::memcpy(new_grid.data(), old_grid.data(), rows * cols * sizeof(double));
    return;
  }

  const double* __restrict__ src = old_grid.data();
  double* __restrict__ dst = new_grid.data();

  // Copy boundaries (top and bottom)
  std::memcpy(dst, src, cols * sizeof(double));
  std::memcpy(dst + (rows - 1) * cols, src + (rows - 1) * cols, cols * sizeof(double));

  // Parallelize across rows
#pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    const std::size_t row_start = i * cols;
    const std::size_t top_start = (i - 1) * cols;
    const std::size_t bot_start = (i + 1) * cols;

    // left
    dst[row_start] = src[row_start];

    // restrict the current row's pointers
    const double* __restrict__ r_top  = src + top_start;
    const double* __restrict__ r_curr = src + row_start;
    const double* __restrict__ r_bot  = src + bot_start;
    double* __restrict__       r_dst  = dst + row_start;

    // trying omp's simd
#pragma omp simd
    for (std::size_t j = 1; j < cols - 1; ++j) {
      r_dst[j] = 0.5 * r_curr[j] +
                 0.125 * (r_top[j] + r_bot[j] + r_curr[j - 1] + r_curr[j + 1]);
    }

    // right
    dst[row_start + cols - 1] = src[row_start + cols - 1];
  }
}
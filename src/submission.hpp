#pragma once

#include <cstddef>
#include <vector>
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
  std::size_t stride_;

  static constexpr std::size_t ALIGNMENT = 64; // assumption, to change as needed
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
  const double* data() const { return &data_[0]; }//ptr to start
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();
  //edge case for a grid with only boundaries
  if (rows <= 2 || cols <= 2) {
    std::memcpy(new_grid.data(), old_grid.data(), rows*cols*sizeof(double));
    return;
  }
  //unique location
  const double* __restrict__ src = old_grid.data();
  double* __restrict__ dst = new_grid.data();

  //copy boundaries(top/bottom)
  std::memcpy(dst,src, cols*sizeof(double)); //index 0
  std::memcpy(dst + (rows-1) * cols, src + (rows-1)*cols,cols*sizeof(double));

  //detect no. lanes and form the  vector object
  using V = stdx::native_simd<double>;
  constexpr std::size_t LANES = V::size();
  const V v_half(0.5); //current cell
  const V v_eighth(0.125); //surrounding four

  //#pragma omp parallel for schedule(static) //cmake 18-21
    for (size_t i = 1; i < rows - 1; ++i) {
      //calculate where the row starts
      std::size_t row_start = i * cols;
      std::size_t top_start = (i-1) * cols;
      std::size_t bot_start = (i+1) * cols;


      dst[row_start] = src[row_start]; //copy the left boundary element
      std::size_t j = 1;
      //loop thru cols
      for (; j + LANES <= cols-1; j += LANES) {
        V c;      c.copy_from(src + row_start + j, stdx::element_aligned);
        V top;    top.copy_from(src + top_start + j, stdx::element_aligned);
        V bot;    bot.copy_from(src + bot_start + j, stdx::element_aligned);
        V left;   left.copy_from(src + row_start + j - 1, stdx::element_aligned); //around the cell
        V right;  right.copy_from(src + row_start + j + 1, stdx::element_aligned);

        //0.5 * original + 0.125(top + bot + left + right)
        V res = v_half * c + v_eighth * (top + bot + left + right);
        res.copy_to(dst + row_start + j, stdx::element_aligned);
      }
      //Cleaner upper loop
      for (; j < cols-1; ++j) {
        dst[row_start + j] = 0.5 * src[row_start + j] +
          0.125 * (src[top_start +j] + src[bot_start + j] + src[row_start + j-1] + src[row_start + j +1]);
      }
      //rightmost
      dst[row_start + cols -1] = src[row_start + cols-1];
    }
}

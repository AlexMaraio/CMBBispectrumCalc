// Created by Alessandro Maraio on 02/04/2020.
// Copyright (c) 2020 University of Sussex.
// contributor: Alessandro Maraio <am963@sussex.ac.uk>


/*
This file is dedicated to performing the calculation of the non-Gaussian corrections to the alm coefficients
 of the CMB bispectrum, so that way we can perform accurate map-making.
We use C++ for it's greater speed over Python, as already this calculation is taking too long here.

Note that our working precision here is float, as this reduces the memory usage for storing the bispectrum
 values and Wigner-3j symbols in half. This is needed as this library is a fairly high-memory usage program.
*/


#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <complex>
#include <chrono>

#include "csv.h"
#include "memo.hpp"
#include "omp.h"


// Declare the function to compute the Wigner-3j symbols, which will link against the Fortran code
extern "C" void getthreejs(double *data, int *ell2, int *ell3, int *m2, int *m3);


// Define custom vector types, which makes initialising our 3D array easier
// We use templates so that way they apply to regular numbers & complex number types too.

// 1D array
template<typename T>
using data1D = std::vector<T>;

// 2D array
template<typename T>
using data2D = std::vector<std::vector<T>>;

// 3D array
template<typename T>
using data3D = std::vector<std::vector<std::vector<T>>>;


// My function that taken in the full (ell1, ell2, ell3) values to return the appropriate
// 3j symbol out of the returned array from Fortran
float wigner3j(const int ell1, int ell2, int ell3, int m1, int m2, int m3)
{
  // Demand that the sum of the m's are zero
  if (m1 + m2 + m3 != 0) return 0;
  // Enforce the triangle condition on the ell's
  if (((ell1 + ell2 < ell3) || (ell1 + ell3 < ell2)) || (ell2 + ell3 < ell1)) return 0;

  // Find the 'start' and 'end' values of the returned ell array
  const int start = std::max(std::abs(ell2 - ell3), std::abs(m2 + m3));
  const int end = ell2 + ell3;

  // Compute the 'length' of this array
  const int length = (end - start) + 1;

  // Initiate the vector, which is where the Fortran code will store the output in
  double data[length];

  // Call the Fortran code with appropriate arguments
  getthreejs(data, &ell2, &ell3, &m2, &m3);

  // Using the provided ell1 argument, we find where this is in the returned array
  const int ell1_pos = ell1 - start;

  // Return the data from the required position.
  // We cast to float here, as we are storing values as floats to conserve memory usage
  return static_cast<float>(data[ell1_pos]);
}

// My function to calculate the Gaunt integral, by using the Wigner-3j symbols
float gaunt(const int ell1, const int ell2, const int ell3, const int m1, const int m2, const int m3)
{
  // Test the required conditions on the Gaunt integral
  if (m1 + m2 + m3 != 0) return 0;
  if ((ell1 + ell2 + ell3) % 2 != 0) return 0;
  if (((ell1 + ell2 < ell3) || (ell1 + ell3 < ell2)) || (ell2 + ell3 < ell1)) return 0;

  // Compute the individual factors
  const float sqrt_arg = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1);
  const float prefactor = sqrt(sqrt_arg / (4 * M_PI));
  const float wigner1 = wigner3j(ell1, ell2, ell3, m1, m2, m3);
  const float wigner2 = wigner3j(ell1, ell2, ell3, 0, 0, 0);

  return prefactor * wigner1 * wigner2;
}

// We enclose the main library functions in this extern, so that way they are callable from Python
extern "C"
{

void build_ell_grid_cpp(const int ell_max, const int ell_step, const char *file_path, const bool full_grid)
{
  /*
  This function is designed to build an (ell1, ell2, ell3) grid given the
  parameters ell_max and ell_step, and save the grid to the provided file path.
  */

  std::cout << "--- Starting to build ell grid ---" << std::endl;

  // Create the output file stream object, with the provided file path
  std::ofstream outf(file_path, std::ios_base::out | std::ios_base::trunc);

  // Output the header for the csv file
  outf << "ell1" << "," << "ell2" << "," << "ell3" << "\n";

  int num_configurations = 0;

  // Loop through each ell combination, starting at zero and ending at ell_max
  for (int ell1 = 0; ell1 <= ell_max; ell1 += ell_step)
  {
    for (int ell2 = 0; ell2 <= ell_max; ell2 += ell_step)
    {
      // If we are only saving the cut grid, only save configurations where ell1 >= ell2 >= ell3
      if (!full_grid)
      { if (ell2 > ell1) continue; }

      for (int ell3 = 0; ell3 <= ell_max; ell3 += ell_step)
      {
        if (!full_grid)
        { if (ell3 > ell2) continue; }

        // First triangle condition
        if ((ell1 + ell2 + ell3) % 2 != 0) continue;

        // Second triangle condition
        if (((ell1 + ell2 <= ell3) || (ell1 + ell3 <= ell2)) || (ell2 + ell3 <= ell1)) continue;

        // Save the successful configuration to disk
        outf << std::to_string(ell1) << "," << std::to_string(ell2) << "," << std::to_string(ell3) << "\n";

        // Increment the number of configurations by one for a successful configuration
        ++num_configurations;
      }
    }
  }

  // Close the file stream
  outf.close();

  std::cout << "--- Finished building grid ---" << std::endl;
  std::cout << "--- Number of configurations: " << num_configurations << " ---" << std::endl;
}


// Create our 3D array interp_bispec which is where we will store interpolated bispectrum in
data1D<float> vec1d(900, 0);  // TODO: change this 900 to something more appropriate, and adjustable
data2D<float> vec2d(900, vec1d);
static data3D<float> interp_bispec(900, vec2d);

// Set up arrays for storing the read in cl & alm data. Note that the alm's are complex.
static data1D<float> cls(2500, 0.0);
data1D<std::complex<float>> cl_complex(2551);
static data2D<std::complex<float>> alms(2551, cl_complex); // TODO: change these ell_max's


// This function is needed because the interpolated bispectrum is only evaluated for
// ell1 >= ell2 >= ell3, whereas it is easier if the sum is over all permutations of
// ell2 and ell3.
float cmb_bispectrum(const int ell1, const int ell2, const int ell3, const data3D<float> &data)
{
  // Create a vector, and then sort this, so that way the largest values are at the start
  std::vector<int> ells = {ell1, ell2, ell3};
  std::sort(ells.begin(), ells.end(), std::greater<>());

  return data[ells[0]][ells[1]][ells[2]];
}


void initalise()
{
  // Use the provided CSV reader in csv.h to read in the interpolated CMB bispectrum
  io::CSVReader<4> cmb_bispec("/home/amaraio/Documents/CMBBispectrumCalc/lib/interpolated_grid.csv");
  // TODO: make these input file names adjustable and default to something more sensible

  // Set the header of the csv file
  cmb_bispec.read_header(io::ignore_no_column, "ell1", "ell2", "ell3", "data");

  // Use the csv.h functionality to go through the data and store the
  // data which exists in the flattened list into the 3D array/
  int ell1_row;
  int ell2_row;
  int ell3_row;
  float data;
  while (cmb_bispec.read_row(ell1_row, ell2_row, ell3_row, data))
  {
    interp_bispec[ell1_row][ell2_row][ell3_row] = data;
  }

  // Now read in the cl and alm values, computed from the power spectrum
  io::CSVReader<2> cl_in("/home/amaraio/Documents/CMBBispectrumCalc/lib/cl.csv");
  io::CSVReader<4> alm_in("/home/amaraio/Documents/CMBBispectrumCalc/lib/alm.csv");

  cl_in.read_header(io::ignore_no_column, "ell", "cl");
  alm_in.read_header(io::ignore_no_column, "ell", "m", "alm_re", "alm_im");

  int ell;
  int m;
  float cl;
  float alm_re;
  float alm_im;
  while (cl_in.read_row(ell, cl))
  {
    cls[ell] = cl;
  }

  while (alm_in.read_row(ell, m, alm_re, alm_im))
  {
    alms[ell][m] = std::complex<float>(alm_re, alm_im);
  }
}

// Older, more naive way of calculating the alm coefficients
void calc_alm(const int ell1, const int m1, const int ell_max, float &alm_re_in, float &alm_im_in)
{
  std::complex<float> j(0.0, 0.0);

  auto start_time = std::chrono::system_clock::now();

  // Note, we start here from two as for ell=0,1 we have Cl=alm=0 and so does not contribute
  #pragma omp parallel for  //for reduction(+:j) NOLINT(openmp-use-default-none)
  for (int ell2 = 2; ell2 <= ell_max; ++ell2)
  {
    std::complex<float> local_result(0.0, 0.0);
    for (int ell3 = 2; ell3 <= ell_max; ++ell3)
    {
      if ((ell1 + ell2 + ell3) % 2 != 0) continue;

      if (((ell1 + ell2 < ell3) || (ell1 + ell3 < ell2)) || (ell2 + ell3 < ell1)) continue;

      for (int m2 = -ell2; m2 <= ell2; ++m2)
      {
        for (int m3 = -ell3; m3 <= ell3; ++m3)
        {
          if (m1 + m2 + m3 != 0) continue;

          local_result += cmb_bispectrum(ell1, ell2, ell3, interp_bispec) * gaunt(ell1, ell2, ell3, m1, m2, m3)
                          * std::conj(alms[ell2][std::abs(m2)] * alms[ell3][std::abs(m3)]) / (cls[ell2] * cls[ell3]);
        }
      }
    }

    #pragma omp critical
    j += local_result;
  }

  auto finish_time = std::chrono::system_clock::now();
  float elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(finish_time - start_time).count();

  std::cout << "gaunt integral: " << j << std::endl;
  std::cout << "Took: " << elapsed_seconds << " microseconds" << std::endl;

  alm_re_in = std::real(j);
  alm_im_in = std::imag(j);
}

// More efficient way to compute the non-Gaussian alm's. Here, we take arguments of ell1 & m1, which correspond to
// the specific aln that we wish to compute, and two references to floats, in which we store the real and imaginary
// parts of the computed alm coefficient.
void calc_alm2(const int ell1, const int m1, const int ell_max, float &alm_re_in, float &alm_im_in)
{
  // Declare complex float alm to zero, for the values to get added into as the sum progresses
  std::complex<float> alm(0.0, 0.0);

  // Note, we start here from two as for ell=0,1 we have Cl=alm=0 and so does not contribute
  // Here, we use OpenMP for simple parallelization, which speeds up the computation significantly
  #pragma omp parallel for  // NOLINT(openmp-use-default-none)
  for (int ell2 = 2; ell2 <= ell_max; ++ell2)
  {
    for (int ell3 = 2; ell3 <= ell2; ++ell3)
    {
      // Test for the triangle conditions on the ells.
      if ((ell1 + ell2 + ell3) % 2 != 0) continue;

      if (((ell1 + ell2 < ell3) || (ell1 + ell3 < ell2)) || (ell2 + ell3 < ell1)) continue;

      // Now, the combination of ells is valid, so create new local variable for sum to be added into
      std::complex<float> local_result(0.0, 0.0);

      for (int m2 = -ell2; m2 <= ell2; ++m2)
      {
        // Since m3 is determined from m1 & m2, use it - but also test that it is valid given ell3
        int m3 = -(m1 + m2);
        if (std::abs(m3) > ell3) continue;

        // Add to the local result the evaluation of the m sum.
        local_result += wigner3j(ell1, ell2, ell3, m1, m2, m3) *
                        std::conj(alms[ell2][std::abs(m2)] * alms[ell3][std::abs(m3)]);
      }

      // Now, we can multiply the m sum by the parts that only depend on the ell vales.
      unsigned long sqrt_arg = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1);
      local_result *= wigner3j(ell1, ell2, ell3, 0, 0, 0) *
                      cmb_bispectrum(ell1, ell2, ell3, interp_bispec) / (cls[ell2] * cls[ell3]) *
                      sqrt(sqrt_arg / (4 * M_PI));

      // If ell2 != ell3, then as we are only doing ell2 >= ell3, we need to multiply by two for the ell3 > ell2 case.
      if (ell2 != ell3) local_result *= 2;

      // Use OpenMP pragma to ensure that only one thread is writing their result into the output at once.
      #pragma omp critical
      alm += local_result;
    }
  }

  // Store the real and imaginary parts of the computed alm into the provided reference.
  alm_re_in = std::real(alm);
  alm_im_in = std::imag(alm);
}

int main()
{
  float alm_re;
  float alm_im;

  auto start_time = std::chrono::system_clock::now();
  initalise();
  auto finish_time = std::chrono::system_clock::now();
  float elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(finish_time - start_time).count();
  std::cout << "Initialisation took: " << elapsed_seconds << " microseconds" << std::endl;

  calc_alm2(10, 5, 500, alm_re, alm_im);
  std::cout << alm_re << std::endl;
  std::cout << alm_im << std::endl;

  calc_alm2(25, 5, 500, alm_re, alm_im);
  std::cout << alm_re << std::endl;
  std::cout << alm_im << std::endl;

  calc_alm2(50, 5, 500, alm_re, alm_im);
  std::cout << alm_re << std::endl;
  std::cout << alm_im << std::endl;

  calc_alm2(100, 5, 500, alm_re, alm_im);
  std::cout << alm_re << std::endl;
  std::cout << alm_im << std::endl;

  calc_alm2(250, 5, 500, alm_re, alm_im);
  calc_alm2(250, 6, 500, alm_re, alm_im);
  calc_alm2(450, 6, 500, alm_re, alm_im);
  std::cout << alm_re << std::endl;
  std::cout << alm_im << std::endl;

  return 0;
}


}  // END: extern "C"

#include <Signal.h>


/**
 * @brief Computes the Fourier encoding for a given set of parameters.
 *
 * This function calculates the Fourier encoding using the provided spatial coordinates,
 * k-space trajectories, and phase offsets. The result is a complex array representing
 * the Fourier encoded data.
 *
 * @param r A matrix of spatial coordinates (Nx3).
 * @param kx A 3D tensor representing the k-space trajectory in the x-direction.
 * @param ky A 3D tensor representing the k-space trajectory in the y-direction.
 * @param kz A 3D tensor representing the k-space trajectory in the z-direction.
 * @param phi_off A vector of phase offsets.
 * @param i The index for the first dimension of the k-space tensors.
 * @param j The index for the second dimension of the k-space tensors.
 * @param k The index for the third dimension of the k-space tensors.
 * @return A complex array representing the Fourier encoded data.
 */
template <typename T>
Array<std::complex<T>, Dynamic, 1> FourierEncoding(
  const Matrix<T, Dynamic, Dynamic> &r, 
  const Tensor<T, 3> kx, 
  const Tensor<T, 3> ky, 
  const Tensor<T, 3> kz, 
  const Vector<T, Dynamic> &phi_off, 
  const uint i, 
  const uint j, 
  const uint k
  ){
    const std::complex<T> i1(0.0, 1.0);
    return (- i1 * (r.col(0) * kx(i,j,k) + r.col(1) * ky(i,j,k) + r.col(2) * kz(i,j,k) + phi_off)).array().exp();
}

template Array<std::complex<float>, Dynamic, 1> FourierEncoding<float>(
    const Matrix<float, Dynamic, Dynamic> &, 
    const Tensor<float, 3>, 
    const Tensor<float, 3>, 
    const Tensor<float, 3>, 
    const Vector<float, Dynamic> &, 
    const uint, 
    const uint, 
    const uint
  );

template Array<std::complex<double>, Dynamic, 1> FourierEncoding<double>(
    const Matrix<double, Dynamic, Dynamic> &, 
    const Tensor<double, 3>, 
    const Tensor<double, 3>, 
    const Tensor<double, 3>, 
    const Vector<double, Dynamic> &, 
    const uint, 
    const uint, 
    const uint
  );



/**
 * @brief Computes the signal from the given matrices and vectors.
 * 
 * This function calculates the signal by performing element-wise multiplication
 * of the specified column of matrix Mxy, the fourier vector, and the decay vector,
 * then multiplying the result by the sparse matrix M, and finally summing up all
 * the elements of the resulting vector.
 * 
 * @tparam T The type of the elements in the matrices and vectors.
 * @param M A sparse matrix of type T.
 * @param Mxy A dense matrix of complex numbers of type T.
 * @param fourier A vector of complex numbers of type T.
 * @param decay A vector of type T representing the decay factors.
 * @param enc_dir An unsigned integer specifying the encoding direction (column index) in Mxy.
 * @return A complex number of type T representing the computed signal.
 */
template <typename T>
std::complex<T> signal(
  const SparseMatrix<T> &M, 
  const Matrix<std::complex<T>, Dynamic, Dynamic> &Mxy, 
  const Vector<std::complex<T>, Dynamic> &fourier, 
  const Vector<T, Dynamic> &decay, 
  const uint &enc_dir
  ){
  return (M * Mxy.col(enc_dir).cwiseProduct(fourier).cwiseProduct(decay)).sum();
}

template std::complex<float> signal<float>(
    const SparseMatrix<float> &, 
    const Matrix<std::complex<float>, Dynamic, Dynamic> &, 
    const Vector<std::complex<float>, Dynamic> &, 
    const Vector<float, Dynamic> &, 
    const uint &
  );

template std::complex<double> signal<double>(
    const SparseMatrix<double> &, 
    const Matrix<std::complex<double>, Dynamic, Dynamic> &, 
    const Vector<std::complex<double>, Dynamic> &, 
    const Vector<double, Dynamic> &, 
    const uint &
  );



/**
 * @brief Computes the signal for a given encoding direction.
 *
 * This function calculates the signal by performing a series of matrix and vector operations.
 * It multiplies a sparse matrix with a column of a complex matrix, element-wise multiplies 
 * the result with a complex vector, sums the resulting vector, and then scales it by a decay factor.
 *
 * @tparam T The numeric type used for the real part of the complex numbers.
 * 
 * @param M A sparse matrix of type T.
 * @param Mxy A dense matrix of complex numbers with real part of type T.
 * @param fourier A vector of complex numbers with real part of type T.
 * @param decay A decay factor of type T.
 * @param enc_dir The encoding direction, represented as an unsigned integer.
 * 
 * @return A complex number of type T representing the computed signal.
 */
template <typename T>
std::complex<T> signal(
  const SparseMatrix<T> &M, 
  const Matrix<std::complex<T>, Dynamic, Dynamic> &Mxy, 
  const Vector<std::complex<T>, Dynamic> &fourier, 
  const T &decay, 
  const uint &enc_dir
  ){
  return (M * Mxy.col(enc_dir).cwiseProduct(fourier)).sum() * decay;
}

template std::complex<float> signal<float>(
    const SparseMatrix<float> &, 
    const Matrix<std::complex<float>, Dynamic, Dynamic> &, 
    const Vector<std::complex<float>, Dynamic> &, 
    const float &, 
    const uint &
  );

template std::complex<double> signal<double>(
    const SparseMatrix<double> &, 
    const Matrix<std::complex<double>, Dynamic, Dynamic> &, 
    const Vector<std::complex<double>, Dynamic> &, 
    const double &, 
    const uint &
  );



/**
 * @brief Updates the position matrix based on initial position, velocity, and time step.
 * 
 * @tparam T The type of the elements in the matrices (e.g., float, double).
 * @param r0 The initial position matrix.
 * @param v0 The initial velocity matrix.
 * @param dt The time step.
 * @return A matrix representing the updated positions.
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> UpdatePosition(
  const Matrix<T, Dynamic, Dynamic> &r0, 
  const Matrix<T, Dynamic, Dynamic> &v0, 
  const T &dt){
  return r0 + v0 * dt;
}

template Matrix<float, Dynamic, Dynamic> UpdatePosition<float>(
    const Matrix<float, Dynamic, Dynamic> &, 
    const Matrix<float, Dynamic, Dynamic> &, 
    const float &
  );

template Matrix<double, Dynamic, Dynamic> UpdatePosition<double>(
    const Matrix<double, Dynamic, Dynamic> &, 
    const Matrix<double, Dynamic, Dynamic> &, 
    const double &
  );



/**
 * @brief Updates the position of an object based on its initial position, velocity, acceleration, and time step.
 * 
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @param r0 The initial position matrix.
 * @param v0 The initial velocity matrix.
 * @param a0 The initial acceleration matrix.
 * @param dt The time step.
 * @return Matrix<T, Dynamic, Dynamic> The updated position matrix.
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> UpdatePosition(
  const Matrix<T, Dynamic, Dynamic> &r0, 
  const Matrix<T, Dynamic, Dynamic> &v0, 
  const Matrix<T, Dynamic, Dynamic> &a0, 
  const T &dt){
  return r0 + v0 * dt + a0 * (dt * dt) / 2;
}

template Matrix<float, Dynamic, Dynamic> UpdatePosition<float>(
    const Matrix<float, Dynamic, Dynamic> &, 
    const Matrix<float, Dynamic, Dynamic> &, 
    const Matrix<float, Dynamic, Dynamic> &, 
    const float &
  );

template Matrix<double, Dynamic, Dynamic> UpdatePosition<double>(
    const Matrix<double, Dynamic, Dynamic> &, 
    const Matrix<double, Dynamic, Dynamic> &, 
    const Matrix<double, Dynamic, Dynamic> &, 
    const double &
  );
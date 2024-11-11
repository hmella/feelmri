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
ArrayXcf FourierEncoding(const MatrixXf &r, const Tensor<float, 3> kx, const Tensor<float, 3> ky, const Tensor<float, 3> kz, const VectorXf &phi_off, const uint i, const uint j, const uint k){
  return (- i1 * (r.col(0) * kx(i,j,k) + r.col(1) * ky(i,j,k) + r.col(2) * kz(i,j,k) + phi_off)).array().exp();
}

/**
 * @brief Computes the signal by applying a sparse matrix transformation, 
 *        element-wise product with a Fourier vector, and a decay factor.
 * 
 * @param M A sparse matrix of floats representing the transformation matrix.
 * @param Mxy A complex matrix where each column represents a different state.
 * @param fourier A complex vector representing the Fourier coefficients.
 * @param decay A float reference representing the decay factor to be applied.
 * @return cfloat The computed signal as a complex float.
 */
cfloat signal(const SparseMatrix<float> &M, const MatrixXcf &Mxy, const VectorXcf &fourier, const VectorXf &decay, const uint &enc_dir){
  return (M * Mxy.col(enc_dir).cwiseProduct(fourier).cwiseProduct(decay)).sum();
}

cfloat signal(const SparseMatrix<float> &M, const MatrixXcf &Mxy, const VectorXcf &fourier, const float &decay, const uint &enc_dir){
  return (M * Mxy.col(enc_dir).cwiseProduct(fourier)).sum() * decay;
}

/**
 * @brief Updates the position of an object based on its initial position, velocity, and time step.
 * 
 * @param r0 Initial position matrix.
 * @param v0 Initial velocity matrix.
 * @param dt Time step.
 * @return MatrixXf Updated position matrix.
 */
MatrixXf UpdatePosition(const MatrixXf &r0, const MatrixXf &v0, const float &dt){
  return r0 + v0 * dt;
}

/**
 * @brief Updates the position of an object based on its initial position, velocity, and time step.
 * 
 * @param r0 Initial position matrix.
 * @param v0 Initial velocity matrix.
 * @param dt Time step.
 * @return MatrixXf Updated position matrix.
 */
MatrixXf UpdatePosition(const MatrixXf &r0, const MatrixXf &v0, const MatrixXf &a0, const float &dt){
  return r0 + v0 * dt + a0 * dt * dt / 2;
}
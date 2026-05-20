#include <vector>
#include <complex>
#include <cmath>
#include <FEUtils.h>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h> // REQUIRED for PyBind11 to seamlessly cast Eigen types
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

#ifdef FEELMRI_GPU
#include "kernels/MRIAssemble_gpu.hpp"
#include "runtime/device_init.hpp"
#endif

// Alias for convenience to avoid typing pybind11:: constantly
namespace py = pybind11;

// =============================================================================
// MRI SIGNAL ASSEMBLER
// =============================================================================
// Template class to allow varying precision (e.g., float or double, typically float)
template <typename T>
class SignalAssembler
{
public:
    // Convenience type aliases for complex numbers and Eigen Tensors
    using C = std::complex<T>;
    using Tensor3 = Eigen::Tensor<T, 3>; 
    using Tensor4CR = Eigen::Tensor<C, 4, Eigen::RowMajor>;

    // Constructor initializes the mesh topology, nodes, and computes quadrature rules.
    // Passed arguments as standard const references for clean C++ 
    // signatures, avoiding PyBind11 specific memory wrappers.
    SignalAssembler(
        const Eigen::MatrixXi& elems,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& nodes,
        const std::string& meshio_type,
        int quadrature_degree)
      : elems_(elems) // Store element connectivity
    {
        nelem_ = elems_.rows(); // Number of elements in the mesh
        
        // Precompute finite element characteristics (weights, points, shape functions)
        cache_ = BuildFEQuadratureCache<T>(elems_, nodes, meshio_type, quadrature_degree);

        nb_nodes_ = nodes.rows(); // Total number of nodes in the mesh

        // Extract static X, Y, Z coordinates for all nodes into 1D arrays
        f_nodes_x0_ = nodes.col(0).array();
        f_nodes_x1_ = nodes.col(1).array();
        f_nodes_x2_ = nodes.col(2).array();
        
        // Allocate space for dynamic (moving) nodal coordinates
        f_dyn_nodes_x0_.resize(nb_nodes_);
        f_dyn_nodes_x1_.resize(nb_nodes_);
        f_dyn_nodes_x2_.resize(nb_nodes_);

        nq_ = (int)cache_.wq[0].size(); // Number of quadrature points per element
        total_q_ = nelem_ * nq_;        // Total number of quadrature points globally

        // Allocate space for static and dynamic quadrature point coordinates and weights
        f_xq0_.resize(total_q_);
        f_xq1_.resize(total_q_);
        f_xq2_.resize(total_q_);
        f_wq_.resize(total_q_);

        f_dyn_xq0_.resize(total_q_);
        f_dyn_xq1_.resize(total_q_);
        f_dyn_xq2_.resize(total_q_);

        // Flatten the element-wise quadrature coordinates and weights into global 1D arrays
        for (int e = 0; e < nelem_; ++e)
        {
            const int offset = e * nq_;
            f_xq0_.segment(offset, nq_) = cache_.xq[e].col(0).array();
            f_xq1_.segment(offset, nq_) = cache_.xq[e].col(1).array();
            f_xq2_.segment(offset, nq_) = cache_.xq[e].col(2).array();
            f_wq_.segment(offset, nq_)  = cache_.wq[e].array();
        }

        // Build a Sparse Matrix (S_global_) that maps nodal values to quadrature point values
        S_global_.resize(total_q_, nb_nodes_);
        std::vector<Eigen::Triplet<T>> triplets;
        const int nne = elems_.cols(); // Nodes per element
        triplets.reserve(total_q_ * nne);

        // Populate the triplets with shape function values evaluated at quadrature points
        for (int e = 0; e < nelem_; ++e) {
            for (int q = 0; q < nq_; ++q) {
                const int global_q = e * nq_ + q;
                for (int a = 0; a < nne; ++a) {
                    triplets.emplace_back(global_q, elems_(e, a), cache_.SqT[e](q, a));
                }
            }
        }
        
        // Assemble the sparse projection matrix and compress it for fast multiplication
        S_global_.setFromTriplets(triplets.begin(), triplets.end());
        S_global_.makeCompressed(); 
    }

    // Stores and interpolates static MRI tissue parameters (T2 relaxation, B0 off-resonance).
    void set_static_fields(
        const Eigen::Array<T, Eigen::Dynamic, 1>& T2,
        const Eigen::Array<T, Eigen::Dynamic, 1>& phi_dB0)
    {
        // Precompute inverse T2 for faster exponential calculations later
        Eigen::Array<T, Eigen::Dynamic, 1> inv_T2 = T2.inverse();
        const int nne = elems_.cols();

        // Allocate arrays for interpolated parameters at quadrature points
        f_invT2_.resize(total_q_);
        f_phi_.resize(total_q_);

        // Store the raw nodal parameters
        f_nodes_invT2_ = inv_T2;
        f_nodes_phi_ = phi_dB0;

        // Interpolate nodal T2 and B0 parameters to every quadrature point
        for (int e = 0; e < nelem_; ++e)
        {
            const int offset = e * nq_;
            
            Eigen::Vector<T, Eigen::Dynamic> invT2_e(nne);
            Eigen::Vector<T, Eigen::Dynamic> phi_e(nne);
            
            // Gather nodal values for the current element
            for (int a = 0; a < nne; ++a) {
                const int idx = elems_(e, a);
                invT2_e(a) = inv_T2(idx);
                phi_e(a)   = phi_dB0(idx);
            }

            // Multiply shape functions by nodal values to get quadrature point values
            f_invT2_.segment(offset, nq_) = cache_.SqT[e] * invT2_e;
            f_phi_.segment(offset, nq_)   = cache_.SqT[e] * phi_e;
        }
    }

    // Updates transverse magnetization strictly at the nodes (used for fast nodal sums).
    void update_magnetization(const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>& Mxy)
    {
        nv_ = (int)Mxy.cols(); // Number of receiving coils / isochromats
        f_Mxy_nodes_ = Mxy; 
    }

    // Updates and interpolates transverse magnetization to all quadrature points.
    void update_full_magnetization(const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>& Mxy)
    {
        nv_ = (int)Mxy.cols();
#ifdef FEELMRI_GPU
        // GPU quadrature path projects on the device from f_Mxy_nodes_;
        // the host-side f_Mxy_ is unused there, so skip the projection
        // entirely (the dominant per-frame host cost at large meshes).
        if (device_ == "gpu") {
            return;
        }
#endif
        f_Mxy_.resize(total_q_, nv_);

        // S_global_ (built in the constructor, shape Q x N RowMajor sparse,
        // real-valued) encodes exactly the per-element shape-function
        // projection that the legacy per-element loop computed. Going
        // through it directly:
        //   - eliminates O(nelem) small Eigen::Matrix allocations and
        //     per-element complex casts (the dominant host cost at scale,
        //     ~200 ms per call on a 550k-node 4D-flow mesh),
        //   - replaces them with two real sparse-times-dense matmuls
        //     (Eigen handles AVX2 vectorisation internally) plus a single
        //     vectorisable interleave pass.
        // S_global_ stays real; we split Mxy into real / imag halves to
        // keep both matmuls real-typed (a complex-typed sparse matrix
        // would double the storage and run slower).
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Mxy_re = Mxy.real();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Mxy_im = Mxy.imag();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_Mxy_re =
            S_global_ * Mxy_re;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_Mxy_im =
            S_global_ * Mxy_im;
        for (int q = 0; q < total_q_; ++q) {
            for (int v = 0; v < nv_; ++v) {
                f_Mxy_(q, v) = C(f_Mxy_re(q, v), f_Mxy_im(q, v));
            }
        }
    }

    // Pre-multiplies magnetization by a mass matrix (M) for Galerkin-style nodal integration.
    void update_nodal_magnetization(
        const Eigen::SparseMatrix<T>& M,
        const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>& Mxy)
    {
        nv_ = (int)Mxy.cols();
        // M is real; Mxy is complex. Split into real/imag halves to keep
        // both matmuls real-typed (parity with update_full_magnetization's
        // vectorisation). The previous `M.cast<C>() * Mxy` materialised a
        // complex sparse matrix per call, doubling the storage and the
        // matmul cost. Splitting eliminates that.
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Mxy_re = Mxy.real();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Mxy_im = Mxy.imag();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mM_re =
            M * Mxy_re;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mM_im =
            M * Mxy_im;
        f_M_Mxy_nodes_.resize(mM_re.rows(), nv_);
        for (int n = 0; n < mM_re.rows(); ++n) {
            for (int v = 0; v < nv_; ++v) {
                f_M_Mxy_nodes_(n, v) = C(mM_re(n, v), mM_im(n, v));
            }
        }
    }

    // Switch this assembler between the CPU and GPU compute paths.
    // The GPU path requires a build with FEELMRI_ENABLE_GPU=ON and a
    // visible device; raises a clear error otherwise.
    void set_device(const std::string& device) {
      if (device != "cpu" && device != "gpu") {
        throw std::invalid_argument(
          "SignalAssembler.set_device: must be 'cpu' or 'gpu', got '" + device + "'");
      }
#ifndef FEELMRI_GPU
      if (device == "gpu") {
        throw std::runtime_error(
          "SignalAssembler.set_device('gpu'): this build was compiled without "
          "GPU support. Rebuild with -DFEELMRI_ENABLE_GPU=ON.");
      }
#endif
      device_ = device;
    }

    const std::string& device() const { return device_; }

    // Returns a characteristic length scale (cube root of volume) for each element
    Eigen::Vector<T, Eigen::Dynamic> estimate_element_sizes() const
    {
        Eigen::Vector<T, Eigen::Dynamic> sizes(nelem_);
        for (int e = 0; e < nelem_; ++e)
        {
            const int offset = e * nq_;
            T volume = f_wq_.segment(offset, nq_).sum(); // Summing weights gives element volume
            sizes(e) = std::cbrt(volume);
        }
        return sizes;
    }

    // =========================================================================
    // FAST NODAL INTEGRATION SIGNAL GENERATOR (SUM OF NODAL VALUES)
    // =========================================================================   
    // Reverted py::array_t manual mappings to standard Eigen::Tensor and
    // const reference parameters, keeping the pure C++ nature of the functions intact.
    // Replaced the Python callable `pod_trajectory` with pre-computed `modes` 
    // and `weights` matrices to entirely eliminate the Global Interpreter Lock (GIL) and allow AVX2 math.
    Tensor4CR signal_sum(
        const std::vector<Tensor3> &kloc,
        const Tensor3 &t,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_x,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_y,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_z,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& weights,
        bool has_traj)
    {
#ifdef FEELMRI_GPU
        if (device_ == "gpu") {
            return signal_node_gpu_impl(kloc, t, modes_x, modes_y, modes_z,
                                          weights, has_traj, /*use_M_Mxy=*/false);
        }
#endif

        const C i1(T(0), T(1));
        const T two_pi  = T(2) * T(M_PI);

        // Extract dimensions from the k-space trajectory tensor
        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S = nb_meas * nb_lines * nb_kz; // Total number of k-space samples

        // Allocate the output matrix (Samples x Coils) and a temporary row vector
        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kspace_mat(S, nv_);
        Eigen::RowVector<C, Eigen::Dynamic> s(nv_);
        
        T t_old = T(-1); // Tracks previous time step to avoid redundant calculations

        // Cache-blocking parameters to ensure arrays fit in CPU L1/L2 cache
        const int BLOCK_SIZE = 8192;
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);

        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(nb_nodes_);
        Eigen::Array<T, Eigen::Dynamic, 1> f_po(nb_nodes_);

        // Select the correct coordinate reference based on presence of trajectory
        Eigen::Array<T, Eigen::Dynamic, 1>& x0_ref = has_traj ? f_dyn_nodes_x0_ : f_nodes_x0_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x1_ref = has_traj ? f_dyn_nodes_x1_ : f_nodes_x1_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x2_ref = has_traj ? f_dyn_nodes_x2_ : f_nodes_x2_;

        // Loop over all k-space points (Measurements x Lines x Slices)
        for (uint i = 0, row = 0; i < nb_meas; ++i)
        for (uint j = 0; j < nb_lines; ++j)
        for (uint k = 0; k < nb_kz; ++k, ++row)
        {
            // Extract current time and k-space coordinates (kx, ky, kz)
            const T tij = t(i, j, k); 
            const T kx  = two_pi * kloc[0](i, j, k);
            const T ky  = two_pi * kloc[1](i, j, k);
            const T kz  = two_pi * kloc[2](i, j, k);

            // Optimization flag: Only update mesh/relaxation if time has changed
            const bool update_time = (tij != t_old);

            // Update node positions by executing a highly-optimized matrix-vector multiplication
            if (has_traj && update_time) 
            {
                // Extract the pre-computed weights vector for this specific 
                // time frame (M_modes x 1) and multiply by static modes (N_nodes x M_modes).
                // The `.noalias()` flag instructs Eigen to skip temporary allocation and 
                // unroll the math directly into CPU AVX2/FMA vector instructions.
                auto w = weights.row(row).transpose(); 
                f_dyn_nodes_x0_ = f_nodes_x0_ + (modes_x * w).array();
                f_dyn_nodes_x1_ = f_nodes_x1_ + (modes_y * w).array();
                f_dyn_nodes_x2_ = f_nodes_x2_ + (modes_z * w).array();
            }

            s.setZero(); // Reset signal accumulator for this k-space point
            if (update_time) t_old = tij;

            // Process nodes in chunks (BLOCK_SIZE) to maximize CPU cache hits
            for (int q_start = 0; q_start < nb_nodes_; q_start += BLOCK_SIZE)
            {
                const int q_count = std::min(BLOCK_SIZE, nb_nodes_ - q_start);
                
                // If time advanced, update the T2 decay (magnitude) and B0 phase accumulation
                if (update_time) {
                    f_mag.segment(q_start, q_count) = (-tij * f_nodes_invT2_.segment(q_start, q_count)).exp();
                    f_po.segment(q_start, q_count)  = f_nodes_phi_.segment(q_start, q_count) * tij;
                }

                // Compute total phase: B0 phase accumulation - spatial encoding (k * r)
                phase_block.head(q_count) = f_po.segment(q_start, q_count)
                                            - kx * x0_ref.segment(q_start, q_count) 
                                            - ky * x1_ref.segment(q_start, q_count) 
                                            - kz * x2_ref.segment(q_start, q_count);

                // Euler's formula: construct complex exponentials (real=cos, imag=sin) multiplied by decay magnitude
                fourier_block.head(q_count).array().real() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).sin();

                // Vectorized multiply-add: integrate signal from this block of nodes
                s.noalias() += fourier_block.head(q_count).transpose() * f_Mxy_nodes_.middleRows(q_start, q_count);
            }
            
            // Store the integrated signal for this k-space point
            kspace_mat.row(row) = s;
        }

        // Return the flat matrix mapped back to a 4D Tensor format for Python
        return Eigen::TensorMap<Tensor4CR>(kspace_mat.data(), nb_meas, nb_lines, nb_kz, (uint)nv_);
    }

    // =========================================================================
    // FAST NODAL INTEGRATION SIGNAL GENERATOR (MASS MATRIX)
    // =========================================================================
    // Cleaned up signatures to native Eigen types while keeping the
    // AVX2 Modes-Weights multiplication architecture.
    Tensor4CR signal_nodal(
        const std::vector<Tensor3> &kloc,
        const Tensor3 &t,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_x,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_y,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_z,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& weights,
        bool has_traj)
    {
#ifdef FEELMRI_GPU
        if (device_ == "gpu") {
            return signal_node_gpu_impl(kloc, t, modes_x, modes_y, modes_z,
                                          weights, has_traj, /*use_M_Mxy=*/true);
        }
#endif
        // This function is structurally identical to signal_sum, except it integrates
        // using the pre-computed mass-matrix projection (f_M_Mxy_nodes_) instead of raw Mxy.
        const C i1(T(0), T(1));
        const T two_pi  = T(2) * T(M_PI);

        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S = nb_meas * nb_lines * nb_kz;

        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kspace_mat(S, nv_);
        Eigen::RowVector<C, Eigen::Dynamic> s(nv_);
        
        T t_old = T(-1);

        const int BLOCK_SIZE = 8192;
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);

        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(nb_nodes_);
        Eigen::Array<T, Eigen::Dynamic, 1> f_po(nb_nodes_);

        Eigen::Array<T, Eigen::Dynamic, 1>& x0_ref = has_traj ? f_dyn_nodes_x0_ : f_nodes_x0_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x1_ref = has_traj ? f_dyn_nodes_x1_ : f_nodes_x1_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x2_ref = has_traj ? f_dyn_nodes_x2_ : f_nodes_x2_;

        // Standard k-space integration loop
        for (uint i = 0, row = 0; i < nb_meas; ++i)
        for (uint j = 0; j < nb_lines; ++j)
        for (uint k = 0; k < nb_kz; ++k, ++row)
        {
            const T tij = t(i, j, k); 
            const T kx  = two_pi * kloc[0](i, j, k);
            const T ky  = two_pi * kloc[1](i, j, k);
            const T kz  = two_pi * kloc[2](i, j, k);

            const bool update_time = (tij != t_old);

            // Fetch and apply dynamic trajectory positions (AVX2 optimized)
            if (has_traj && update_time) 
            {
                auto w = weights.row(row).transpose(); 
                f_dyn_nodes_x0_ = f_nodes_x0_ + (modes_x * w).array();
                f_dyn_nodes_x1_ = f_nodes_x1_ + (modes_y * w).array();
                f_dyn_nodes_x2_ = f_nodes_x2_ + (modes_z * w).array();
            }

            s.setZero();
            if (update_time) t_old = tij;

            // Blocked evaluation for performance
            for (int q_start = 0; q_start < nb_nodes_; q_start += BLOCK_SIZE)
            {
                const int q_count = std::min(BLOCK_SIZE, nb_nodes_ - q_start);
                
                if (update_time) {
                    f_mag.segment(q_start, q_count) = (-tij * f_nodes_invT2_.segment(q_start, q_count)).exp();
                    f_po.segment(q_start, q_count)  = f_nodes_phi_.segment(q_start, q_count) * tij;
                }

                phase_block.head(q_count) = f_po.segment(q_start, q_count)
                                            - kx * x0_ref.segment(q_start, q_count) 
                                            - ky * x1_ref.segment(q_start, q_count) 
                                            - kz * x2_ref.segment(q_start, q_count);

                fourier_block.head(q_count).array().real() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).sin();

                // Key difference here: f_M_Mxy_nodes_ is used instead of f_Mxy_nodes_
                s.noalias() += fourier_block.head(q_count).transpose() * f_M_Mxy_nodes_.middleRows(q_start, q_count);
            }
            
            kspace_mat.row(row) = s;
        }

        return Eigen::TensorMap<Tensor4CR>(kspace_mat.data(), nb_meas, nb_lines, nb_kz, (uint)nv_);
    }

    // =========================================================================
    // STANDARD QUADRATURE INTEGRATION SIGNAL GENERATOR (FULL)
    // =========================================================================
    // Reverted to pure Eigen::Tensor signatures for broad compatibility.
    Tensor4CR signal_full(
        const std::vector<Tensor3> &kloc, 
        const Tensor3 &t,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_x,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_y,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_z,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& weights,
        bool has_traj)
    {
        return signal(kloc, t, modes_x, modes_y, modes_z, weights, has_traj); // Logic is identical to signal(), wrapped for compatibility
    }

    // =========================================================================
    // STANDARD QUADRATURE INTEGRATION SIGNAL GENERATOR
    // =========================================================================
    // This evaluates the signal purely at Quadrature Points (highest accuracy, but slower)
    // Reverted to native C++ Eigen::Tensors.
    // Includes an explicit sparse-dense matrix multiplication step to project the fast
    // nodal displacements onto the exact quadrature evaluation points.
    Tensor4CR signal(
        const std::vector<Tensor3> &kloc,
        const Tensor3 &t,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_x,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_y,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_z,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& weights,
        bool has_traj)
    {
#ifdef FEELMRI_GPU
        if (device_ == "gpu") {
            return signal_quadrature_gpu_impl(kloc, t, modes_x, modes_y,
                                                modes_z, weights, has_traj);
        }
#endif
        const C i1(T(0), T(1));
        const T two_pi  = T(2) * T(M_PI);

        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S = nb_meas * nb_lines * nb_kz;

        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kspace_mat(S, nv_);
        Eigen::RowVector<C, Eigen::Dynamic> s(nv_);

        T t_old = T(-1);

        const int BLOCK_SIZE = 8192;
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);

        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(total_q_);
        Eigen::Array<T, Eigen::Dynamic, 1> f_po(total_q_);

        // Reference pointers dynamically switch based on trajectory presence
        Eigen::Array<T, Eigen::Dynamic, 1>& x0_ref = has_traj ? f_dyn_xq0_ : f_xq0_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x1_ref = has_traj ? f_dyn_xq1_ : f_xq1_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x2_ref = has_traj ? f_dyn_xq2_ : f_xq2_;

        for (uint i = 0, row = 0; i < nb_meas; ++i)
        for (uint j = 0; j < nb_lines; ++j)
        for (uint k = 0; k < nb_kz; ++k, ++row)
        {
            const T tij = t(i, j, k); 
            const T kx  = two_pi * kloc[0](i, j, k);
            const T ky  = two_pi * kloc[1](i, j, k);
            const T kz  = two_pi * kloc[2](i, j, k);

            const bool update_time = (tij != t_old);

            if (has_traj && update_time) 
            {
                auto w = weights.row(row).transpose(); 
                
                // 1. Compute nodal displacements using the fast AVX2 GEMV
                Eigen::Matrix<T, Eigen::Dynamic, 1> disp_x = modes_x * w;
                Eigen::Matrix<T, Eigen::Dynamic, 1> disp_y = modes_y * w;
                Eigen::Matrix<T, Eigen::Dynamic, 1> disp_z = modes_z * w;

                // 2. Map from Node displacements to Quadrature Point displacements
                // Matrix multiply: Projection Matrix (Q x N) * Nodal Disp (N x 3) = Quad Disp (Q x 3)
                // This updates the quadrature point coordinates for the integration loop natively in C++.
                f_dyn_xq0_ = f_xq0_ + (S_global_ * disp_x).array();
                f_dyn_xq1_ = f_xq1_ + (S_global_ * disp_y).array();
                f_dyn_xq2_ = f_xq2_ + (S_global_ * disp_z).array();
            }

            s.setZero();
            if (update_time) t_old = tij;

            // Iterate through every single quadrature point globally
            for (int q_start = 0; q_start < total_q_; q_start += BLOCK_SIZE)
            {
                const int q_count = std::min(BLOCK_SIZE, total_q_ - q_start);
                
                // Key difference here: T2 decay is pre-multiplied by the quadrature weight (f_wq_)
                if (update_time) {
                    f_mag.segment(q_start, q_count) = f_wq_.segment(q_start, q_count) * (-tij * f_invT2_.segment(q_start, q_count)).exp();
                    f_po.segment(q_start, q_count)  = f_phi_.segment(q_start, q_count) * tij;
                }

                phase_block.head(q_count) = f_po.segment(q_start, q_count) 
                                            - kx * x0_ref.segment(q_start, q_count) 
                                            - ky * x1_ref.segment(q_start, q_count) 
                                            - kz * x2_ref.segment(q_start, q_count);

                fourier_block.head(q_count).array().real() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).sin();

                // Multiply by magnetization evaluated at quadrature points (f_Mxy_)
                s.noalias() += fourier_block.head(q_count).transpose() * f_Mxy_.middleRows(q_start, q_count);
            }
            
            kspace_mat.row(row) = s;
        }

        return Eigen::TensorMap<Tensor4CR>(kspace_mat.data(), nb_meas, nb_lines, nb_kz, (uint)nv_);
    }

private:
    // Internal variables defining the mesh and physics setup
    Eigen::MatrixXi elems_;    
    int nelem_, nv_, nq_, total_q_, nb_nodes_;
    FEQuadratureCache<T> cache_;

    // S_global_ maps Nodal values -> Quadrature values
    Eigen::SparseMatrix<T, Eigen::RowMajor> S_global_;

    // Quadrature Point parameters (Static / Weights / B0 / T2 / Dynamic)
    Eigen::Array<T, Eigen::Dynamic, 1> f_xq0_, f_xq1_, f_xq2_, f_wq_;
    Eigen::Array<T, Eigen::Dynamic, 1> f_invT2_, f_phi_;
    Eigen::Array<T, Eigen::Dynamic, 1> f_dyn_xq0_, f_dyn_xq1_, f_dyn_xq2_;    

    // Nodal Point parameters
    Eigen::Array<T, Eigen::Dynamic, 1> f_nodes_x0_, f_nodes_x1_, f_nodes_x2_;
    Eigen::Array<T, Eigen::Dynamic, 1> f_dyn_nodes_x0_, f_dyn_nodes_x1_, f_dyn_nodes_x2_;
    Eigen::Array<T, Eigen::Dynamic, 1> f_nodes_invT2_, f_nodes_phi_;
    
    // Magnetization vectors matrices
    Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_M_Mxy_nodes_;
    Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_Mxy_nodes_;
    Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_Mxy_;

    // Compute backend selector. Controls whether signal_sum / signal_nodal
    // and signal / signal_full dispatch to the host AVX2 path (default)
    // or the CUDA kernel.
    std::string device_ = "cpu";

    // Cached quadrature-projected modes. Recomputed only when the caller-
    // supplied modes_x.data() pointer or n_modes changes. POD modes are
    // static across one Phantom.mri_signal loop (~100 calls in a 4D-flow
    // run), so this avoids re-doing the O(Q x M) sparse-times-dense
    // projection — which would otherwise allocate multi-GB per call at
    // large meshes (e.g. ~60 GB total host alloc at N = 600k, horder = 4).
    const T* cached_modes_x_ptr_ = nullptr;
    const T* cached_modes_y_ptr_ = nullptr;
    const T* cached_modes_z_ptr_ = nullptr;
    int      cached_n_modes_     = 0;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        cached_modes_q_x_, cached_modes_q_y_, cached_modes_q_z_;

#ifdef FEELMRI_GPU
    // Run signal() / signal_full() (quadrature integration) on the GPU.
    // The kernel itself is the same one used by signal_sum / signal_nodal
    // — we just feed it quadrature-point arrays instead of node arrays:
    //   - "nodes" become the static quadrature coordinates f_xq*_.
    //   - "modes" become the projected modes S_global_ * modes_*.
    //   - "Mxy_nodes" becomes f_wq_-weighted f_Mxy_ (so the weight is
    //     folded into the magnetisation; the magnitude inside the
    //     kernel stays the same `exp(-tij/T2)`).
    //   - invT2 / phi become their quadrature-interpolated f_invT2_,
    //     f_phi_ (already populated by set_static_fields).
    Tensor4CR signal_quadrature_gpu_impl(
        const std::vector<Tensor3>& kloc,
        const Tensor3& t,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_x,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_y,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_z,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& weights,
        bool has_traj)
    {
        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S        = nb_meas * nb_lines * nb_kz;

        // Flatten kloc / t into iteration-order arrays (i outer, k inner).
        std::vector<T> flat_kx(S), flat_ky(S), flat_kz(S), flat_t(S);
        for (int i = 0, row = 0; i < nb_meas; ++i)
        for (int j = 0; j < nb_lines; ++j)
        for (int k = 0; k < nb_kz;    ++k, ++row) {
            flat_kx[row] = kloc[0](i, j, k);
            flat_ky[row] = kloc[1](i, j, k);
            flat_kz[row] = kloc[2](i, j, k);
            flat_t[row]  = t(i, j, k);
        }

        // Project modes nodes -> quadrature points. Cache the RowMajor
        // result on the assembler keyed by (host pointer, n_modes); the
        // recompute happens on the first call and on the rare cases
        // where the caller swaps the POD modes object. At scale this
        // saves ~30 GB/call host allocation + the matching H2D upload.
        const int n_modes = has_traj ? static_cast<int>(modes_x.cols()) : 0;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            rm_weights;
        if (has_traj) {
            const bool cache_stale = (modes_x.data() != cached_modes_x_ptr_)
                                  || (modes_y.data() != cached_modes_y_ptr_)
                                  || (modes_z.data() != cached_modes_z_ptr_)
                                  || (n_modes        != cached_n_modes_);
            if (cache_stale) {
                cached_modes_q_x_ = S_global_ * modes_x;
                cached_modes_q_y_ = S_global_ * modes_y;
                cached_modes_q_z_ = S_global_ * modes_z;
                cached_modes_x_ptr_ = modes_x.data();
                cached_modes_y_ptr_ = modes_y.data();
                cached_modes_z_ptr_ = modes_z.data();
                cached_n_modes_     = n_modes;
            }
            rm_weights = weights;
        }

        // Output buffer: RowMajor (S, nv).
        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            kspace_mat(S, nv_);

        // Fused projection + signal kernel. The host no longer materialises
        // f_Mxy_ at quadrature points: we hand the GPU the per-node Mxy
        // (already cached on the assembler by update_magnetization()) plus
        // the S_global_ CSR triple, and the projection runs on the device.
        const int rc = feelmri_mri_signal_with_projection_gpu_f32(
            f_xq0_.data(), f_xq1_.data(), f_xq2_.data(),
            f_invT2_.data(), f_phi_.data(),
            f_Mxy_nodes_.data(),
            nb_nodes_,
            S_global_.outerIndexPtr(),
            S_global_.innerIndexPtr(),
            S_global_.valuePtr(),
            static_cast<int>(S_global_.nonZeros()),
            f_wq_.data(),
            has_traj ? cached_modes_q_x_.data() : nullptr,
            has_traj ? cached_modes_q_y_.data() : nullptr,
            has_traj ? cached_modes_q_z_.data() : nullptr,
            has_traj ? rm_weights.data() : nullptr,
            has_traj ? 1 : 0,
            n_modes, total_q_, nv_,
            flat_kx.data(), flat_ky.data(), flat_kz.data(), flat_t.data(),
            S,
            kspace_mat.data());

        if (rc != 0) {
            const char* msg = feelmri_device_last_error_string();
            throw std::runtime_error(
                std::string("SignalAssembler GPU quadrature path: device "
                            "runtime reported error: ")
                + (msg && msg[0] ? msg : "unknown"));
        }

        return Eigen::TensorMap<Tensor4CR>(
            kspace_mat.data(), nb_meas, nb_lines, nb_kz, (uint)nv_);
    }

    // Run signal_sum (use_M_Mxy = false) or signal_nodal (use_M_Mxy = true)
    // on the GPU. Mirrors the host math exactly; the only host-side cost
    // is repacking inputs into the row-major flat layout the kernel expects.
    Tensor4CR signal_node_gpu_impl(
        const std::vector<Tensor3>& kloc,
        const Tensor3& t,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_x,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_y,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_z,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& weights,
        bool has_traj,
        bool use_M_Mxy)
    {
        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S        = nb_meas * nb_lines * nb_kz;

        // Flatten kloc / t into iteration-order arrays (i outer, k inner).
        std::vector<T> flat_kx(S), flat_ky(S), flat_kz(S), flat_t(S);
        for (int i = 0, row = 0; i < nb_meas; ++i)
        for (int j = 0; j < nb_lines; ++j)
        for (int k = 0; k < nb_kz;    ++k, ++row) {
            flat_kx[row] = kloc[0](i, j, k);
            flat_ky[row] = kloc[1](i, j, k);
            flat_kz[row] = kloc[2](i, j, k);
            flat_t[row]  = t(i, j, k);
        }

        // Row-major (n_nodes, n_modes) layout for modes; (n_samples, n_modes)
        // for weights. The Python caller already passes C-contiguous numpy,
        // but pybind11 + Eigen::Matrix copies into ColMajor on the way in,
        // so we have to swap back here.
        const int n_modes = has_traj ? static_cast<int>(modes_x.cols()) : 0;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            rm_modes_x, rm_modes_y, rm_modes_z, rm_weights;
        if (has_traj) {
            rm_modes_x = modes_x;
            rm_modes_y = modes_y;
            rm_modes_z = modes_z;
            rm_weights = weights;
        }

        // Pick the magnetisation buffer (f_Mxy_nodes_ for signal_sum,
        // f_M_Mxy_nodes_ for signal_nodal). Both are stored RowMajor on
        // the class — no repack needed.
        const auto& Mxy_src = use_M_Mxy ? f_M_Mxy_nodes_ : f_Mxy_nodes_;
        const std::complex<T>* Mxy_ptr = Mxy_src.data();

        // Output buffer: RowMajor (S, nv).
        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            kspace_mat(S, nv_);

        const int rc = feelmri_mri_signal_gpu_f32(
            f_nodes_x0_.data(), f_nodes_x1_.data(), f_nodes_x2_.data(),
            f_nodes_invT2_.data(), f_nodes_phi_.data(),
            Mxy_ptr,
            has_traj ? rm_modes_x.data() : nullptr,
            has_traj ? rm_modes_y.data() : nullptr,
            has_traj ? rm_modes_z.data() : nullptr,
            has_traj ? rm_weights.data() : nullptr,
            has_traj ? 1 : 0,
            n_modes, nb_nodes_, nv_,
            flat_kx.data(), flat_ky.data(), flat_kz.data(), flat_t.data(),
            S,
            kspace_mat.data());

        if (rc != 0) {
            const char* msg = feelmri_device_last_error_string();
            throw std::runtime_error(
                std::string("SignalAssembler GPU path: device runtime reported "
                            "error: ") + (msg && msg[0] ? msg : "unknown"));
        }

        return Eigen::TensorMap<Tensor4CR>(
            kspace_mat.data(), nb_meas, nb_lines, nb_kz, (uint)nv_);
    }
#endif  // FEELMRI_GPU
};


// =============================================================================
// PYBIND11 MODULE BINDINGS
// =============================================================================
// This macro creates the shared library entry point that Python imports
PYBIND11_MODULE(MRIAssemble, m)
{
    m.doc() = "Highly optimized MRI Finite Element Assembly Module";

    using T = float; // Matches the Python float32 arrays
    using Assembler = SignalAssembler<T>;

    // Bind the core Signal Assembler routines to Python space
    py::class_<Assembler>(m, "SignalAssembler")
        // Expose Constructor arguments
        // Reverted standard const references for signature binding.
        .def(py::init<const Eigen::MatrixXi&,
                      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&,
                      const std::string&,
                      int>(),
             py::arg("elems"),
             py::arg("nodes"),
             py::arg("meshio_type"),
             py::arg("quadrature_degree"))
        
        // Expose Static Fields setup
        .def("set_static_fields", &Assembler::set_static_fields,
             py::arg("T2"), 
             py::arg("phi_dB0"))
        
        // Expose Magnetization setups
        .def("update_magnetization", &Assembler::update_magnetization,
             py::arg("Mxy"))

        .def("update_full_magnetization", &Assembler::update_full_magnetization,
             py::arg("Mxy"))

        .def("update_nodal_magnetization", &Assembler::update_nodal_magnetization,
             py::arg("M"), py::arg("Mxy"),
             "Pre-computes the M * Mxy product for fast nodal Galerkin projection.")

        .def("estimate_element_sizes", &Assembler::estimate_element_sizes,
             "Returns the characteristic 3D length of each element based on its Jacobian volume.")

        // Switch this assembler between the CPU and GPU compute paths.
        // GPU support requires a build with FEELMRI_ENABLE_GPU=ON.
        .def("set_device", &Assembler::set_device, py::arg("device"),
             "Select 'cpu' (default, AVX2) or 'gpu' (CUDA) for signal_sum / signal_nodal.")
        .def_property_readonly("device", &Assembler::device)

        // Expose signal integration methods, mapped to the new mode/weight signatures.
        .def("signal_sum", &Assembler::signal_sum,
             py::arg("kloc"), py::arg("t"),
             py::arg("modes_x"), py::arg("modes_y"), py::arg("modes_z"), 
             py::arg("weights"), py::arg("has_traj"),
             "Simulate MRI k-space signal summing over nodes with pre-computed POD fields.")

        .def("signal_full", &Assembler::signal_full,
             py::arg("kloc"), py::arg("t"),
             py::arg("modes_x"), py::arg("modes_y"), py::arg("modes_z"), 
             py::arg("weights"), py::arg("has_traj"),
             "Simulate MRI k-space signal over quadrature points with pre-computed POD fields.")

        .def("signal", &Assembler::signal,
             py::arg("kloc"), py::arg("t"),
             py::arg("modes_x"), py::arg("modes_y"), py::arg("modes_z"), 
             py::arg("weights"), py::arg("has_traj"),
             "Simulate MRI k-space signal over quadrature points with pre-computed POD fields.")

        .def("signal_nodal", &Assembler::signal_nodal,
             py::arg("kloc"), py::arg("t"),
             py::arg("modes_x"), py::arg("modes_y"), py::arg("modes_z"), 
             py::arg("weights"), py::arg("has_traj"),
             "Simulate MRI k-space signal using ultra-fast nodal mass matrix integration.");
}
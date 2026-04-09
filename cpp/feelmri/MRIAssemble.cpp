#include <vector>
#include <complex>
#include <cmath>
#include <FEUtils.h>
#include <Eigen/Sparse>

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// =============================================================================
// STATEFUL MRI SIGNAL ASSEMBLER
// =============================================================================
// This class precomputes and caches Finite Element (FE) spatial characteristics
// (quadrature points, shape functions, element volumes) to accelerate the 
// evaluation of the MRI signal equation. It utilizes L1-cache blocking, AVX 
// SIMD vectorization, and Sparse Matrix algebra to eliminate memory bottlenecks.
template <typename T>
class SignalAssembler
{
public:
    using C = std::complex<T>;
    using Tensor3 = Eigen::Tensor<T, 3>; 
    using Tensor4CR = Eigen::Tensor<C, 4, Eigen::RowMajor>;

    SignalAssembler(
        const Eigen::MatrixXi &elems,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &nodes,
        const std::string& meshio_type,
        int quadrature_degree)
      : elems_(elems)
    {
        nelem_ = elems_.rows();
        cache_ = BuildFEQuadratureCache<T>(elems_, nodes, meshio_type, quadrature_degree);

        // Store the total number of nodes in the mesh
        nb_nodes_ = nodes.rows();

        // Extract and store static nodal coordinates
        f_nodes_x0_ = nodes.col(0).array();
        f_nodes_x1_ = nodes.col(1).array();
        f_nodes_x2_ = nodes.col(2).array();
        
        // Allocate buffers for dynamically moving nodal coordinates
        f_dyn_nodes_x0_.resize(nb_nodes_);
        f_dyn_nodes_x1_.resize(nb_nodes_);
        f_dyn_nodes_x2_.resize(nb_nodes_);

        // Calculate total number of quadrature points across the entire domain
        nq_ = (int)cache_.wq[0].size();
        total_q_ = nelem_ * nq_;

        // Allocate flattened, contiguous arrays for static quadrature geometry
        f_xq0_.resize(total_q_);
        f_xq1_.resize(total_q_);
        f_xq2_.resize(total_q_);
        f_wq_.resize(total_q_);

        // Allocate flattened buffers for dynamically moving quadrature coordinates
        f_dyn_xq0_.resize(total_q_);
        f_dyn_xq1_.resize(total_q_);
        f_dyn_xq2_.resize(total_q_);

        // Populate the static quadrature coordinates and integration weights
        for (int e = 0; e < nelem_; ++e)
        {
            const int offset = e * nq_;
            f_xq0_.segment(offset, nq_) = cache_.xq[e].col(0).array();
            f_xq1_.segment(offset, nq_) = cache_.xq[e].col(1).array();
            f_xq2_.segment(offset, nq_) = cache_.xq[e].col(2).array();
            f_wq_.segment(offset, nq_)  = cache_.wq[e].array();
        }

        // -----------------------------------------------------------------
        // GLOBAL SPARSE INTERPOLATION MATRIX (S_global_)
        // -----------------------------------------------------------------
        // Maps (nb_nodes_ x 3) nodal displacements to (total_q_ x 3) quadrature 
        // displacements in a single, highly optimized Sparse Matrix multiplication,
        // completely bypassing the standard element-by-element nested loops.
        S_global_.resize(total_q_, nb_nodes_);
        std::vector<Eigen::Triplet<T>> triplets;
        
        const int nne = elems_.cols();
        triplets.reserve(total_q_ * nne);

        for (int e = 0; e < nelem_; ++e) {
            for (int q = 0; q < nq_; ++q) {
                const int global_q = e * nq_ + q;
                for (int a = 0; a < nne; ++a) {
                    triplets.emplace_back(global_q, elems_(e, a), cache_.SqT[e](q, a));
                }
            }
        }
        
        S_global_.setFromTriplets(triplets.begin(), triplets.end());
        S_global_.makeCompressed(); // Lock matrix for maximum multiplication speed
    }

    // Maps static physical tissue properties to the finite element mesh
    void set_static_fields(
        const Eigen::Array<T, Eigen::Dynamic, 1> &T2,
        const Eigen::Array<T, Eigen::Dynamic, 1> &phi_dB0)
    {
        Eigen::Array<T, Eigen::Dynamic, 1> inv_T2 = T2.inverse();
        const int nne = elems_.cols();

        f_invT2_.resize(total_q_);
        f_phi_.resize(total_q_);

        // Store physics at the purely nodal level for Galerkin projection
        f_nodes_invT2_ = inv_T2;
        f_nodes_phi_ = phi_dB0;

        // Project and store physics at the quadrature points for standard integration
        for (int e = 0; e < nelem_; ++e)
        {
            const int offset = e * nq_;
            
            Eigen::Vector<T, Eigen::Dynamic> invT2_e(nne);
            Eigen::Vector<T, Eigen::Dynamic> phi_e(nne);
            
            for (int a = 0; a < nne; ++a) {
                const int idx = elems_(e, a);
                invT2_e(a) = inv_T2(idx);
                phi_e(a)   = phi_dB0(idx);
            }

            f_invT2_.segment(offset, nq_) = cache_.SqT[e] * invT2_e;
            f_phi_.segment(offset, nq_)   = cache_.SqT[e] * phi_e;
        }
    }

    // Stores the initial transverse magnetization state
    void update_magnetization(const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic> &Mxy)
    {
        nv_ = (int)Mxy.cols();
        
        // Notice we do NOT project to quadrature points here.
        // Due to matrix associativity, we store the nodal magnetization directly 
        // and project the evaluated Fourier exponentials back to the nodes later.
        f_Mxy_nodes_ = Mxy; 
    }

    void update_full_magnetization(const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic> &Mxy)
    {
        nv_ = (int)Mxy.cols();
        const int nne = elems_.cols();
        
        f_Mxy_.resize(total_q_, nv_);

        for (int e = 0; e < nelem_; ++e)
        {
            const int offset = e * nq_;
            
            Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mxy_e(nne, nv_);
            for (int a = 0; a < nne; ++a) {
                Mxy_e.row(a) = Mxy.row(elems_(e, a));
            }

            Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SqTC = cache_.SqT[e].template cast<C>();
            f_Mxy_.middleRows(offset, nq_).noalias() = SqTC * Mxy_e;
        }
    }

    // Specialized initialization for the Nodal Galerkin Approximation method
    void update_nodal_magnetization(
        const Eigen::SparseMatrix<T>& M, 
        const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>& Mxy)
    {
        nv_ = (int)Mxy.cols();
        
        // Pre-compute the Mass Matrix * Magnetization product exactly once.
        // This fully absorbs the finite element shape functions and volumes into a single term.
        f_M_Mxy_nodes_.noalias() = M.template cast<C>() * Mxy;
    }

    // Calculates the characteristic length of each 3D element based on its volume
    Eigen::Vector<T, Eigen::Dynamic> estimate_element_sizes() const
    {
        Eigen::Vector<T, Eigen::Dynamic> sizes(nelem_);
        for (int e = 0; e < nelem_; ++e)
        {
            const int offset = e * nq_;
            T volume = f_wq_.segment(offset, nq_).sum();
            sizes(e) = std::cbrt(volume);
        }
        return sizes;
    }

    // =========================================================================
    // FAST NODAL INTEGRATION SIGNAL GENERATOR (SUM OF NODAL VALUES)
    // =========================================================================   
    Tensor4CR signal_sum(const std::vector<Tensor3> &kloc, 
                         const Tensor3 &t, 
                         const py::object &pod_trajectory = py::none())
    {
        const C i1(T(0), T(1));
        const T two_pi  = T(2) * T(M_PI);

        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S = nb_meas * nb_lines * nb_kz;

        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kspace_mat(S, nv_);
        Eigen::RowVector<C, Eigen::Dynamic> s(nv_);
        
        T t_old = T(-1);
        T t_last_geom_update = T(-1000.0); 
        const T GEOM_UPDATE_TOL = T(1.0e-3); // Minimum time delta (ms) to trigger a mesh geometry update
        
        const bool has_traj = !pod_trajectory.is_none();
        py::function pod_func;
        if (has_traj) pod_func = pod_trajectory.cast<py::function>();

        // Pre-allocate L1 cache sized memory blocks (8192 floats = ~32 KB)
        const int BLOCK_SIZE = 8192;
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);

        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(nb_nodes_);
        Eigen::Array<T, Eigen::Dynamic, 1> f_po(nb_nodes_);

        // Dynamic reference mapping prevents branching (if/else) inside the integration loops
        Eigen::Array<T, Eigen::Dynamic, 1>& x0_ref = has_traj ? f_dyn_nodes_x0_ : f_nodes_x0_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x1_ref = has_traj ? f_dyn_nodes_x1_ : f_nodes_x1_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x2_ref = has_traj ? f_dyn_nodes_x2_ : f_nodes_x2_;

        for (uint i = 0, row = 0; i < nb_meas; ++i)
        for (uint j = 0; j < nb_lines; ++j)
        for (uint k = 0; k < nb_kz; ++k, ++row)
        {
            const T tij = t(i, j, k); 
            const T kx  = two_pi * kloc[0](i, j, k);
            const T ky  = two_pi * kloc[1](i, j, k);
            const T kz  = two_pi * kloc[2](i, j, k);

            // Rate-limited mesh geometry evaluation
            if (has_traj && std::abs(tij - t_last_geom_update) >= GEOM_UPDATE_TOL) 
            {
                auto arr = pod_func(tij).template cast<py::array_t<T, py::array::c_style>>();
                
                if (arr.ndim() == 2 && arr.shape(0) == nb_nodes_ && arr.shape(1) == 3)
                {
                    // Complex non-rigid deformation fields
                    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>> nodes_disp(arr.data(), nb_nodes_, 3);
                    f_dyn_nodes_x0_ = f_nodes_x0_ + nodes_disp.col(0).array();
                    f_dyn_nodes_x1_ = f_nodes_x1_ + nodes_disp.col(1).array();
                    f_dyn_nodes_x2_ = f_nodes_x2_ + nodes_disp.col(2).array();
                }
                else if ((arr.ndim() == 2 && arr.shape(0) == 1 && arr.shape(1) == 3) || 
                         (arr.ndim() == 1 && arr.shape(0) == 3))
                {
                    // Rigid global translation
                    const T* t_data = arr.data();
                    f_dyn_nodes_x0_ = f_nodes_x0_ + t_data[0];
                    f_dyn_nodes_x1_ = f_nodes_x1_ + t_data[1];
                    f_dyn_nodes_x2_ = f_nodes_x2_ + t_data[2];
                }
                t_last_geom_update = tij;
            }

            s.setZero();

            // Track time evolution to avoid recalculating static T2* decay
            const bool update_time = (tij != t_old);
            if (update_time) t_old = tij;

            // Ultra-fast L1 loop evaluating exactly over the nodes
            for (int q_start = 0; q_start < nb_nodes_; q_start += BLOCK_SIZE)
            {
                const int q_count = std::min(BLOCK_SIZE, nb_nodes_ - q_start);
                
                if (update_time) {
                    f_mag.segment(q_start, q_count) = (-tij * f_nodes_invT2_.segment(q_start, q_count)).exp();
                    f_po.segment(q_start, q_count)  = f_nodes_phi_.segment(q_start, q_count) * tij;
                }

                // AVX vectorized phase computation
                phase_block.head(q_count) = -kx * x0_ref.segment(q_start, q_count) 
                                            -ky * x1_ref.segment(q_start, q_count) 
                                            -kz * x2_ref.segment(q_start, q_count) 
                                            + f_po.segment(q_start, q_count);

                // Hardware-fused sincos evaluation (exp(i * phase))
                fourier_block.head(q_count).array().real() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).sin();

                // ---------------------------------------------------------------------
                // PURE POINT-SOURCE SUMMATION
                // Multiplies the evaluated physics directly against the raw nodal 
                // magnetization, completely bypassing the Mass Matrix projection.
                // ---------------------------------------------------------------------
                s.noalias() += fourier_block.head(q_count).transpose() * f_Mxy_nodes_.middleRows(q_start, q_count);
            }
            
            kspace_mat.row(row) = s;
        }

        return Eigen::TensorMap<Tensor4CR>(kspace_mat.data(), nb_meas, nb_lines, nb_kz, (uint)nv_);
    }

    // =========================================================================
    // FAST NODAL INTEGRATION SIGNAL GENERATOR
    // =========================================================================
    // Computes the signal by evaluating physics strictly at the mesh nodes rather 
    // than quadrature points, relying on the pre-computed Nodal Mass Matrix.
    // Mathematically scales as O(N_nodes) instead of O(N_quadrature).
    Tensor4CR signal_nodal(const std::vector<Tensor3> &kloc, 
                              const Tensor3 &t, 
                              const py::object &pod_trajectory = py::none())
    {
        const C i1(T(0), T(1));
        const T two_pi  = T(2) * T(M_PI);

        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S = nb_meas * nb_lines * nb_kz;

        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kspace_mat(S, nv_);
        Eigen::RowVector<C, Eigen::Dynamic> s(nv_);
        
        T t_old = T(-1);
        T t_last_geom_update = T(-1000.0); 
        const T GEOM_UPDATE_TOL = T(1.0e-3); // Minimum time delta (ms) to trigger a mesh geometry update
        
        const bool has_traj = !pod_trajectory.is_none();
        py::function pod_func;
        if (has_traj) pod_func = pod_trajectory.cast<py::function>();

        // Pre-allocate L1 cache sized memory blocks (8192 floats = ~32 KB)
        const int BLOCK_SIZE = 8192;
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);

        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(nb_nodes_);
        Eigen::Array<T, Eigen::Dynamic, 1> f_po(nb_nodes_);

        // Dynamic reference mapping prevents branching (if/else) inside the integration loops
        Eigen::Array<T, Eigen::Dynamic, 1>& x0_ref = has_traj ? f_dyn_nodes_x0_ : f_nodes_x0_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x1_ref = has_traj ? f_dyn_nodes_x1_ : f_nodes_x1_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x2_ref = has_traj ? f_dyn_nodes_x2_ : f_nodes_x2_;

        for (uint i = 0, row = 0; i < nb_meas; ++i)
        for (uint j = 0; j < nb_lines; ++j)
        for (uint k = 0; k < nb_kz; ++k, ++row)
        {
            const T tij = t(i, j, k); 
            const T kx  = two_pi * kloc[0](i, j, k);
            const T ky  = two_pi * kloc[1](i, j, k);
            const T kz  = two_pi * kloc[2](i, j, k);

            // Rate-limited mesh geometry evaluation
            if (has_traj && std::abs(tij - t_last_geom_update) >= GEOM_UPDATE_TOL) 
            {
                auto arr = pod_func(tij).template cast<py::array_t<T, py::array::c_style>>();
                
                if (arr.ndim() == 2 && arr.shape(0) == nb_nodes_ && arr.shape(1) == 3)
                {
                    // Complex non-rigid deformation fields
                    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>> nodes_disp(arr.data(), nb_nodes_, 3);
                    f_dyn_nodes_x0_ = f_nodes_x0_ + nodes_disp.col(0).array();
                    f_dyn_nodes_x1_ = f_nodes_x1_ + nodes_disp.col(1).array();
                    f_dyn_nodes_x2_ = f_nodes_x2_ + nodes_disp.col(2).array();
                }
                else if ((arr.ndim() == 2 && arr.shape(0) == 1 && arr.shape(1) == 3) || 
                         (arr.ndim() == 1 && arr.shape(0) == 3))
                {
                    // Rigid global translation
                    const T* t_data = arr.data();
                    f_dyn_nodes_x0_ = f_nodes_x0_ + t_data[0];
                    f_dyn_nodes_x1_ = f_nodes_x1_ + t_data[1];
                    f_dyn_nodes_x2_ = f_nodes_x2_ + t_data[2];
                }
                t_last_geom_update = tij;
            }

            s.setZero();

            // Track time evolution to avoid recalculating static T2* decay
            const bool update_time = (tij != t_old);
            if (update_time) t_old = tij;

            // Ultra-fast L1 loop evaluating exactly over the nodes
            for (int q_start = 0; q_start < nb_nodes_; q_start += BLOCK_SIZE)
            {
                const int q_count = std::min(BLOCK_SIZE, nb_nodes_ - q_start);
                
                if (update_time) {
                    // Note: No spatial weights (f_wq_) are applied here because the 
                    // geometric volume weighting is already baked into f_M_Mxy_nodes_
                    f_mag.segment(q_start, q_count) = (-tij * f_nodes_invT2_.segment(q_start, q_count)).exp();
                    f_po.segment(q_start, q_count)  = f_nodes_phi_.segment(q_start, q_count) * tij;
                }

                // AVX vectorized phase computation
                phase_block.head(q_count) = -kx * x0_ref.segment(q_start, q_count) 
                                            -ky * x1_ref.segment(q_start, q_count) 
                                            -kz * x2_ref.segment(q_start, q_count) 
                                            + f_po.segment(q_start, q_count);

                // Hardware-fused sincos evaluation (exp(i * phase))
                fourier_block.head(q_count).array().real() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).sin();

                // Multiplies the evaluated nodal physics against the pre-computed Mass Matrix
                s.noalias() += fourier_block.head(q_count).transpose() * f_M_Mxy_nodes_.middleRows(q_start, q_count);
            }
            
            kspace_mat.row(row) = s;
        }

        return Eigen::TensorMap<Tensor4CR>(kspace_mat.data(), nb_meas, nb_lines, nb_kz, (uint)nv_);
    }

    // =========================================================================
    // STANDARD QUADRATURE INTEGRATION SIGNAL GENERATOR
    // =========================================================================
    // Computes the signal by evaluating physics strictly at the integration 
    // (quadrature) points to perfectly maintain high-frequency spatial dispersion.    
    Tensor4CR signal_full(const std::vector<Tensor3> &kloc, const Tensor3 &t,
                        const py::object &pod_trajectory = py::none())
    {
        const C i1(T(0), T(1));
        const T two_pi  = T(2) * T(M_PI);

        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S = nb_meas * nb_lines * nb_kz;

        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kspace_mat(S, nv_);
        Eigen::RowVector<C, Eigen::Dynamic> s(nv_);
        
        T t_old = T(-1);
        T t_last_geom_update = T(-1000.0); 
        const T GEOM_UPDATE_TOL = T(1.0e-3);
        
        const bool has_traj = !pod_trajectory.is_none();
        py::function pod_func;
        if (has_traj) pod_func = pod_trajectory.cast<py::function>();

        // Pre-allocate the global displacement matrix OUTSIDE the loop
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::ColMajor> dq_global(total_q_, 3);

        const int BLOCK_SIZE = 8192;
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);

        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(total_q_);
        Eigen::Array<T, Eigen::Dynamic, 1> f_po(total_q_);

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

            if (has_traj && std::abs(tij - t_last_geom_update) >= GEOM_UPDATE_TOL) 
            {
                auto arr = pod_func(tij).template cast<py::array_t<T, py::array::c_style>>();
                
                if (arr.ndim() == 2 && arr.shape(0) == nb_nodes_ && arr.shape(1) == 3)
                {
                    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>> nodes_disp(arr.data(), nb_nodes_, 3);
                    
                    dq_global.noalias() = S_global_ * nodes_disp;

                    f_dyn_xq0_ = f_xq0_ + dq_global.col(0).array();
                    f_dyn_xq1_ = f_xq1_ + dq_global.col(1).array();
                    f_dyn_xq2_ = f_xq2_ + dq_global.col(2).array();
                }
                else if ((arr.ndim() == 2 && arr.shape(0) == 1 && arr.shape(1) == 3) || 
                         (arr.ndim() == 1 && arr.shape(0) == 3))
                {
                    const T* t_data = arr.data();
                    for (int q = 0; q < total_q_; ++q) {
                        f_dyn_xq0_.data()[q] = f_xq0_.data()[q] + t_data[0];
                        f_dyn_xq1_.data()[q] = f_xq1_.data()[q] + t_data[1];
                        f_dyn_xq2_.data()[q] = f_xq2_.data()[q] + t_data[2];
                    }
                }
                else
                {
                    throw std::runtime_error("pod_trajectory returned invalid shape. Expected (nb_nodes, 3) or (1, 3).");
                }
                t_last_geom_update = tij;
            }

            s.setZero();

            const bool update_time = (tij != t_old);
            if (update_time) t_old = tij;

            // Identical high-performance L1 cached physics loop for Python bindings
            for (int q_start = 0; q_start < total_q_; q_start += BLOCK_SIZE)
            {
                const int q_count = std::min(BLOCK_SIZE, total_q_ - q_start);
                
                if (update_time) {
                    f_mag.segment(q_start, q_count) = f_wq_.segment(q_start, q_count) * (-tij * f_invT2_.segment(q_start, q_count)).exp();
                    f_po.segment(q_start, q_count)  = f_phi_.segment(q_start, q_count) * tij;
                }

                phase_block.head(q_count) = -kx * x0_ref.segment(q_start, q_count) 
                                            -ky * x1_ref.segment(q_start, q_count) 
                                            -kz * x2_ref.segment(q_start, q_count) 
                                            + f_po.segment(q_start, q_count);

                fourier_block.head(q_count).array().real() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).sin();

                s.noalias() += fourier_block.head(q_count).transpose() * f_Mxy_.middleRows(q_start, q_count);
            }
            
            kspace_mat.row(row) = s;
        }

        return Eigen::TensorMap<Tensor4CR>(kspace_mat.data(), nb_meas, nb_lines, nb_kz, (uint)nv_);
    }

    // =========================================================================
    // STANDARD QUADRATURE INTEGRATION SIGNAL GENERATOR
    // =========================================================================
    // Computes the signal by evaluating physics strictly at the integration 
    // (quadrature) points to perfectly maintain high-frequency spatial dispersion.
    Tensor4CR signal(const std::vector<Tensor3> &kloc, const Tensor3 &t, 
                        const py::object &pod_trajectory = py::none())
    {
        const C i1(T(0), T(1));
        const T two_pi  = T(2) * T(M_PI);

        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S = nb_meas * nb_lines * nb_kz;

        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kspace_mat(S, nv_);
        Eigen::RowVector<C, Eigen::Dynamic> s(nv_);
        
        T t_old = T(-1);
        T t_last_geom_update = T(-1000.0); 
        const T GEOM_UPDATE_TOL = T(1.0e-3);
        
        const bool has_traj = !pod_trajectory.is_none();
        py::function pod_func;
        if (has_traj) pod_func = pod_trajectory.cast<py::function>();

        // Pre-allocate the global displacement matrix OUTSIDE the loop to prevent heap allocations
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::ColMajor> dq_global(total_q_, 3);

        // Pre-allocate L1 cache sized memory blocks (8192 elements fits perfectly in CPU L1)
        const int BLOCK_SIZE = 8192;
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);

        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(total_q_);
        Eigen::Array<T, Eigen::Dynamic, 1> f_po(total_q_);

        // Dynamic reference mapping
        Eigen::Array<T, Eigen::Dynamic, 1>& x0_ref = has_traj ? f_dyn_xq0_ : f_xq0_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x1_ref = has_traj ? f_dyn_xq1_ : f_xq1_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x2_ref = has_traj ? f_dyn_xq2_ : f_xq2_;

        // Pre-allocate the full Fourier vector OUTSIDE the time loop
        // This array stores the Euler exponential for the algebraic associativity trick
        Eigen::Vector<C, Eigen::Dynamic> F_full(total_q_);

        for (uint i = 0, row = 0; i < nb_meas; ++i)
        for (uint j = 0; j < nb_lines; ++j)
        for (uint k = 0; k < nb_kz; ++k, ++row)
        {
            const T tij = t(i, j, k); 
            const T kx  = two_pi * kloc[0](i, j, k);
            const T ky  = two_pi * kloc[1](i, j, k);
            const T kz  = two_pi * kloc[2](i, j, k);

            // Rate-limited mesh geometry evaluation
            if (has_traj && std::abs(tij - t_last_geom_update) >= GEOM_UPDATE_TOL) 
            {
                auto arr = pod_func(tij).template cast<py::array_t<T, py::array::c_style>>();
                
                if (arr.ndim() == 2 && arr.shape(0) == nb_nodes_ && arr.shape(1) == 3)
                {
                    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>> nodes_disp(arr.data(), nb_nodes_, 3);
                    
                    // The Global Sparse Projection!
                    // Projects nodal displacements to all quadrature points via a single sparse multiplication.
                    dq_global.noalias() = S_global_ * nodes_disp;

                    f_dyn_xq0_ = f_xq0_ + dq_global.col(0).array();
                    f_dyn_xq1_ = f_xq1_ + dq_global.col(1).array();
                    f_dyn_xq2_ = f_xq2_ + dq_global.col(2).array();
                }
                else if ((arr.ndim() == 2 && arr.shape(0) == 1 && arr.shape(1) == 3) || 
                         (arr.ndim() == 1 && arr.shape(0) == 3))
                {
                    const T* t_data = arr.data();
                    for (int q = 0; q < total_q_; ++q) {
                        f_dyn_xq0_.data()[q] = f_xq0_.data()[q] + t_data[0];
                        f_dyn_xq1_.data()[q] = f_xq1_.data()[q] + t_data[1];
                        f_dyn_xq2_.data()[q] = f_xq2_.data()[q] + t_data[2];
                    }
                }
                else
                {
                    throw std::runtime_error("pod_trajectory returned invalid shape. Expected (nb_nodes, 3) or (1, 3).");
                }
                t_last_geom_update = tij;
            }

            s.setZero();

            const bool update_time = (tij != t_old);
            if (update_time) t_old = tij;

            // Cache-Blocked Integration Loop
            for (int q_start = 0; q_start < total_q_; q_start += BLOCK_SIZE)
            {
                const int q_count = std::min(BLOCK_SIZE, total_q_ - q_start);
                
                if (update_time) {
                    // Physics evaluations are volume weighted (f_wq_) inside this loop
                    f_mag.segment(q_start, q_count) = f_wq_.segment(q_start, q_count) * (-tij * f_invT2_.segment(q_start, q_count)).exp();
                    f_po.segment(q_start, q_count)  = f_phi_.segment(q_start, q_count) * tij;
                }

                phase_block.head(q_count) = -kx * x0_ref.segment(q_start, q_count) 
                                            -ky * x1_ref.segment(q_start, q_count) 
                                            -kz * x2_ref.segment(q_start, q_count) 
                                            + f_po.segment(q_start, q_count);

                // Single-line binary expression compiling to fused sincos AVX instruction
                fourier_block.head(q_count).array().real() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).sin();

                // Store F_full directly instead of multiplying against Mxy
                F_full.segment(q_start, q_count) = fourier_block.head(q_count);
            }
            
            // THE ALGEBRAIC ASSOCIATIVITY TRICK
            // F^T * (S * M) is mathematically equivalent to (S^T * F)^T * M
            // By projecting the Fourier vector backwards through the shape functions, 
            // we eliminate the quadrature array entirely, allowing for a tiny dense 
            // nodal matrix multiplication at the end.
            Eigen::Vector<C, Eigen::Dynamic> P = S_global_.transpose() * F_full;
            s.noalias() = P.transpose() * f_Mxy_nodes_;
            
            kspace_mat.row(row) = s;
        }

        return Eigen::TensorMap<Tensor4CR>(kspace_mat.data(), nb_meas, nb_lines, nb_kz, (uint)nv_);
    }

private:
    Eigen::MatrixXi elems_;    
    int nelem_, nv_, nq_, total_q_, nb_nodes_;
    FEQuadratureCache<T> cache_;

    // Sparse Global Interpolation Matrix
    Eigen::SparseMatrix<T, Eigen::RowMajor> S_global_;

    // Flattened Class State (Quadrature Equivalents)
    Eigen::Array<T, Eigen::Dynamic, 1> f_xq0_, f_xq1_, f_xq2_, f_wq_;
    Eigen::Array<T, Eigen::Dynamic, 1> f_invT2_, f_phi_;

    // Arrays for dynamic/moving quadrature coordinates
    Eigen::Array<T, Eigen::Dynamic, 1> f_dyn_xq0_, f_dyn_xq1_, f_dyn_xq2_;    

    // Nodal Equivalents for Mass-Matrix Integration
    Eigen::Array<T, Eigen::Dynamic, 1> f_nodes_x0_, f_nodes_x1_, f_nodes_x2_;
    Eigen::Array<T, Eigen::Dynamic, 1> f_dyn_nodes_x0_, f_dyn_nodes_x1_, f_dyn_nodes_x2_;
    Eigen::Array<T, Eigen::Dynamic, 1> f_nodes_invT2_, f_nodes_phi_;
    
    // Nodal Matrix Storage
    Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_M_Mxy_nodes_;
    Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_Mxy_nodes_; 
    Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_Mxy_; 
  };


// =============================================================================
// PYBIND11 MODULE BINDINGS
// =============================================================================
PYBIND11_MODULE(MRIAssemble, m)
{
    m.doc() = "Highly optimized MRI Finite Element Assembly Module";

    using T = float; // Matches the Python float32 arrays
    using Assembler = SignalAssembler<T>;

    // Bind the core Signal Assembler routines
    py::class_<Assembler>(m, "SignalAssembler")
        .def(py::init<const Eigen::MatrixXi&,
                      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&,
                      const std::string&,
                      int>(),
             py::arg("elems"),
             py::arg("nodes"),
             py::arg("meshio_type"),
             py::arg("quadrature_degree"))
        
        .def("set_static_fields", &Assembler::set_static_fields,
             py::arg("T2"), 
             py::arg("phi_dB0"))
        
        .def("update_magnetization", &Assembler::update_magnetization,
             py::arg("Mxy"))

        .def("update_full_magnetization", &Assembler::update_full_magnetization,
             py::arg("Mxy"))

        .def("update_nodal_magnetization", &Assembler::update_nodal_magnetization,
             py::arg("M"), py::arg("Mxy"),
             "Pre-computes the M * Mxy product for fast nodal Galerkin projection.")

        .def("estimate_element_sizes", &Assembler::estimate_element_sizes,
             "Returns the characteristic 3D length of each element based on its Jacobian volume.")

        .def("signal_sum", &Assembler::signal_sum,
             py::arg("kloc"), 
             py::arg("t"),
             py::arg("traj") = py::none(),
             "Simulate MRI k-space signal with optional moving mesh trajectory.")

        .def("signal_full", &Assembler::signal_full,
             py::arg("kloc"), 
             py::arg("t"),
             py::arg("traj") = py::none(),
             "Simulate MRI k-space signal with optional moving mesh trajectory.")

        .def("signal", &Assembler::signal,
             py::arg("kloc"), 
             py::arg("t"),
             py::arg("traj") = py::none(),
             "Simulate MRI k-space signal with optional moving mesh trajectory.")

        .def("signal_nodal", &Assembler::signal_nodal,
             py::arg("kloc"), py::arg("t"), py::arg("traj") = py::none(),
             "Simulate MRI k-space signal using ultra-fast nodal mass matrix integration.");
}
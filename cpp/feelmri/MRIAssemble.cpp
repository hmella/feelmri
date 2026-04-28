#include <vector>
#include <complex>
#include <cmath>
#include <FEUtils.h>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

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

    // Constructor initializes the mesh topology, nodes, and computes quadrature rules
    SignalAssembler(
        const Eigen::MatrixXi &elems,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &nodes,
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

    // Stores and interpolates static MRI tissue parameters (T2 relaxation, B0 off-resonance)
    void set_static_fields(
        const Eigen::Array<T, Eigen::Dynamic, 1> &T2,
        const Eigen::Array<T, Eigen::Dynamic, 1> &phi_dB0)
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

    // Updates transverse magnetization strictly at the nodes (used for fast nodal sums)
    void update_magnetization(const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic> &Mxy)
    {
        nv_ = (int)Mxy.cols(); // Number of receiving coils / isochromats
        f_Mxy_nodes_ = Mxy; 
    }

    // Updates and interpolates transverse magnetization to all quadrature points
    void update_full_magnetization(const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic> &Mxy)
    {
        nv_ = (int)Mxy.cols();
        const int nne = elems_.cols();
        
        f_Mxy_.resize(total_q_, nv_);

        // For each element, project nodal Mxy to quadrature Mxy
        for (int e = 0; e < nelem_; ++e)
        {
            const int offset = e * nq_;
            Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mxy_e(nne, nv_);
            for (int a = 0; a < nne; ++a) {
                Mxy_e.row(a) = Mxy.row(elems_(e, a));
            }
            // Cast shape functions to complex to match Mxy type, then multiply
            Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SqTC = cache_.SqT[e].template cast<C>();
            f_Mxy_.middleRows(offset, nq_).noalias() = SqTC * Mxy_e;
        }
    }

    // Pre-multiplies magnetization by a mass matrix (M) for Galerkin-style nodal integration
    void update_nodal_magnetization(
        const Eigen::SparseMatrix<T>& M, 
        const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>& Mxy)
    {
        nv_ = (int)Mxy.cols();
        // Compute Mass Matrix * Mxy to avoid doing it inside the time loop
        f_M_Mxy_nodes_.noalias() = M.template cast<C>() * Mxy;
    }

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
    Tensor4CR signal_sum(const std::vector<Tensor3> &kloc, 
                         const Tensor3 &t, 
                         const py::object &pod_trajectory = py::none())
    {
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
        
        // Check if a Python trajectory object was passed
        const bool has_traj = !pod_trajectory.is_none();
        py::function pod_func;
        if (has_traj) pod_func = pod_trajectory.cast<py::function>();

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

            // Update node positions by calling the Python POD trajectory function
            if (has_traj && update_time) 
            {
                // Call Python function, cast result to a contiguous NumPy array mapped to C++
                auto arr = pod_func(tij).template cast<py::array_t<T, py::array::c_style>>();
                // Case 1: Nodal displacement array (N_nodes x 3)
                if (arr.ndim() == 2 && arr.shape(0) == nb_nodes_ && arr.shape(1) == 3) {
                    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>> nodes_disp(arr.data(), nb_nodes_, 3);
                    f_dyn_nodes_x0_ = f_nodes_x0_ + nodes_disp.col(0).array();
                    f_dyn_nodes_x1_ = f_nodes_x1_ + nodes_disp.col(1).array();
                    f_dyn_nodes_x2_ = f_nodes_x2_ + nodes_disp.col(2).array();
                }
                // Case 2: Global displacement vector (1 x 3 or just 3)
                else if ((arr.ndim() == 2 && arr.shape(0) == 1 && arr.shape(1) == 3) || (arr.ndim() == 1 && arr.shape(0) == 3)) {
                    const T* t_data = arr.data();
                    f_dyn_nodes_x0_ = f_nodes_x0_ + t_data[0];
                    f_dyn_nodes_x1_ = f_nodes_x1_ + t_data[1];
                    f_dyn_nodes_x2_ = f_nodes_x2_ + t_data[2];
                }
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
    Tensor4CR signal_nodal(const std::vector<Tensor3> &kloc, 
                           const Tensor3 &t, 
                           const py::object &pod_trajectory = py::none())
    {
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
        
        const bool has_traj = !pod_trajectory.is_none();
        py::function pod_func;
        if (has_traj) pod_func = pod_trajectory.cast<py::function>();

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

            // Fetch and apply dynamic trajectory positions (requires Python GIL)
            if (has_traj && update_time) 
            {
                auto arr = pod_func(tij).template cast<py::array_t<T, py::array::c_style>>();
                if (arr.ndim() == 2 && arr.shape(0) == nb_nodes_ && arr.shape(1) == 3) {
                    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>> nodes_disp(arr.data(), nb_nodes_, 3);
                    f_dyn_nodes_x0_ = f_nodes_x0_ + nodes_disp.col(0).array();
                    f_dyn_nodes_x1_ = f_nodes_x1_ + nodes_disp.col(1).array();
                    f_dyn_nodes_x2_ = f_nodes_x2_ + nodes_disp.col(2).array();
                }
                else if ((arr.ndim() == 2 && arr.shape(0) == 1 && arr.shape(1) == 3) || (arr.ndim() == 1 && arr.shape(0) == 3)) {
                    const T* t_data = arr.data();
                    f_dyn_nodes_x0_ = f_nodes_x0_ + t_data[0];
                    f_dyn_nodes_x1_ = f_nodes_x1_ + t_data[1];
                    f_dyn_nodes_x2_ = f_nodes_x2_ + t_data[2];
                }
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
    Tensor4CR signal_full(const std::vector<Tensor3> &kloc, const Tensor3 &t,
                          const py::object &pod_trajectory = py::none())
    {
        return signal(kloc, t, pod_trajectory); // Logic is identical to signal(), wrapped for compatibility
    }

    // =========================================================================
    // STANDARD QUADRATURE INTEGRATION SIGNAL GENERATOR
    // =========================================================================
    // This evaluates the signal purely at Quadrature Points (highest accuracy, but slower)
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
        
        const bool has_traj = !pod_trajectory.is_none();
        py::function pod_func;
        if (has_traj) pod_func = pod_trajectory.cast<py::function>();

        // Pre-allocate temporary matrix for mapping nodal displacements to quadrature points
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::ColMajor> dq_global(total_q_, 3);

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
                auto arr = pod_func(tij).template cast<py::array_t<T, py::array::c_style>>();
                // Map from Node displacements to Quadrature Point displacements
                if (arr.ndim() == 2 && arr.shape(0) == nb_nodes_ && arr.shape(1) == 3) {
                    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>> nodes_disp(arr.data(), nb_nodes_, 3);
                    // Matrix multiply: Projection Matrix (Q x N) * Nodal Disp (N x 3) = Quad Disp (Q x 3)
                    dq_global.noalias() = S_global_ * nodes_disp;
                    f_dyn_xq0_ = f_xq0_ + dq_global.col(0).array();
                    f_dyn_xq1_ = f_xq1_ + dq_global.col(1).array();
                    f_dyn_xq2_ = f_xq2_ + dq_global.col(2).array();
                }
                // Global flat displacement applies universally to all quadrature points
                else if ((arr.ndim() == 2 && arr.shape(0) == 1 && arr.shape(1) == 3) || (arr.ndim() == 1 && arr.shape(0) == 3)) {
                    const T* t_data = arr.data();
                    for (int q = 0; q < total_q_; ++q) {
                        f_dyn_xq0_.data()[q] = f_xq0_.data()[q] + t_data[0];
                        f_dyn_xq1_.data()[q] = f_xq1_.data()[q] + t_data[1];
                        f_dyn_xq2_.data()[q] = f_xq2_.data()[q] + t_data[2];
                    }
                }
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

        // Expose signal integration methods, linking kwargs and default args (py::none)
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
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
      // Reorder to Basix DOF order once: elems_ then serves both the geometry map
      // (via the quadrature cache) and the global assembly indices below.
      : elems_(permute_meshio_to_basix(elems, meshio_type))
    {
        nelem_ = elems_.rows(); // Number of elements in the mesh
        
        // Precompute finite element characteristics (weights, points, shape functions)
        cache_ = BuildFEQuadratureCache<T>(elems_, nodes, meshio_type, quadrature_degree);

        nb_nodes_ = nodes.rows(); // Total number of nodes in the mesh

        // Every node counts until told otherwise, which reproduces the serial
        // result exactly and leaves single-rank runs unchanged.
        f_node_owned_ = Eigen::Array<T, Eigen::Dynamic, 1>::Ones(nb_nodes_);

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

    // Quadrature-space POD modes: Phi_q = S_global_ * Phi, cached.
    //
    // The quadrature signal path needs the displacement AT quadrature points,
    // which it used to obtain per k-space sample as
    //     S_global_ * (modes * w)      -- a dense GEMV plus a sparse projection
    // Folding the projection into the modes once turns that into a single dense
    // GEMV per sample, and makes the data layout identical to the nodal path so
    // the same cache blocking applies. Rebuilt only when the caller passes a
    // different modes array (pointer + shape fingerprint).
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mq_x_, mq_y_, mq_z_;
    const void* mq_key_ = nullptr;
    int mq_cols_ = -1;

    void ensure_quadrature_modes(
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_x,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_y,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& modes_z)
    {
        const void* key = static_cast<const void*>(modes_x.data());
        if (key == mq_key_ && modes_x.cols() == mq_cols_) return;
        mq_x_.noalias() = S_global_ * modes_x;
        mq_y_.noalias() = S_global_ * modes_y;
        mq_z_.noalias() = S_global_ * modes_z;
        mq_key_ = key;
        mq_cols_ = (int)modes_x.cols();
    }

    // Updates transverse magnetization strictly at the nodes (used for fast nodal sums).
    //
    // The projection to quadrature points is deferred to
    // ensure_full_magnetization(), which the quadrature paths call on demand: it
    // costs O(total_q * nne * nv) and the nodal paths never read f_Mxy_.
    void update_magnetization(const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>& Mxy)
    {
        nv_ = (int)Mxy.cols(); // Number of receiving coils / isochromats
        f_Mxy_nodes_ = Mxy;
        f_Mxy_dirty_ = true;
    }

    // Projects the stored nodal magnetization onto the quadrature points if it has
    // changed since the last projection. Repeated k-space calls against a single
    // magnetization update therefore project once.
    void ensure_full_magnetization()
    {
        if (!f_Mxy_dirty_) return;

        const int nne = elems_.cols();
        f_Mxy_.resize(total_q_, nv_);

        // For each element, project nodal Mxy to quadrature Mxy
        for (int e = 0; e < nelem_; ++e)
        {
            const int offset = e * nq_;
            Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mxy_e(nne, nv_);
            for (int a = 0; a < nne; ++a) {
                Mxy_e.row(a) = f_Mxy_nodes_.row(elems_(e, a));
            }
            // Cast shape functions to complex to match Mxy type, then multiply
            Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SqTC = cache_.SqT[e].template cast<C>();
            f_Mxy_.middleRows(offset, nq_).noalias() = SqTC * Mxy_e;
        }

        f_Mxy_dirty_ = false;
    }

    // Eager form of the above, for callers that want the projection cost up front.
    void update_full_magnetization(const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>& Mxy)
    {
        update_magnetization(Mxy);
        ensure_full_magnetization();
    }

    // Marks which nodes this rank is responsible for in the raw nodal sum.
    //
    // A rank owns every node its elements touch, so nodes on a partition boundary
    // belong to several ranks. signal_sum is a plain sum over local nodes and the
    // caller reduces it with MPI_SUM, so without this mask interface nodes are
    // counted once per owning rank. signal_nodal and the quadrature paths are
    // unaffected: their mass matrix and elements are owned exclusively.
    void set_node_ownership(const Eigen::Array<T, Eigen::Dynamic, 1>& owned)
    {
        if (owned.size() != nb_nodes_)
            throw std::runtime_error(
                "set_node_ownership: expected " + std::to_string(nb_nodes_)
                + " entries, got " + std::to_string(owned.size()));
        f_node_owned_ = owned;
    }

    // Pre-multiplies magnetization by a mass matrix (M) for Galerkin-style nodal integration.
    void update_nodal_magnetization(
        const Eigen::SparseMatrix<T>& M, 
        const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>& Mxy)
    {
        nv_ = (int)Mxy.cols();
        // Keep the nodal copy so a quadrature call on this group projects the
        // current magnetization.
        f_Mxy_nodes_ = Mxy;
        f_Mxy_dirty_ = true;
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
        
        // Cache-blocking parameters to ensure arrays fit in CPU L1/L2 cache
        const int BLOCK_SIZE = 8192;
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);
        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(BLOCK_SIZE), f_po(BLOCK_SIZE);
        Eigen::Array<T, Eigen::Dynamic, 1> dx0(BLOCK_SIZE), dx1(BLOCK_SIZE), dx2(BLOCK_SIZE);

        // Flatten the per-sample scalars once; S is small (one readout).
        Eigen::Array<T, Eigen::Dynamic, 1> tv(S), kxv(S), kyv(S), kzv(S);
        for (uint i = 0, row = 0; i < nb_meas; ++i)
        for (uint j = 0; j < nb_lines; ++j)
        for (uint k = 0; k < nb_kz; ++k, ++row) {
            tv(row)  = t(i, j, k);
            kxv(row) = two_pi * kloc[0](i, j, k);
            kyv(row) = two_pi * kloc[1](i, j, k);
            kzv(row) = two_pi * kloc[2](i, j, k);
        }

        kspace_mat.setZero();

        // Node blocks outermost, k-points innermost.
        //
        // With k-points outermost the whole (n_nodes x n_modes) mode matrix is
        // re-streamed from memory for every k-space sample, since a Cartesian
        // readout gives each sample its own acquisition time and the `update_time`
        // guard always fires. With the block outermost, a block's modes
        // (BLOCK_SIZE x n_modes) stay resident across the whole readout.
        //
        // Block order is unchanged, so each output row accumulates its block
        // contributions in the same sequence and the result is bit-identical.
        for (int q_start = 0; q_start < nb_nodes_; q_start += BLOCK_SIZE)
        {
            const int q_count = std::min(BLOCK_SIZE, nb_nodes_ - q_start);

            auto x0b    = f_nodes_x0_.segment(q_start, q_count);
            auto x1b    = f_nodes_x1_.segment(q_start, q_count);
            auto x2b    = f_nodes_x2_.segment(q_start, q_count);
            auto invT2b = f_nodes_invT2_.segment(q_start, q_count);
            auto phib   = f_nodes_phi_.segment(q_start, q_count);
            auto ownb   = f_node_owned_.segment(q_start, q_count);

            T t_old = T(-1); // per block: the k-point walk restarts here

            for (int row = 0; row < S; ++row)
            {
                const T tij = tv(row), kx = kxv(row), ky = kyv(row), kz = kzv(row);
                const bool update_time = (tij != t_old);

                if (update_time) {
                    if (has_traj) {
                        auto w = weights.row(row).transpose();
                        dx0.head(q_count) = x0b + (modes_x.middleRows(q_start, q_count) * w).array();
                        dx1.head(q_count) = x1b + (modes_y.middleRows(q_start, q_count) * w).array();
                        dx2.head(q_count) = x2b + (modes_z.middleRows(q_start, q_count) * w).array();
                    }
                    // Ownership mask: interface nodes carry weight 0 on every rank
                    // but their canonical owner.
                    f_mag.head(q_count) = ownb * (-tij * invT2b).exp();
                    f_po.head(q_count)  = phib * tij;
                    t_old = tij;
                }

                if (has_traj) {
                    phase_block.head(q_count) = f_po.head(q_count)
                                                - kx * dx0.head(q_count)
                                                - ky * dx1.head(q_count)
                                                - kz * dx2.head(q_count);
                } else {
                    phase_block.head(q_count) = f_po.head(q_count)
                                                - kx * x0b - ky * x1b - kz * x2b;
                }

                fourier_block.head(q_count).array().real() = f_mag.head(q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.head(q_count) * phase_block.head(q_count).sin();

                kspace_mat.row(row).noalias() +=
                    fourier_block.head(q_count).transpose() * f_Mxy_nodes_.middleRows(q_start, q_count);
            }
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
        
        const int BLOCK_SIZE = 8192;
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);
        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(BLOCK_SIZE), f_po(BLOCK_SIZE);
        Eigen::Array<T, Eigen::Dynamic, 1> dx0(BLOCK_SIZE), dx1(BLOCK_SIZE), dx2(BLOCK_SIZE);

        Eigen::Array<T, Eigen::Dynamic, 1> tv(S), kxv(S), kyv(S), kzv(S);
        for (uint i = 0, row = 0; i < nb_meas; ++i)
        for (uint j = 0; j < nb_lines; ++j)
        for (uint k = 0; k < nb_kz; ++k, ++row) {
            tv(row)  = t(i, j, k);
            kxv(row) = two_pi * kloc[0](i, j, k);
            kyv(row) = two_pi * kloc[1](i, j, k);
            kzv(row) = two_pi * kloc[2](i, j, k);
        }

        kspace_mat.setZero();

        // Node blocks outermost, k-points innermost -- see signal_sum. Identical
        // to that routine except the accumulation is against the mass-matrix
        // weighted magnetisation f_M_Mxy_nodes_.
        for (int q_start = 0; q_start < nb_nodes_; q_start += BLOCK_SIZE)
        {
            const int q_count = std::min(BLOCK_SIZE, nb_nodes_ - q_start);

            auto x0b    = f_nodes_x0_.segment(q_start, q_count);
            auto x1b    = f_nodes_x1_.segment(q_start, q_count);
            auto x2b    = f_nodes_x2_.segment(q_start, q_count);
            auto invT2b = f_nodes_invT2_.segment(q_start, q_count);
            auto phib   = f_nodes_phi_.segment(q_start, q_count);

            T t_old = T(-1);

            for (int row = 0; row < S; ++row)
            {
                const T tij = tv(row), kx = kxv(row), ky = kyv(row), kz = kzv(row);
                const bool update_time = (tij != t_old);

                if (update_time) {
                    if (has_traj) {
                        auto w = weights.row(row).transpose();
                        dx0.head(q_count) = x0b + (modes_x.middleRows(q_start, q_count) * w).array();
                        dx1.head(q_count) = x1b + (modes_y.middleRows(q_start, q_count) * w).array();
                        dx2.head(q_count) = x2b + (modes_z.middleRows(q_start, q_count) * w).array();
                    }
                    f_mag.head(q_count) = (-tij * invT2b).exp();
                    f_po.head(q_count)  = phib * tij;
                    t_old = tij;
                }

                if (has_traj) {
                    phase_block.head(q_count) = f_po.head(q_count)
                                                - kx * dx0.head(q_count)
                                                - ky * dx1.head(q_count)
                                                - kz * dx2.head(q_count);
                } else {
                    phase_block.head(q_count) = f_po.head(q_count)
                                                - kx * x0b - ky * x1b - kz * x2b;
                }

                fourier_block.head(q_count).array().real() = f_mag.head(q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.head(q_count) * phase_block.head(q_count).sin();

                kspace_mat.row(row).noalias() +=
                    fourier_block.head(q_count).transpose() * f_M_Mxy_nodes_.middleRows(q_start, q_count);
            }
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
        // Only path that reads f_Mxy_, so the nodal -> quadrature projection is
        // performed here rather than on every magnetization update.
        ensure_full_magnetization();

        const C i1(T(0), T(1));
        const T two_pi  = T(2) * T(M_PI);

        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S = nb_meas * nb_lines * nb_kz;

        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kspace_mat(S, nv_);
        Eigen::RowVector<C, Eigen::Dynamic> s(nv_);
        
        const int BLOCK_SIZE = 8192;
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);
        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(BLOCK_SIZE), f_po(BLOCK_SIZE);
        Eigen::Array<T, Eigen::Dynamic, 1> dx0(BLOCK_SIZE), dx1(BLOCK_SIZE), dx2(BLOCK_SIZE);

        if (has_traj) ensure_quadrature_modes(modes_x, modes_y, modes_z);

        Eigen::Array<T, Eigen::Dynamic, 1> tv(S), kxv(S), kyv(S), kzv(S);
        for (uint i = 0, row = 0; i < nb_meas; ++i)
        for (uint j = 0; j < nb_lines; ++j)
        for (uint k = 0; k < nb_kz; ++k, ++row) {
            tv(row)  = t(i, j, k);
            kxv(row) = two_pi * kloc[0](i, j, k);
            kyv(row) = two_pi * kloc[1](i, j, k);
            kzv(row) = two_pi * kloc[2](i, j, k);
        }

        kspace_mat.setZero();

        // Quadrature blocks outermost, k-points innermost -- see signal_sum for
        // the rationale. Here the win is larger because the per-sample work the
        // old order repeated was a dense GEMV *and* a sparse (Q x N) projection.
        for (int q_start = 0; q_start < total_q_; q_start += BLOCK_SIZE)
        {
            const int q_count = std::min(BLOCK_SIZE, total_q_ - q_start);

            auto x0b    = f_xq0_.segment(q_start, q_count);
            auto x1b    = f_xq1_.segment(q_start, q_count);
            auto x2b    = f_xq2_.segment(q_start, q_count);
            auto wqb    = f_wq_.segment(q_start, q_count);
            auto invT2b = f_invT2_.segment(q_start, q_count);
            auto phib   = f_phi_.segment(q_start, q_count);

            T t_old = T(-1);

            for (int row = 0; row < S; ++row)
            {
                const T tij = tv(row), kx = kxv(row), ky = kyv(row), kz = kzv(row);
                const bool update_time = (tij != t_old);

                if (update_time) {
                    if (has_traj) {
                        auto w = weights.row(row).transpose();
                        dx0.head(q_count) = x0b + (mq_x_.middleRows(q_start, q_count) * w).array();
                        dx1.head(q_count) = x1b + (mq_y_.middleRows(q_start, q_count) * w).array();
                        dx2.head(q_count) = x2b + (mq_z_.middleRows(q_start, q_count) * w).array();
                    }
                    // T2 decay pre-multiplied by the quadrature weight
                    f_mag.head(q_count) = wqb * (-tij * invT2b).exp();
                    f_po.head(q_count)  = phib * tij;
                    t_old = tij;
                }

                if (has_traj) {
                    phase_block.head(q_count) = f_po.head(q_count)
                                                - kx * dx0.head(q_count)
                                                - ky * dx1.head(q_count)
                                                - kz * dx2.head(q_count);
                } else {
                    phase_block.head(q_count) = f_po.head(q_count)
                                                - kx * x0b - ky * x1b - kz * x2b;
                }

                fourier_block.head(q_count).array().real() = f_mag.head(q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.head(q_count) * phase_block.head(q_count).sin();

                kspace_mat.row(row).noalias() +=
                    fourier_block.head(q_count).transpose() * f_Mxy_.middleRows(q_start, q_count);
            }
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
    Eigen::Array<T, Eigen::Dynamic, 1> f_node_owned_;  ///< 1 for nodes this rank owns, 0 for duplicates
    Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_Mxy_nodes_; 
    Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_Mxy_;
    bool f_Mxy_dirty_ = false;   ///< f_Mxy_ is stale w.r.t. f_Mxy_nodes_ 
};


// Number of quadrature points Basix's default rule uses for a cell type and degree.
//
// Exposed because per-element quadrature cost drives load balance in the assembler
// and no Python-side Basix is available. A hardcoded table would drift from Basix's
// own rule selection, which varies with degree.
inline int quadrature_npoints(const std::string& meshio_type, int degree)
{
    const FEInfo& fe_info = get_fe_info(meshio_type);
    auto qw = basix::quadrature::make_quadrature<double>(
        basix::quadrature::type::Default, fe_info.cell,
        basix::polyset::type::standard, degree);
    return static_cast<int>(qw[1].size());
}

// =============================================================================
// PYBIND11 MODULE BINDINGS
// =============================================================================
// This macro creates the shared library entry point that Python imports
PYBIND11_MODULE(MRIAssemble, m)
{
    m.doc() = "Highly optimized MRI Finite Element Assembly Module";

    m.def("quadrature_npoints", &quadrature_npoints,
          py::arg("meshio_type"), py::arg("degree"),
          "Number of quadrature points Basix's default rule uses for this cell type "
          "and degree. Used to weight elements by their integration cost.");

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

        .def("set_node_ownership", &Assembler::set_node_ownership,
             py::arg("owned"),
             "Per-node weight for the raw nodal sum: 1 on the rank that owns a node, "
             "0 on the other ranks that merely touch it. Prevents signal_sum from "
             "double counting interface nodes under MPI_SUM.")
        .def("update_nodal_magnetization", &Assembler::update_nodal_magnetization,
             py::arg("M"), py::arg("Mxy"),
             "Pre-computes the M * Mxy product for fast nodal Galerkin projection.")

        .def("estimate_element_sizes", &Assembler::estimate_element_sizes,
             "Returns the characteristic 3D length of each element based on its Jacobian volume.")

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
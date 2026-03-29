#include <vector>
#include <complex>
#include <cmath>
#include <FEUtils.h>
#include <OptimizedPOD.h>
#include <Eigen/Sparse>

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// 3. STATEFUL ASSEMBLER CLASS (Identical Memory layout to Signal_FEProjected)
// -----------------------------------------------------------------------------
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

        // Number of nodes
        nb_nodes_ = nodes.rows();

        // Add inside the Constructor, right after nb_nodes_ = nodes.rows();
        f_nodes_x0_ = nodes.col(0).array();
        f_nodes_x1_ = nodes.col(1).array();
        f_nodes_x2_ = nodes.col(2).array();
        f_dyn_nodes_x0_.resize(nb_nodes_);
        f_dyn_nodes_x1_.resize(nb_nodes_);
        f_dyn_nodes_x2_.resize(nb_nodes_);

        // Assume all elements of the same type have the same number of quad points
        nq_ = (int)cache_.wq[0].size();
        total_q_ = nelem_ * nq_;

        // Flattened geometry arrays
        f_xq0_.resize(total_q_);
        f_xq1_.resize(total_q_);
        f_xq2_.resize(total_q_);
        f_wq_.resize(total_q_);

        // Allocate dynamic coordinate buffers
        f_dyn_xq0_.resize(total_q_);
        f_dyn_xq1_.resize(total_q_);
        f_dyn_xq2_.resize(total_q_);

        for (int e = 0; e < nelem_; ++e)
        {
            const int offset = e * nq_;
            
            f_xq0_.segment(offset, nq_) = cache_.xq[e].col(0).array();
            f_xq1_.segment(offset, nq_) = cache_.xq[e].col(1).array();
            f_xq2_.segment(offset, nq_) = cache_.xq[e].col(2).array();
            f_wq_.segment(offset, nq_)  = cache_.wq[e].array();
        }

        // -----------------------------------------------------------------
        // BUILD THE GLOBAL SPARSE INTERPOLATION MATRIX (S_global_)
        // Maps (nb_nodes_ x 3) displacements to (total_q_ x 3) instantly
        // -----------------------------------------------------------------
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
        S_global_.makeCompressed();
    }

    void set_static_fields(
        const Eigen::Array<T, Eigen::Dynamic, 1> &T2,
        const Eigen::Array<T, Eigen::Dynamic, 1> &phi_dB0)
    {
        Eigen::Array<T, Eigen::Dynamic, 1> inv_T2 = T2.inverse();
        const int nne = elems_.cols();

        f_invT2_.resize(total_q_);
        f_phi_.resize(total_q_);

        // Add inside set_static_fields(), right before the element loop
        f_nodes_invT2_ = inv_T2;
        f_nodes_phi_ = phi_dB0;

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

    void update_magnetization(const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic> &Mxy)
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

    // Add this brand new method right after update_magnetization()
    void update_nodal_magnetization(
        const Eigen::SparseMatrix<T>& M, 
        const Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>& Mxy)
    {
        nv_ = (int)Mxy.cols();
        // Pre-compute M * Mxy exactly once and store it in the class state
        f_M_Mxy_nodes_.noalias() = M.template cast<C>() * Mxy;
    }

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

    std::shared_ptr<BaseTrajectory<T>> precompute_trajectory(std::shared_ptr<BaseTrajectory<T>> traj)
    {
        if (!traj || traj->is_global()) return traj;

        auto pod_traj = std::dynamic_pointer_cast<PODTrajectory<T>>(traj);
        if (pod_traj && !pod_traj->is_blocked()) 
        {
            py::print("Trajectory precomputation");
            const int n_modes = pod_traj->get_n_modes();
            const auto& nodal_modes = pod_traj->get_modes();
            const int nb_nodes = nodal_modes.rows() / 3;

            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> q_modes(total_q_ * 3, n_modes);
            const int nne = elems_.cols();
            Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor> mode_e(nne, 3);

            for (int m = 0; m < n_modes; ++m) {
                Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>> nodal_m(
                    nodal_modes.col(m).data(), nb_nodes, 3);
                
                // We can also use S_global_ here if we want, but since this runs exactly once, 
                // the existing element loop is perfectly fine for initialization.
                for (int e = 0; e < nelem_; ++e) {
                    for (int a = 0; a < nne; ++a) mode_e.row(a) = nodal_m.row(elems_(e, a));
                    
                    Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::ColMajor> dq = cache_.SqT[e] * mode_e;
                    const int offset = e * nq_;
                    
                    q_modes.col(m).segment(offset, nq_) = dq.col(0);
                    q_modes.col(m).segment(total_q_ + offset, nq_) = dq.col(1);
                    q_modes.col(m).segment(2 * total_q_ + offset, nq_) = dq.col(2);
                }
            }

            return std::make_shared<PODTrajectory<T>>(
                q_modes, pod_traj->get_knots(), pod_traj->get_coeffs(),
                pod_traj->get_order(), pod_traj->get_n_intervals(), n_modes,
                pod_traj->get_timeshift(), pod_traj->get_period(),
                pod_traj->is_periodic(), pod_traj->is_velocity(), 
                true 
            );
        }
        return traj;
    }

    Tensor4CR signal_nodal_py(const std::vector<Tensor3> &kloc, 
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
        const T GEOM_UPDATE_TOL = T(1.0);
        
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
                    f_dyn_nodes_x0_ = f_nodes_x0_ + nodes_disp.col(0).array();
                    f_dyn_nodes_x1_ = f_nodes_x1_ + nodes_disp.col(1).array();
                    f_dyn_nodes_x2_ = f_nodes_x2_ + nodes_disp.col(2).array();
                }
                else if ((arr.ndim() == 2 && arr.shape(0) == 1 && arr.shape(1) == 3) || 
                         (arr.ndim() == 1 && arr.shape(0) == 3))
                {
                    const T* t_data = arr.data();
                    f_dyn_nodes_x0_ = f_nodes_x0_ + t_data[0];
                    f_dyn_nodes_x1_ = f_nodes_x1_ + t_data[1];
                    f_dyn_nodes_x2_ = f_nodes_x2_ + t_data[2];
                }
                t_last_geom_update = tij;
            }

            s.setZero();

            const bool update_time = (tij != t_old);
            if (update_time) t_old = tij;

            // Ultra-fast L1 loop evaluating exactly over the nodes
            for (int q_start = 0; q_start < nb_nodes_; q_start += BLOCK_SIZE)
            {
                const int q_count = std::min(BLOCK_SIZE, nb_nodes_ - q_start);
                
                if (update_time) {
                    // Notice NO spatial weights (f_wq_) here! The volume weighting is already baked into f_M_Mxy_nodes_
                    f_mag.segment(q_start, q_count) = (-tij * f_nodes_invT2_.segment(q_start, q_count)).exp();
                    f_po.segment(q_start, q_count)  = f_nodes_phi_.segment(q_start, q_count) * tij;
                }

                phase_block.head(q_count) = -kx * x0_ref.segment(q_start, q_count) 
                                            -ky * x1_ref.segment(q_start, q_count) 
                                            -kz * x2_ref.segment(q_start, q_count) 
                                            + f_po.segment(q_start, q_count);

                // Hardware-fused sincos evaluation
                fourier_block.head(q_count).array().real() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).cos();
                fourier_block.head(q_count).array().imag() = f_mag.segment(q_start, q_count) * phase_block.head(q_count).sin();

                // Multiply against the pre-computed M * Mxy matrix!
                s.noalias() += fourier_block.head(q_count).transpose() * f_M_Mxy_nodes_.middleRows(q_start, q_count);
            }
            
            kspace_mat.row(row) = s;
        }

        return Eigen::TensorMap<Tensor4CR>(kspace_mat.data(), nb_meas, nb_lines, nb_kz, (uint)nv_);
    }

    Tensor4CR signal(const std::vector<Tensor3> &kloc, const Tensor3 &t, 
                     std::shared_ptr<BaseTrajectory<T>> raw_traj = nullptr)
    {
        const C i1(T(0), T(1));
        const T two_pi  = T(2) * T(M_PI);

        const int nb_meas  = kloc[0].dimension(0);
        const int nb_lines = kloc[0].dimension(1);
        const int nb_kz    = kloc[0].dimension(2);
        const int S = nb_meas * nb_lines * nb_kz;

        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kspace_mat(S, nv_);
        Eigen::RowVector<C, Eigen::Dynamic> s(nv_);
        
        auto traj = precompute_trajectory(raw_traj);
        const bool has_traj = (traj != nullptr);
        
        Eigen::Vector<T, Eigen::Dynamic> traj_buffer;
        if (has_traj) traj_buffer.resize(traj->output_size());

        // Pre-allocate the global displacement matrix OUTSIDE the loop
        Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::ColMajor> dq_global(total_q_, 3);

        T t_last_geom_update = T(-1000.0); 
        const T GEOM_UPDATE_TOL = T(1.0);
        T t_old = T(-1);

        const int BLOCK_SIZE = 8192; // 8192 fits cleanly in L1 cache
        Eigen::Array<T, Eigen::Dynamic, 1> phase_block(BLOCK_SIZE);
        Eigen::Matrix<C, Eigen::Dynamic, 1> fourier_block(BLOCK_SIZE);

        Eigen::Array<T, Eigen::Dynamic, 1> f_mag(total_q_);
        Eigen::Array<T, Eigen::Dynamic, 1> f_po(total_q_);

        Eigen::Array<T, Eigen::Dynamic, 1>& x0_ref = has_traj ? f_dyn_xq0_ : f_xq0_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x1_ref = has_traj ? f_dyn_xq1_ : f_xq1_;
        Eigen::Array<T, Eigen::Dynamic, 1>& x2_ref = has_traj ? f_dyn_xq2_ : f_xq2_;

        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>> nodes_disp(
            has_traj ? traj_buffer.data() : nullptr, has_traj ? nb_nodes_ : 0, 3);

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
                traj->evaluate(tij, traj_buffer.data());

                if (traj->is_global()) {
                    const T t0 = traj_buffer[0], t1 = traj_buffer[1], t2 = traj_buffer[2];
                    for (int q = 0; q < total_q_; ++q) {
                        f_dyn_xq0_.data()[q] = f_xq0_.data()[q] + t0;
                        f_dyn_xq1_.data()[q] = f_xq1_.data()[q] + t1;
                        f_dyn_xq2_.data()[q] = f_xq2_.data()[q] + t2;
                    }
                } 
                else if (auto pod = std::dynamic_pointer_cast<PODTrajectory<T>>(traj); pod && pod->is_blocked()) {
                    const T* tb = traj_buffer.data();
                    for (int q = 0; q < total_q_; ++q) {
                        f_dyn_xq0_.data()[q] = f_xq0_.data()[q] + tb[q];
                        f_dyn_xq1_.data()[q] = f_xq1_.data()[q] + tb[total_q_ + q];
                        f_dyn_xq2_.data()[q] = f_xq2_.data()[q] + tb[2 * total_q_ + q];
                    }
                }
                else {
                    // Instantly projects the nodes to all quadrature points
                    dq_global.noalias() = S_global_ * nodes_disp;

                    f_dyn_xq0_ = f_xq0_ + dq_global.col(0).array();
                    f_dyn_xq1_ = f_xq1_ + dq_global.col(1).array();
                    f_dyn_xq2_ = f_xq2_ + dq_global.col(2).array();
                }
                t_last_geom_update = tij;
            }

            s.setZero();

            const bool update_time = (tij != t_old);
            if (update_time) t_old = tij;

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

    Tensor4CR signal_py(const std::vector<Tensor3> &kloc, const Tensor3 &t, 
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
        const T GEOM_UPDATE_TOL = T(1.0);
        
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

private:
    Eigen::MatrixXi elems_;    
    int nelem_, nv_, nq_, total_q_, nb_nodes_;
    FEQuadratureCache<T> cache_;

    // Sparse Global Interpolation Matrix
    Eigen::SparseMatrix<T, Eigen::RowMajor> S_global_;

    // Flattened Class State
    Eigen::Array<T, Eigen::Dynamic, 1> f_xq0_, f_xq1_, f_xq2_, f_wq_;
    Eigen::Array<T, Eigen::Dynamic, 1> f_invT2_, f_phi_;
    Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_Mxy_;

    // Arrays for dynamic/moving quadrature coordinates
    Eigen::Array<T, Eigen::Dynamic, 1> f_dyn_xq0_, f_dyn_xq1_, f_dyn_xq2_;    

    // Nodal Equivalents for Mass-Matrix Integration
    Eigen::Array<T, Eigen::Dynamic, 1> f_nodes_x0_, f_nodes_x1_, f_nodes_x2_;
    Eigen::Array<T, Eigen::Dynamic, 1> f_dyn_nodes_x0_, f_dyn_nodes_x1_, f_dyn_nodes_x2_;
    Eigen::Array<T, Eigen::Dynamic, 1> f_nodes_invT2_, f_nodes_phi_;
    Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_M_Mxy_nodes_;
  };


// -----------------------------------------------------------------------------
// 4. PYBIND11 MODULE BINDINGS
// -----------------------------------------------------------------------------
PYBIND11_MODULE(MRIAssemble, m)
{
    m.doc() = "Highly optimized MRI Finite Element Assembly Module";

    using T = float; // Matches the Python float32 arrays
    using Assembler = SignalAssembler<T>;

    // 1. Bind the Base Trajectory
    py::class_<BaseTrajectory<T>, std::shared_ptr<BaseTrajectory<T>>>(m, "BaseTrajectory");

    // 2. Bind the POD Trajectory
    py::class_<PODTrajectory<T>, BaseTrajectory<T>, std::shared_ptr<PODTrajectory<T>>>(m, "PODTrajectory")
        .def(py::init<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&,
                      const std::vector<T>&, const std::vector<T>&, 
                      int, int, int, T, T, bool, bool>());

    // 3. Bind the Respiratory Motion Trajectory
    py::class_<RespiratoryMotionTrajectory<T>, BaseTrajectory<T>, std::shared_ptr<RespiratoryMotionTrajectory<T>>>(m, "RespiratoryMotionTrajectory")
        .def(py::init<const std::vector<T>&, const std::vector<T>&, 
                      int, int, const Eigen::Matrix<T, 3, 1>&, 
                      T, T, bool>());

    // 4. Bind the Signal Assembler
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

        .def("update_nodal_magnetization", &Assembler::update_nodal_magnetization,
             py::arg("M"), py::arg("Mxy"),
             "Pre-computes the M * Mxy product for fast nodal Galerkin projection.")

        .def("estimate_element_sizes", &Assembler::estimate_element_sizes,
             "Returns the characteristic 3D length of each element based on its Jacobian volume.")             

        .def("precompute_trajectory", &Assembler::precompute_trajectory,
             py::arg("traj"),
             "Pre-projects nodal POD modes to quadrature points for ultra-fast loop assembly.")             

        .def("signal", &Assembler::signal,
             py::arg("kloc"), 
             py::arg("t"),
             py::arg("traj") = nullptr,
             "Simulate MRI k-space signal with optional moving mesh trajectory.")

        .def("signal_py", &Assembler::signal_py,
             py::arg("kloc"), 
             py::arg("t"),
             py::arg("traj") = py::none(),
             "Simulate MRI k-space signal with optional moving mesh trajectory.")

        .def("signal_nodal_py", &Assembler::signal_nodal_py,
             py::arg("kloc"), py::arg("t"), py::arg("traj") = py::none(),
             "Simulate MRI k-space signal using ultra-fast nodal mass matrix integration.");
}
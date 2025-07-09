#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <unordered_map>
#include <tuple>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector3d;

// Type aliases
using Vec3 = Vector3d;
using Tet = std::array<int, 4>;
using Edge = std::pair<int, int>;

// Custom hash for Edge
struct EdgeHash {
    std::size_t operator()(const Edge& e) const {
        return std::hash<int>()(e.first) ^ std::hash<int>()(e.second);
    }
};

Edge get_edge_key(int i, int j) {
    return (i < j) ? std::make_pair(i, j) : std::make_pair(j, i);
}

double edge_length(const Vec3& v1, const Vec3& v2) {
    return (v1 - v2).norm();
}

int bisect_edge(std::vector<Vec3>& vertices,
                std::unordered_map<Edge, int, EdgeHash>& edge_midpoints,
                int i, int j) {
    Edge key = get_edge_key(i, j);
    auto it = edge_midpoints.find(key);
    if (it != edge_midpoints.end()) {
        return it->second;
    }
    Vec3 midpoint = 0.5 * (vertices[i] + vertices[j]);
    int idx = static_cast<int>(vertices.size());
    vertices.push_back(midpoint);
    edge_midpoints[key] = idx;
    return idx;
}

std::vector<Tet> split_tet_by_edge(std::vector<Vec3>& vertices,
                                    const Tet& tet,
                                    const Edge& edge,
                                    std::unordered_map<Edge, int, EdgeHash>& edge_midpoints) {
    int i = edge.first, j = edge.second;
    int mid_idx = bisect_edge(vertices, edge_midpoints, i, j);
    std::vector<int> others;
    for (int v : tet) {
        if (v != i && v != j) {
            others.push_back(v);
        }
    }
    return {
        Tet{mid_idx, others[0], others[1], i},
        Tet{mid_idx, others[0], others[1], j}
    };
}

// Template refinement function
template <typename Scalar>
std::tuple<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>,
           Eigen::Matrix<int, Eigen::Dynamic, 4>>
refine_mesh(const Eigen::Matrix<Scalar, Eigen::Dynamic, 3>& vertices_in,
                     const Eigen::Matrix<int, Eigen::Dynamic, 4>& tetrahedra_in,
                     Scalar max_edge_length) {
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Tet = std::array<int, 4>;
    using Edge = std::pair<int, int>;

    std::vector<Vec3> vertices;
    for (int i = 0; i < vertices_in.rows(); ++i)
        vertices.emplace_back(vertices_in.row(i));

    std::vector<Tet> tetrahedra;
    for (int i = 0; i < tetrahedra_in.rows(); ++i)
        tetrahedra.push_back(Tet{
            tetrahedra_in(i, 0),
            tetrahedra_in(i, 1),
            tetrahedra_in(i, 2),
            tetrahedra_in(i, 3)
        });

    while (true) {
        std::unordered_map<Edge, int, EdgeHash> edge_midpoints;
        std::unordered_map<int, Edge> tets_to_refine;

        for (int t_idx = 0; t_idx < static_cast<int>(tetrahedra.size()); ++t_idx) {
            const Tet& tet = tetrahedra[t_idx];
            std::array<Edge, 6> edges = {
                get_edge_key(tet[0], tet[1]), get_edge_key(tet[0], tet[2]),
                get_edge_key(tet[0], tet[3]), get_edge_key(tet[1], tet[2]),
                get_edge_key(tet[1], tet[3]), get_edge_key(tet[2], tet[3])
            };

            Scalar max_len = Scalar(0);
            int max_idx = -1;
            for (int e = 0; e < 6; ++e) {
                Scalar len = (vertices[edges[e].first] - vertices[edges[e].second]).norm();
                if (len > max_len) {
                    max_len = len;
                    max_idx = e;
                }
            }

            if (max_len > max_edge_length) {
                tets_to_refine[t_idx] = edges[max_idx];
            }
        }

        if (tets_to_refine.empty()) break;

        std::vector<Tet> new_tets;
        for (int i = 0; i < static_cast<int>(tetrahedra.size()); ++i) {
            if (tets_to_refine.count(i)) {
                const Tet& tet = tetrahedra[i];
                const Edge& edge = tets_to_refine[i];
                int i0 = edge.first, i1 = edge.second;

                auto key = get_edge_key(i0, i1);
                int mid_idx;
                auto it = edge_midpoints.find(key);
                if (it != edge_midpoints.end()) {
                    mid_idx = it->second;
                } else {
                    Vec3 midpoint = (vertices[i0] + vertices[i1]) * Scalar(0.5);
                    mid_idx = static_cast<int>(vertices.size());
                    vertices.push_back(midpoint);
                    edge_midpoints[key] = mid_idx;
                }

                std::vector<int> others;
                for (int v : tet) {
                    if (v != i0 && v != i1) {
                        others.push_back(v);
                    }
                }

                new_tets.push_back({mid_idx, others[0], others[1], i0});
                new_tets.push_back({mid_idx, others[0], others[1], i1});
            } else {
                new_tets.push_back(tetrahedra[i]);
            }
        }

        tetrahedra = std::move(new_tets);
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 3> V(vertices.size(), 3);
    for (size_t i = 0; i < vertices.size(); ++i)
        V.row(i) = vertices[i];

    Eigen::Matrix<int, Eigen::Dynamic, 4> T(tetrahedra.size(), 4);
    for (size_t i = 0; i < tetrahedra.size(); ++i)
        for (int j = 0; j < 4; ++j)
            T(i, j) = tetrahedra[i][j];

    return std::make_tuple(V, T);
}

PYBIND11_MODULE(MeshRefinement, m) {
  m.def("refine_mesh", &refine_mesh<float>,
        "Refine tetrahedral mesh (float precision)",
        py::arg("vertices"), py::arg("tetrahedra"), py::arg("max_edge_length"));

  m.def("refine_mesh", &refine_mesh<double>,
        "Refine tetrahedral mesh (double precision)",
        py::arg("vertices"), py::arg("tetrahedra"), py::arg("max_edge_length"));
}
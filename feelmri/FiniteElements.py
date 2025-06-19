import basix
import numpy as np
from basix import CellType, ElementFamily, LagrangeVariant

family_dict = {"P": ElementFamily.P, 
          "RT": ElementFamily.RT, 
          "BDM": ElementFamily.BDM, 
          "N1E": ElementFamily.N1E, 
          "N2E": ElementFamily.N2E}

cell_type_dict = {"triangle": CellType.triangle, 
             "tetrahedron": CellType.tetrahedron, 
             "quadrilateral": CellType.quadrilateral, 
             "hexahedron": CellType.hexahedron,
             "prism": CellType.prism,
             "pyramid": CellType.pyramid}

variant_dict = {"equispaced": LagrangeVariant.equispaced, 
           "legendre": LagrangeVariant.legendre, 
           "unset": LagrangeVariant.unset}

rule_dict = {
    "default": basix.QuadratureType.default,
    "gauss_jacobi": basix.QuadratureType.gauss_jacobi,
    "gll": basix.QuadratureType.gll
}

class FiniteElement:
    def __init__(self, family="P", cell_type="tetrahedron", degree=1, variant="equispaced", dtype=np.float32):
        self.family = family_dict[family]
        self.cell_type = cell_type_dict[cell_type]
        self.degree = degree
        self.variant = variant_dict[variant]
        self.dtype = dtype
        self.element = basix.create_element(self.family, self.cell_type, self.degree, self.variant)
        self.dimension = self.element.dim

    def __str__(self):
        """
        String representation of the finite element.
        
        :return: A string describing the finite element.
        """
        return f"FiniteElement(family={self.family}, cell_type={self.cell_type}, degree={self.degree}, variant={self.variant})"

    def tabulate(self, derivative_order, points):
        """
        Tabulate the basis functions at the given points.
        
        :param derivative_order: The order of the derivative to tabulate.
        :param points: The points at which to tabulate the basis functions.
        :return: The values of the basis functions at the given points.
        """
        return self.element.tabulate(derivative_order, points).astype(self.dtype)


class QuadratureRule:
    def __init__(self, cell_type="tetrahedron", order=2, rule="default", dtype=np.float32):
        """
        Create a quadrature rule for the given cell type and order.
        
        :param cell_type: The type of cell (e.g., triangle, tetrahedron).
        :param order: The order of the quadrature rule.
        :param rule: The type of quadrature rule to use.
        """
        self.cell_type = cell_type_dict[cell_type]
        self.order = order
        self.rule = rule_dict[rule]
        points, weights = basix.make_quadrature(self.cell_type, self.order, rule=self.rule)
        self.points = points.astype(dtype)
        self.weights = weights.astype(dtype)
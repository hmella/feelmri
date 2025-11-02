from typing import Literal

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
    def __init__(self, family : Literal["P", "RT", "BDM", "N1E", "N2E"] = "P", 
                 cell_type: Literal["triangle", "tetrahedron", "quadrilateral", "hexahedron", "prism", "pyramid"] = "tetrahedron", 
                 degree : int = 1, 
                 variant: Literal["equispaced", "legendre", "unset"] = "equispaced", 
                 dtype: np.dtype = np.float32):
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
      Returns
      -------
      str
          String describing the finite element.
      """
      return f"FiniteElement(family={self.family}, cell_type={self.cell_type}, degree={self.degree}, variant={self.variant})"

    def tabulate(self, derivative_order, points):
      """
      Tabulate the basis functions at the given points.
      Parameters
      ----------
      derivative_order : list
          The order of derivatives to compute.
      points : np.array
          The points at which to evaluate the basis functions.
      Returns
      -------
      np.array
          The evaluated basis functions.
      """
      return self.element.tabulate(derivative_order, points).astype(self.dtype)


class QuadratureRule:
  def __init__(self, cell_type : Literal["triangle", "tetrahedron", "quadrilateral", "hexahedron", "prism", "pyramid"] = "tetrahedron",
              order : int = 2, 
              rule : Literal["default", "gauss_jacobi", "gll"] = "default", dtype: np.dtype = np.float32):
    """
    Create a quadrature rule for the given cell type and order.
    Parameters
    ----------
    cell_type : Literal["triangle", "tetrahedron", "quadrilateral", "hexahedron", "prism", "pyramid"]
        The type of the cell.
    order : int
        The order of the quadrature rule.
    rule : Literal["default", "gauss_jacobi", "gll"]
        The type of quadrature rule.
    dtype : np.dtype = np.float32
        The data type of the points and weights.
    Returns
    -------
    points : np.array
        The quadrature points.
    weights : np.array
        The quadrature weights.
    """
    self.cell_type = cell_type_dict[cell_type]
    self.order = order
    self.rule = rule_dict[rule]
    points, weights = basix.make_quadrature(self.cell_type, self.order, rule=self.rule)
    self.points = points.astype(dtype)
    self.weights = weights.astype(dtype)
// -------------------------------
// Parameters
// -------------------------------
R = 10.0;
L = 5.0;
r = 2.0;
lc = 0.25; // fine
l  = 0.80; // coarse
numCylinders = 9;

// -------------------------------
// Big circle boundary points (z = -L/2)
// -------------------------------
Point(2) = { R, 0, -0.5*L, l};
Point(3) = { 0, R, -0.5*L, l};
Point(4) = {-R, 0, -0.5*L, l};
Point(5) = { 0,-R, -0.5*L, l};

// -------------------------------
// Small cylinder centers and circles
// -------------------------------
angle = 2 * Pi / (numCylinders - 1);

For i In {1:numCylinders}
  If (i == 1)
    xc[i] = 0;
    yc[i] = 0;
  EndIf
  If (i > 1)
    xc[i] = 2*R/3 * Cos(i*angle);
    yc[i] = 2*R/3 * Sin(i*angle);
  EndIf

  // center point of ith small circle
  Point(5+i) = {xc[i], yc[i], -0.5*L, lc};

  // 4 points on the circle
  Point(5+i*numCylinders+1) = {xc[i] + r, yc[i],     -0.5*L, lc};
  Point(5+i*numCylinders+2) = {xc[i],     yc[i] + r, -0.5*L, lc};
  Point(5+i*numCylinders+3) = {xc[i] - r, yc[i],     -0.5*L, lc};
  Point(5+i*numCylinders+4) = {xc[i],     yc[i] - r, -0.5*L, lc};

  // 4 quarter arcs
  Circle(5+i*numCylinders+1) = {5+i*numCylinders+1, 5+i, 5+i*numCylinders+2};
  Circle(5+i*numCylinders+2) = {5+i*numCylinders+2, 5+i, 5+i*numCylinders+3};
  Circle(5+i*numCylinders+3) = {5+i*numCylinders+3, 5+i, 5+i*numCylinders+4};
  Circle(5+i*numCylinders+4) = {5+i*numCylinders+4, 5+i, 5+i*numCylinders+1};
EndFor

// -------------------------------
// Big exterior circle (Point(6) is the center from i=1 -> Point(5+1))
// -------------------------------
Circle(1) = {3, 6, 2};
Circle(2) = {2, 6, 5};
Circle(3) = {5, 6, 4};
Circle(4) = {4, 6, 3};

// -------------------------------
// Extrude small cylinders + collect their volume tags
// -------------------------------
smallVols[] = {};

For i In {1:numCylinders}
  Curve Loop(10 + i) = {5 + i*numCylinders+1, 5 + i*numCylinders+2,
                        5 + i*numCylinders+3, 5 + i*numCylinders+4};
  Plane Surface(10 + i) = {10 + i};

  out[] = Extrude {0, 0, L} {
    Surface{10 + i};
    Layers{Round(L/lc)};
    Recombine;
  };

  // out[1] is the volume
  smallVols[] += { out[1] };
  Physical Volume(i+1) = { out[1] };
EndFor

// -------------------------------
// Extrude big cylinder surface with holes
// (holes are surfaces 11..(10+numCylinders))
// -------------------------------
Curve Loop(20) = {4, 1, 2, 3};

holeSurfs[] = {};
For i In {1:numCylinders}
  holeSurfs[] += {10 + i};
EndFor

Plane Surface(20) = {20, holeSurfs[]};

outBig[] = Extrude {0, 0, L} {
  Surface{20};
  Layers{Round(L/lc)};
  Recombine;
};

Physical Volume(1) = { outBig[1] };

// =====================================================
// Mesh size control: refine ONLY the small cylinder volumes
// =====================================================

// If you want the background field to dominate (recommended):
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
// Mesh.CharacteristicLengthFromCurvature = 0; // optional

// Coarse size everywhere: use MathEval (constant expression)
Field[1] = MathEval;
Field[1].F = Sprintf("%g", l);

// Restrict coarse field to the big matrix volume (optional but explicit)
Field[2] = Restrict;
Field[2].IField = 1;
Field[2].VolumesList = { outBig[1] };   // note the braces

// Fine size everywhere: use MathEval (constant expression)
Field[3] = MathEval;
Field[3].F = Sprintf("%g", lc);

// Restrict fine field to the small cylinder volumes
Field[4] = Restrict;
Field[4].IField = 3;
Field[4].VolumesList = { smallVols[] }; // IMPORTANT: braces around the array

// Combine: inside small vols -> lc, elsewhere -> l
Field[5] = Min;
Field[5].FieldsList = {2, 4};

Background Field = 5;

Transfinite Volume{:};
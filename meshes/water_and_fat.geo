// Cylinder radius and characteristic length
R = 10.0;
L = 5.0;
r = 2.0;
lc = 0.2;

// Define the number of smaller cylinders and their radius
numCylinders = 9;

// Points
Point(2) = {R, 0, -0.5*L, lc};
Point(3) = {0, R, -0.5*L, lc};
Point(4) = {-R, 0, -0.5*L, lc};
Point(5) = {0, -R, -0.5*L, lc};

// Loop to define the coordinates of the smaller cylinders
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
  Point(5+i) = {xc[i], yc[i], -0.5*L, lc}; // center
  Point(5+i*numCylinders+1) = {xc[i] + r, yc[i], -0.5*L, lc};
  Point(5+i*numCylinders+2) = {xc[i], yc[i] + r, -0.5*L, lc};
  Point(5+i*numCylinders+3) = {xc[i] - r, yc[i], -0.5*L, lc};
  Point(5+i*numCylinders+4) = {xc[i], yc[i] - r, -0.5*L, lc};
  Circle(5+i*numCylinders+1) = {5+i*numCylinders+1, 5+i, 5+i*numCylinders+2};
  Circle(5+i*numCylinders+2) = {5+i*numCylinders+2, 5+i, 5+i*numCylinders+3};
  Circle(5+i*numCylinders+3) = {5+i*numCylinders+3, 5+i, 5+i*numCylinders+4};
  Circle(5+i*numCylinders+4) = {5+i*numCylinders+4, 5+i, 5+i*numCylinders+1};
EndFor

// Add exterior circle
Circle(1) = {3, 6, 2};
Circle(2) = {2, 6, 5};
Circle(3) = {5, 6, 4};
Circle(4) = {4, 6, 3};

// Create plane surfaces for each smaller cylinder
For i In {1:numCylinders}
  Curve Loop(10 + i) = {5 + i*numCylinders+1, 5 + i*numCylinders+2, 5 + i*numCylinders+3, 5 + i*numCylinders+4};
  Plane Surface(10 + i) = {10 + i};
  volumes[] = Extrude {0, 0, L} {
    Surface{10 + i}; Layers{Round(L/lc)}; Recombine;
  };
  Physical Volume(i+1) = {volumes[1]};
EndFor

// Create plane surface for the exterior cylinder
Curve Loop(20) = {4, 1, 2, 3};
Plane Surface(20) = {20,11,12,13,14,15,16,17,18,19};  
volumes[] = Extrude {0, 0, L} {
  Surface{20}; Layers{Round(L/lc)}; Recombine;
};
Physical Volume(1) = {volumes[1]};

Transfinite Volume{:};
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>

namespace py = pybind11;

Eigen::MatrixXd hermite(Eigen::MatrixXd x, int degree)
{
    Eigen::MatrixXd out;
    Eigen::ArrayXXd xa, x2, x4, x6, x8;
    switch (degree) {
        case 0:
            out = Eigen::MatrixXd::Constant(x.rows(), x.cols(), 1.0);
            break;
        case 1:
            out = x;
            break;
        case 2:
            xa = x.array();
            x2 = xa.square();
            out = x2 - 1.0;
            break;
        case 3:
            xa = x.array();
            x2 = xa.square();
            out = xa*(x2 - 3.0);
            break;
        case 4:
            xa = x.array();
            x2 = xa.square();
            x4 = x2.square();
            out = (x4 - 6.0 * x2) + 3.0;
            break;
        case 5:
            xa = x.array();
            x2 = xa.square();
            x4 = x2.square();
            out = xa*((x4 - 10.0 * x2) + 15.0);
            break;
        case 6:
            xa = x.array();
            x2 = xa.square();
            x4 = x2.square();
            x6 = x4 * x2;
            out = ((x6 - 15 * x4) + 45 * x2) - 15;
            break;
        case 7 :
            xa = x.array();
            x2 = xa.square();
            x4 = x2.square();
            x6 = x4 * x2;
            out = xa*(((x6 - 21 * x4) + 105 * x2) - 105);
            break;
        case 8:
            xa = x.array();
            x2 = xa.square();
            x4 = x2.square();
            x6 = x4 * x2;
            x8 = x4.square();
            out = (((x8 - 28 * x6) + 210 * x4) - 420 * x2) + 105;
            break;
    }
    return out;
}

PYBIND11_MODULE(hermite, m) {
    m.doc() = "hermite function wrapped with pybind11";
    m.def("hermite", &hermite, "vectorized hermite function calculator");
}



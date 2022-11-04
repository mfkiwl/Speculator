// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0
#ifndef FASST_TYPEDEF_H
#define FASST_TYPEDEF_H
#include <Eigen/Core>
#include <vector>

typedef Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, 1> VectorMatrixXd;
typedef Eigen::Array<Eigen::MatrixXcd, Eigen::Dynamic, 1> VectorMatrixXcd;

typedef Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic>
    ArrayMatrixXd;
typedef Eigen::Array<Eigen::MatrixXcd, Eigen::Dynamic, Eigen::Dynamic>
    ArrayMatrixXcd;

typedef Eigen::Array<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic>
    ArrayVectorXd;
typedef Eigen::Array<Eigen::VectorXcd, Eigen::Dynamic, Eigen::Dynamic>
    ArrayVectorXcd;

typedef std::vector<VectorMatrixXcd> VectorVectorMatrixXcd;

typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagonalMatrixXd;

typedef std::vector<Eigen::VectorXcd> VectorVectorXcd;
typedef std::vector<Eigen::VectorXd> VectorVectorXd;

typedef std::vector<Eigen::ArrayXXcd> VectorArrayXXcd;

#endif
//
// Created by peri on 11/17/17.
//

#ifndef VANILLA_ACTIVATION_FUNCTIONS_HPP
#define VANILLA_ACTIVATION_FUNCTIONS_HPP

#include <cmath>
#include <iostream>
#include <boost/multi_array.hpp>
#include "boost/array.hpp"
#include <boost/next_prior.hpp>
#include "boost/cstdlib.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

class activation_functions{

public:
    template <typename Array>
    void ReLu(Array& A)
    {
        const auto n = A.data() + A.num_elements();
        for (auto i = A.data(); i != n; ++i){
            if(*i < 0){
                *i = 0.0;
            }
        }
    }

public:
    template <typename Array>
    void sigmoid(Array& A)
    {
        const auto n = A.data() + A.num_elements();
        for (auto i = A.data(); i != n; ++i){
            *i = 1 / (1 + expl(-1 *i));
        }
    }

public:
    template <typename Array>
    void tanh(Array& A)
    {
        const auto n = A.data() + A.num_elements();
        for (auto i = A.data(); i != n; ++i){
            *i = (2*expl(-1 *i) - 1)/(2*expl(-1 *i) + 1);
        }
    }


public:
    template <typename T>
    void ReLu(boost::numeric::ublas::matrix<T>& M)
    {
        for(unsigned i = 0; i < M.size1(); ++ i){
            for(unsigned j = 0; j < M.size2(); ++ j){
                if(M(i, j) < 0){
                    M(i, j) = 0;
                }
            }
        }
    }

public:
    template <typename T>
    void sigmoid(boost::numeric::ublas::matrix<T>& M)
    {
        for(unsigned i = 0; i < M.size1(); ++ i){
            for (unsigned j = 0; j < M.size2(); ++j) {
                    M(i, j) = 1 / (1 + expl(-1 * M(i, j)));
            }
        }

    }

public:
    template <typename T>
    void tanh(boost::numeric::ublas::matrix<T>& M)
    {
        for(unsigned i = 0; i < M.size1(); ++ i){
            for(unsigned j = 0; j < M.size2(); ++ j) {
                M(i, j) = (2 * expl(-1 * M(i, j)) - 1) / (2 * expl(-1 * M(i, j)) + 1);
            }
        }
    }


};
#endif //ACTIVATION_FUNCTIONS_HPP

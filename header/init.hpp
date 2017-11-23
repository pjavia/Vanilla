#ifndef VANILLA_INIT_H
#define VANILLA_INIT_H

#include <iostream>
#include <boost/multi_array.hpp>
#include "boost/array.hpp"
#include <boost/next_prior.hpp>
#include "boost/cstdlib.hpp"
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <ctime>

class init
{
public:
    template <typename Array>
    void ones(Array& A)
    {
        const auto n = A.data() + A.num_elements();
        for (auto i = A.data(); i != n; ++i){
            *i = 1.0;
        }
    }

public:
    template <typename Array>
    void normal(Array& A)
    {
        boost::mt19937 gene;
        boost::variate_generator<boost::mt19937, boost::normal_distribution<long double> > gen(gene, boost::normal_distribution<long double>(0.01, 0.02));
        const auto n = A.data() + A.num_elements();
        for (auto i = A.data(); i != n; ++i){
            *i = gen();
        }
    }

public:
    template <typename T>
    void normal(boost::numeric::ublas::matrix<T>& M)
    {
        boost::mt19937 gene;
        boost::variate_generator<boost::mt19937, boost::normal_distribution<long double > > gen(gene, boost::normal_distribution<long double>(0.01, 0.02));
        for (unsigned i = 0; i < M.size1(); ++ i){
            for (unsigned j = 0; j < M.size2(); ++ j){
                M(i, j) = gen();
            }
        }

    }
};
#endif

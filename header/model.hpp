//
// Created by peri on 11/19/17.
//

#ifndef VANILLA_MODEL_HPP
#define VANILLA_MODEL_HPP

#include "../header/activation_functions.hpp"
#include "../header/init.hpp"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>


class model {

    boost::numeric::ublas::matrix<long double > W_1;
    boost::numeric::ublas::matrix<long double > W_2;
    boost::numeric::ublas::matrix<long double > W_b;
    boost::numeric::ublas::matrix<long double > W_3;
    boost::numeric::ublas::matrix<long double > W_4;
    boost::numeric::ublas::matrix<long double > W_5;
    boost::numeric::ublas::matrix<long double > W_o;

    boost::numeric::ublas::matrix<long double > b_1;
    boost::numeric::ublas::matrix<long double > b_2;
    boost::numeric::ublas::matrix<long double > b_3;
    boost::numeric::ublas::matrix<long double > b_4;
    boost::numeric::ublas::matrix<long double > b_5;

    boost::numeric::ublas::matrix<long double> h1;
    boost::numeric::ublas::matrix<long double> h2;
    boost::numeric::ublas::matrix<long double> hb;
    boost::numeric::ublas::matrix<long double> h3;
    boost::numeric::ublas::matrix<long double> h4;
    boost::numeric::ublas::matrix<long double> h5;
    boost::numeric::ublas::matrix<long double> p;

    boost::numeric::ublas::matrix<long double> x;



public:
    model();

public:
    boost::numeric::ublas::matrix<long double > forward(boost::numeric::ublas::matrix<long double >& input);

public:
    void backward(boost::numeric::ublas::matrix<long double >& truth, boost::numeric::ublas::matrix<long double >& prediction);
};


#endif //VANILLA_MODEL_HPP

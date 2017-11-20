//
// Created by peri on 11/19/17.
//

#ifndef VANILLA_MODEL_HPP
#define VANILLA_MODEL_HPP

#include "../header/activation_functions.hpp"
#include "../header/init.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_map>

class model {

    boost::numeric::ublas::matrix<long double > hidden_layer_1;
    boost::numeric::ublas::matrix<long double > hidden_layer_2;
    boost::numeric::ublas::matrix<long double > bottle_neck_layer;
    boost::numeric::ublas::matrix<long double > hidden_layer_3;
    boost::numeric::ublas::matrix<long double > hidden_layer_4;
    boost::numeric::ublas::matrix<long double > hidden_layer_5;
    boost::numeric::ublas::matrix<long double > output_layer;

    //boost::numeric::ublas::matrix<long double > bias_layer_1;
    //boost::numeric::ublas::matrix<long double > bias_layer_2;
    //boost::numeric::ublas::matrix<long double > bias_bottle_neck_layer;
    //boost::numeric::ublas::matrix<long double > bias_layer_3;
    //boost::numeric::ublas::matrix<long double > bias_layer_4;
    //boost::numeric::ublas::matrix<long double > bias_layer_5;


public:
    model();

public:
    boost::numeric::ublas::matrix<long double > forward(boost::numeric::ublas::matrix<long double >& input);

public:
    void backward();
};


#endif //VANILLA_MODEL_HPP

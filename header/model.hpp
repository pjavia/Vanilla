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

    boost::numeric::ublas::matrix<double > hidden_layer_1;
    boost::numeric::ublas::matrix<double > hidden_layer_2;
    boost::numeric::ublas::matrix<double > bottle_neck_layer;
    boost::numeric::ublas::matrix<double > hidden_layer_3;
    boost::numeric::ublas::matrix<double > hidden_layer_4;
    boost::numeric::ublas::matrix<double > hidden_layer_5;
    boost::numeric::ublas::matrix<double > output_layer;

public:
    model();

public:
    boost::numeric::ublas::matrix<double > forward(boost::numeric::ublas::matrix<double >& input);

};


#endif //VANILLA_MODEL_HPP

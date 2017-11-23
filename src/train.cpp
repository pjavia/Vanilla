//
// Created by peri on 11/22/17.
//

#include "../header/pretty_print.hpp"
#include "../header/init.hpp"
#include "../header/model.hpp"


void train(){

    model Net;
    boost::numeric::ublas::matrix<long double > input(32, 784);
    init initializer;
    initializer.normal(input);
    boost::numeric::ublas::matrix<long double > output = Net.forward(input);
    Net.backward(input, output);

}
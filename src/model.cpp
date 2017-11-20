//
// Created by peri on 11/19/17.
//

#include "../header/model.hpp"


model::model() {

    init initialize;

    hidden_layer_1.resize(784, 256);
    initialize.normal(hidden_layer_1);
    hidden_layer_2.resize(256, 128);
    initialize.normal(hidden_layer_2);
    bottle_neck_layer.resize(128, 64);
    initialize.normal(bottle_neck_layer);
    hidden_layer_3.resize(64, 128);
    initialize.normal(hidden_layer_3);
    hidden_layer_4.resize(128, 256);
    initialize.normal(hidden_layer_4);
    hidden_layer_5.resize(256, 512);
    initialize.normal(hidden_layer_5);
    output_layer.resize(512, 784);
    initialize.normal(output_layer);

}

boost::numeric::ublas::matrix<long double > model::forward(boost::numeric::ublas::matrix<long double >& input) {

    activation_functions non_linearity;

    boost::numeric::ublas::matrix<long double> layer1 = boost::numeric::ublas::prec_prod(input, hidden_layer_1);
    non_linearity.ReLu(layer1);
    boost::numeric::ublas::matrix<long double> layer2 = boost::numeric::ublas::prec_prod(layer1, hidden_layer_2);
    non_linearity.ReLu(layer2);
    boost::numeric::ublas::matrix<long double> layer3 = boost::numeric::ublas::prec_prod(layer2, bottle_neck_layer);
    non_linearity.ReLu(layer3);
    boost::numeric::ublas::matrix<long double> layer4 = boost::numeric::ublas::prec_prod(layer3, hidden_layer_3);
    non_linearity.ReLu(layer4);
    boost::numeric::ublas::matrix<long double> layer5 = boost::numeric::ublas::prec_prod(layer4, hidden_layer_4);
    non_linearity.ReLu(layer5);
    boost::numeric::ublas::matrix<long double> layer6 = boost::numeric::ublas::prec_prod(layer5, hidden_layer_5);
    non_linearity.ReLu(layer6);
    boost::numeric::ublas::matrix<long double> layer7 = boost::numeric::ublas::prec_prod(layer6, output_layer);
    non_linearity.sigmoid(layer7);

    return layer7;

}

void model::backward(boost::numeric::ublas::matrix<long double >&) {

}
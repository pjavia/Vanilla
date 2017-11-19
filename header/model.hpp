//
// Created by peri on 11/19/17.
//

#ifndef VANILLA_MODEL_HPP
#define VANILLA_MODEL_HPP

#include "../header/activation_functions.hpp"
#include "../header/init.hpp"
#include <vector>
#include <string>
#include <map>

class model {

private:
    std::vector<unsigned int> layers;
    std::vector<std::string> non_linearity;
    std::vector<boost::numeric::ublas::matrix*> network;
    std::map<unsigned int, boost::numeric::ublas::matrix*> weights;

public:
    void model(int input_layer_neurons)
    {
        layers.push_back(input_layer_neurons);
        non_linearity.push_back("None");
    }

public:
    void append_linear_layer(const unsigned int hidden_neurons, const std::string& act_func)
    {

        layers.push_back(hidden_neurons);
        non_linearity.push_back(act_func);
    }

public:
    template <typename T>
    void forward_propagation(){

        for(int i = 0; i < layers.size()-1; i++){

            weights[i] = new boost::numeric::ublas::matrix<T>(layers[i], layers[i+1]);

        }
    }
};


#endif //VANILLA_MODEL_HPP

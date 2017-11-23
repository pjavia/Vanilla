//
// Created by peri on 11/19/17.
//

#include "../header/model.hpp"



model::model() {

    init initialize;

    W_1.resize(784, 256);
    initialize.normal(W_1);
    W_2.resize(256, 128);
    initialize.normal(W_2);
    W_b.resize(128, 64);
    initialize.normal(W_b);
    W_3.resize(64, 128);
    initialize.normal(W_3);
    W_4.resize(128, 256);
    initialize.normal(W_4);
    W_5.resize(256, 512);
    initialize.normal(W_5);
    W_o.resize(512, 784);
    initialize.normal(W_o);

    b_1.resize(1, 256);
    b_2.resize(1, 128);
    b_3.resize(1, 128);
    b_4.resize(1, 256);
    b_5.resize(1, 512);

}

boost::numeric::ublas::matrix<long double > model::forward(boost::numeric::ublas::matrix<long double >& input) {

    activation_functions non_linearity;

    x = input;
    size_t batch = x.size1();
    boost::numeric::ublas::matrix<int > transform(batch, 1, 1);

    h1 = boost::numeric::ublas::prec_prod(input, W_1) + boost::numeric::ublas::prec_prod(transform, b_1);
    non_linearity.ReLu(h1);
    h2 = boost::numeric::ublas::prec_prod(h1, W_2) + boost::numeric::ublas::prec_prod(transform, b_2);
    non_linearity.ReLu(h2);
    hb = boost::numeric::ublas::prec_prod(h2, W_b);
    non_linearity.ReLu(h3);
    h3 = boost::numeric::ublas::prec_prod(hb, W_3) + boost::numeric::ublas::prec_prod(transform, b_3);
    non_linearity.ReLu(h3);
    h4 = boost::numeric::ublas::prec_prod(h3, W_4) + boost::numeric::ublas::prec_prod(transform, b_4);
    non_linearity.ReLu(h4);
    h5 = boost::numeric::ublas::prec_prod(h4, W_5) + boost::numeric::ublas::prec_prod(transform, b_5);
    non_linearity.ReLu(h5);
    p = boost::numeric::ublas::prec_prod(h5, W_o);
    non_linearity.sigmoid(p);

    return p;

}

void model::backward(boost::numeric::ublas::matrix<long double >& truth, boost::numeric::ublas::matrix<long double >& prediction) {

    long double learning_rate = 0.001;
    size_t batch = prediction.size1();
    long double loss = 0.0;
    boost::numeric::ublas::matrix<long double > d_loss_p(prediction.size1(), prediction.size2());
    d_loss_p = prediction - truth;
    for(size_t i = 0; i < truth.size1(); ++i) {
        for (size_t j = 0; j < truth.size2(); ++j) {
            loss += std::abs(prediction(i, j) - truth(i, j));
            d_loss_p(i, j) = std::abs(d_loss_p(i, j))/batch;
        }
    }
    loss = loss/batch;
    std::cout << loss << std::endl;

    // First Stage
    //Loss
    // Second Stage
    boost::numeric::ublas::matrix<long double > d_p_W_o = h5;
    boost::numeric::ublas::matrix<long double > d_p_h5 = W_o;
    // Third Stage
    boost::numeric::ublas::matrix<long double > d_h5_W_5 = h4;
    boost::numeric::ublas::matrix<long double > d_h5_h4 = W_5;
    int d_h5_b5 = 1;
    // Fourth Stage
    boost::numeric::ublas::matrix<long double > d_h4_W_4 = h3;
    boost::numeric::ublas::matrix<long double > d_h4_h3 = W_4;
    int d_h4_b4 = 1;
    // Fifth Stage
    boost::numeric::ublas::matrix<long double > d_h3_W_3 = hb;
    boost::numeric::ublas::matrix<long double > d_h3_hb = W_3;
    int d_h3_b3 = 1;
    // Sixth Stage
    boost::numeric::ublas::matrix<long double > d_hb_W_b = h2;
    boost::numeric::ublas::matrix<long double > d_hb_h2 = W_b;
    // Seventh Stage
    boost::numeric::ublas::matrix<long double > d_h2_W_2 = h1;
    boost::numeric::ublas::matrix<long double > d_h2_h1 = W_2;
    int d_h2_b2 = 1;
    // Eighth Stage
    boost::numeric::ublas::matrix<long double > d_h1_W_1 = x;
    boost::numeric::ublas::matrix<long double > d_h1_x = W_1;
    int d_h1_b_1 = 1;

    // back propagation

    boost::numeric::ublas::matrix<long double > dL_dWo = boost::numeric::ublas::prec_prod(boost::numeric::ublas::trans(d_p_W_o), d_loss_p);
    boost::numeric::ublas::matrix<long double > delta1 = boost::numeric::ublas::prec_prod(d_loss_p, boost::numeric::ublas::trans(d_p_h5));
    // Wo
    W_o -= learning_rate*dL_dWo;

    boost::numeric::ublas::matrix<long double > dL_dW5 = boost::numeric::ublas::prec_prod(boost::numeric::ublas::trans(d_h5_W_5), delta1);
    boost::numeric::ublas::matrix<long double > delta2 = boost::numeric::ublas::prec_prod(delta1, boost::numeric::ublas::trans(d_h5_h4));
    W_5 -= learning_rate*dL_dW5;

    boost::numeric::ublas::matrix<long double > dL_dW4 = boost::numeric::ublas::prec_prod(boost::numeric::ublas::trans(d_h4_W_4), delta2);
    boost::numeric::ublas::matrix<long double > delta3 = boost::numeric::ublas::prec_prod(delta2, boost::numeric::ublas::trans(d_h4_h3));
    W_4 -= learning_rate*dL_dW4;

    boost::numeric::ublas::matrix<long double > dL_dW3 = boost::numeric::ublas::prec_prod(boost::numeric::ublas::trans(d_h3_W_3), delta3);
    boost::numeric::ublas::matrix<long double > delta_b = boost::numeric::ublas::prec_prod(delta3, boost::numeric::ublas::trans(d_h3_hb));

    W_3 -= learning_rate*dL_dW3;

    boost::numeric::ublas::matrix<long double > dL_dWb = boost::numeric::ublas::prec_prod(boost::numeric::ublas::trans(d_hb_W_b), delta_b);
    boost::numeric::ublas::matrix<long double > delta4 = boost::numeric::ublas::prec_prod(delta_b, boost::numeric::ublas::trans(d_hb_h2));

    W_b -= learning_rate*dL_dWb;

    boost::numeric::ublas::matrix<long double > dL_dW2 = boost::numeric::ublas::prec_prod(boost::numeric::ublas::trans(d_h2_W_2), delta4);
    boost::numeric::ublas::matrix<long double > delta5 = boost::numeric::ublas::prec_prod(delta4, boost::numeric::ublas::trans(d_h2_h1));

    W_2 -= learning_rate*dL_dW2;

    boost::numeric::ublas::matrix<long double > dL_dW1 = boost::numeric::ublas::prec_prod(boost::numeric::ublas::trans(d_h1_W_1), delta5);

    W_1 -= learning_rate*dL_dW1;
    
}
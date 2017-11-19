#include "../header/pretty_print.hpp"
#include "../header/init.hpp"
#include "../header/activation_functions.hpp"

int main(int argc, char **argv) {

    typedef boost::multi_array<double, 3> tensor;
    boost::array<tensor::index, 3> shape = {{5, 7, 2}};
    tensor  W(shape);
    typedef boost::numeric::ublas::matrix<double> Weight;
    init obj2;
    activation_functions non_linearity;
    Weight M(728, 512);
    obj2.normal(M);
    non_linearity.tanh(M);
    std::cout << M << std::endl;
    pretty_print obj;
    init obj1;
    obj1.normal<tensor>(W);
    obj.print(std::cout, W);
    return 0;

}

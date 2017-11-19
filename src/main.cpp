#include "../header/pretty_print.hpp"
#include "../header/init.hpp"
#include "../header/model.hpp"

int main(int argc, char **argv) {
    model Net;
    boost::numeric::ublas::matrix<double> input(32, 784);
    init initializer;
    initializer.normal(input);
    auto output = Net.forward(input);
    std::cout << output;
    return 0;

}

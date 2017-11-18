#ifndef PRETTY_PRINT_H
#define PRETTY_PRINT_H

#include <iostream>
#include <boost/multi_array.hpp>
#include "boost/array.hpp"
#include <boost/next_prior.hpp>
#include "boost/cstdlib.hpp"

class pretty_print
{
    
    public:
        template <typename Array>
        void print(std::ostream& os, const Array& A)
        {
            typename Array::const_iterator i;
            os << "[";
            for (i = A.begin(); i != A.end(); ++i) {
            print(os, *i);
            if (boost::next(i) != A.end())
                os << ',';
            }
            os << "]";
        }
        void print(std::ostream& os, const double& x)
        {
            os << x;
        }
};

#endif

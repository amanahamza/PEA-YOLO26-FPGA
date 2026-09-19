#include "mac_array.h"
#include <iostream>

int main() {
    data_t a[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    data_t b[N] = {8, 7, 6, 5, 4, 3, 2, 1};
    // Expected: (1*8)+(2*7)+(3*6)+(4*5)+(5*4)+(6*3)+(7*2)+(8*1) = 120

    acc_t result = mac_array(a, b);
    std::cout << "Result: " << result << " (expected 120)\n";

    if (result == 120) {
        std::cout << "TEST PASSED\n";
        return 0;
    }
    std::cout << "TEST FAILED\n";
    return 1;
}
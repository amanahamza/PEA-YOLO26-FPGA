#include "mac.h"
#include <iostream>

int main() {
    acc_t acc = 0;
    data_t a1 = 5, b1 = 3;
    acc = mac(a1, b1, acc);
    std::cout << "After 5*3: " << acc << " (expected 15)\n";

    data_t a2 = -4, b2 = 10;
    acc = mac(a2, b2, acc);
    std::cout << "After -4*10: " << acc << " (expected -25)\n";

    data_t a3 = 127, b3 = 127;
    acc = mac(a3, b3, acc);
    std::cout << "After 127*127: " << acc << " (expected " << (-25 + 127*127) << ")\n";

    if (acc == (-25 + 127*127)) {
        std::cout << "TEST PASSED\n";
        return 0;
    }
    std::cout << "TEST FAILED\n";
    return 1;
}
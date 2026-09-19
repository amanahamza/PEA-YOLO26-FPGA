#include "sliding_window_h.h"
#include <iostream>

int main() {
    data_t input[IMG_H][IMG_W];
    data_t weights[KERNEL_H] = {1, 2, 3};  // simple known weights
    acc_t output[IMG_H][IMG_W];

    // fill input with a simple known pattern: value = row number
    for (int r = 0; r < IMG_H; r++)
        for (int c = 0; c < IMG_W; c++)
            input[r][c] = r % 10;  // keep values small, avoid INT8 overflow

    conv1d_vertical(input, weights, output);

    // Manually check output at row=5, col=0:
    // window should be rows {3,4,5} = {3,4,5}
    // sum = 3*1 + 4*2 + 5*3 = 3+8+15 = 26
    std::cout << "output[5][0] = " << output[5][0] << " (expected 26)\n";

    if (output[5][0] == 26) {
        std::cout << "TEST PASSED\n";
        return 0;
    }
    std::cout << "TEST FAILED\n";
    return 1;
}
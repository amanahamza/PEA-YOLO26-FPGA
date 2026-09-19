#include "sliding_window_h.h"

void conv1d_vertical(
    data_t input[IMG_H][IMG_W],
    data_t weights[KERNEL_H],
    acc_t output[IMG_H][IMG_W]
) {
#pragma HLS ARRAY_PARTITION variable=weights complete

    data_t window[KERNEL_H];
#pragma HLS ARRAY_PARTITION variable=window complete

    for (int col = 0; col < IMG_W; col++) {
        // reset window at the start of each column
        window[0] = 0; window[1] = 0; window[2] = 0;

        for (int row = 0; row < IMG_H; row++) {
#pragma HLS PIPELINE II=1
            // shift the window down by one row
            window[0] = window[1];
            window[1] = window[2];
            window[2] = input[row][col];

            // compute the dot product for this window position
            acc_t sum = 0;
            for (int k = 0; k < KERNEL_H; k++) {
#pragma HLS UNROLL
                sum += (acc_t)(window[k] * weights[k]);
            }

            output[row][col] = sum;
        }
    }
}
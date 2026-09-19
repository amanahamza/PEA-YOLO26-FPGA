#include "mac_array.h"

acc_t mac_array(data_t a[N], data_t b[N]) {
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=a complete
#pragma HLS ARRAY_PARTITION variable=b complete

    acc_t sum = 0;
    for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
        sum += (acc_t)(a[i] * b[i]);
    }
    return sum;
}
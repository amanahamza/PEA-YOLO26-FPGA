#include "mac.h"
acc_t mac(data_t a, data_t b, acc_t acc_in) {
#pragma HLS PIPELINE II=1
    return acc_in + (acc_t)(a * b);
}
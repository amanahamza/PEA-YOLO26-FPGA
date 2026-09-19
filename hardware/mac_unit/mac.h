#ifndef MAC_H
#define MAC_H
#include "ap_int.h"
typedef ap_int<8>  data_t;
typedef ap_int<32> acc_t;
acc_t mac(data_t a, data_t b, acc_t acc_in);
#endif
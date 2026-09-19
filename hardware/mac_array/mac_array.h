#ifndef MAC_ARRAY_H
#define MAC_ARRAY_H

#include "ap_int.h"

#define N 8   // number of parallel MACs — dot product width

typedef ap_int<8>  data_t;
typedef ap_int<32> acc_t;

acc_t mac_array(data_t a[N], data_t b[N]);

#endif
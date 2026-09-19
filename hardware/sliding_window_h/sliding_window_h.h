#ifndef SLIDING_WINDOW_H_H
#define SLIDING_WINDOW_H_H

#include "ap_int.h"

#define IMG_H 80
#define IMG_W 80
#define KERNEL_H 3

typedef ap_int<8>  data_t;
typedef ap_int<32> acc_t;

void conv1d_vertical(
    data_t input[IMG_H][IMG_W],
    data_t weights[KERNEL_H],
    acc_t output[IMG_H][IMG_W]
);

#endif
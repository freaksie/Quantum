#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 10
#define N_INPUT_2_1 10
#define N_TIME_STEPS_2 10
#define N_OUT_2 64
#define N_OUT_3 64
#define N_LAYER_4 64
#define N_LAYER_4 64
#define N_LAYER_6 32
#define N_LAYER_6 32
#define N_LAYER_8 16
#define N_LAYER_8 16
#define N_LAYER_10 4
#define N_LAYER_10 4
#define N_LAYER_12 1
#define N_LAYER_12 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<8,2> input_t;
typedef ap_fixed<8,2> model_default_t;
typedef ap_fixed<8,2> layer2_t;
typedef ap_fixed<18,8> lstm_table_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<8,2> layer3_t;
typedef ap_fixed<18,8> lstm_1_table_t;
typedef ap_uint<1> layer3_index;
typedef ap_fixed<8,2> layer4_t;
typedef ap_uint<1> layer4_index;
typedef ap_fixed<8,2> layer5_t;
typedef ap_fixed<18,8> dense_5_selu_table_t;
typedef ap_fixed<8,2> layer6_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<8,2> layer7_t;
typedef ap_fixed<18,8> dense_6_selu_table_t;
typedef ap_fixed<8,2> layer8_t;
typedef ap_uint<1> layer8_index;
typedef ap_fixed<8,2> layer9_t;
typedef ap_fixed<18,8> dense_7_selu_table_t;
typedef ap_fixed<8,2> layer10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<8,2> layer11_t;
typedef ap_fixed<18,8> dense_8_selu_table_t;
typedef ap_fixed<8,2> layer12_t;
typedef ap_uint<1> layer12_index;
typedef ap_fixed<8,2> result_t;
typedef ap_fixed<18,8> dense_9_sigmoid_table_t;

#endif

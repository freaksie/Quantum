#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t input_2[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer13_out[N_LAYER_12]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_2,layer13_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 2560>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 16384>(wr2, "wr2.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(br2, "br2.txt");
        nnet::load_weights_from_txt<model_default_t, 16384>(w3, "w3.txt");
        nnet::load_weights_from_txt<model_default_t, 16384>(wr3, "wr3.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b3, "b3.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(br3, "br3.txt");
        nnet::load_weights_from_txt<model_default_t, 4096>(w4, "w4.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 2048>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b6, "b6.txt");
        nnet::load_weights_from_txt<model_default_t, 512>(w8, "w8.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b8, "b8.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(w10, "w10.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b10, "b10.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(w12, "w12.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b12, "b12.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_TIME_STEPS_2*N_OUT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::lstm_stack<input_t, layer2_t, config2>(input_2, layer2_out, w2, wr2, b2, br2); // lstm

    layer3_t layer3_out[N_OUT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::lstm_stack<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, wr3, b3, br3); // lstm_1

    layer4_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // dense_5

    layer5_t layer5_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::selu<layer4_t, layer5_t, selu_config5>(layer4_out, layer5_out); // dense_5_selu

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // dense_6

    layer7_t layer7_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::selu<layer6_t, layer7_t, selu_config7>(layer6_out, layer7_out); // dense_6_selu

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // dense_7

    layer9_t layer9_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::selu<layer8_t, layer9_t, selu_config9>(layer8_out, layer9_out); // dense_7_selu

    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // dense_8

    layer11_t layer11_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::selu<layer10_t, layer11_t, selu_config11>(layer10_out, layer11_out); // dense_8_selu

    layer12_t layer12_out[N_LAYER_12];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::dense<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // dense_9

    nnet::sigmoid<layer12_t, result_t, sigmoid_config13>(layer12_out, layer13_out); // dense_9_sigmoid

}

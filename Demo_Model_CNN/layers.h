#include <iostream>
#include <cmath>
#include "layers.cpp"

// Activation function
void relu(const double *input, double *output, int size);
void leaky_relu(const double *input, double *output, int size);
void sigmoid(const double *input, double *output, int size);
void tanh(const double *input, double *output, int size);
void softmax(const double *input, double *output, int size);
void hard_swish(const double *input, double *output, int size);

// Padding
void pad_image(const double *input, double *output, int imageWidth, int imageHeight, int imageDepth, int p_h, int p_w);
void add_padding_middle(const double *input, double *output, int imageWidth, int imageHeight, int imageDepth, int z_h, int z_w);

// Convolution
void convolution (int c_in, int c_out, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, const double flattenedImage[], const double kernels[], const double biases[], double output[]);
void transposed_convolution(int c_in, int c_out, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, const double flattenedImage[], const double kernels[], const double biases[], double output[]);

// Pooling
void max_pooling(int c, int h, int w, int k_h, int k_w, int s_h, int s_w, const double *input, double *output);
void avg_pooling(int c, int h, int w, int k_h, int k_w, int s_h, int s_w, const double *input, double *output);

// Fully connected
void linear(const double input[], double output[], const double weights[], const double bias[], int inputSize, int outputSize);

// Noise
void addNoise(double *input, double noise, int size);
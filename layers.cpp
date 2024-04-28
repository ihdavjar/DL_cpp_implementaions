
#include <iostream>
#include <cmath>


// ############################## Activation Functions ##############################

// ReLU Activation Function
void relu(double *input, double *output, int size) {
    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = (input[i] < 0) ? 0 : input[i];
    }   
}

// Leaky ReLU Activation Function
void leaky_relu(double *input, double *output, int size) {
    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = (input[i] < 0) ? 0.01 * input[i] : input[i];
    }
}

// Sigmoid Activation Function
void sigmoid(double *input, double *output, int size) {
    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = 1 / (1 + std::exp(-input[i]));
    }
}

// Tanh Activation Function
void tanh(double *input, double *output, int size) {
    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = (std::exp(input[i]) - std::exp(-input[i])) / (std::exp(input[i]) + std::exp(-input[i]));
    }
}

// Softmax Activation Function
void softmax(double *input, double *output, int size) {
    double sum = 0;
    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = std::exp(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] /= sum;
    }
}

// Hard Swish Activation Function
void hard_swish(double *input, double *output, int size) {
    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = (input[i] * std::min(std::max(input[i] + 3, 0.0), 6.0)) / 6.0;
    }
}

// ############################## Convolution Layers ##############################

// Convolution Layer
// Input: flattenedImage, kernels, biases, imageWidth, imageHeight, imageDepth, k_h, k_w, numKernels, stride, padding
// Output: output
// Description of arguments:
// flattenedImage: Flattened image of size imageWidth * imageHeight * imageDepth
// kernels: Weights of the kernels of size numKernels * k_h * k_w * imageDepth
// biases: Biases of the kernels of size numKernels
// imageWidth: Width of the image
// imageHeight: Height of the image
// imageDepth: Depth of the image
// k_h: Height of the kernel
// k_w: Width of the kernel
// numKernels: Number of kernels
// stride: Stride of the convolution
// padding: Padding of the convolution

//Function to do the padding
void pad_image(const double *input, double *output, int imageWidth, int imageHeight, int imageDepth, int p_h, int p_w) {
    int paddedWidth = imageWidth + 2 * p_h;
    int paddedHeight = imageHeight + 2 * p_w;

    for (int i = 0; i < paddedHeight; ++i) {
        for (int j = 0; j < paddedWidth; ++j) {
            for (int k = 0; k < imageDepth; ++k) {
                if (i < p_h || j < p_w || i >= paddedHeight - p_h || j >= paddedWidth - p_w) {
                    output[k * paddedHeight * paddedWidth + i * paddedWidth + j] = 0;
                } else {
                    output[k * paddedHeight * paddedWidth + i * paddedWidth + j] = input[k * imageHeight * imageWidth + (i - p_h) * imageWidth + (j - p_w)];
                }
            }
        }
    }
}


void convolution (int c_in, int c_out, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, const double flattenedImage[], const double kernels[], const double biases[], double output[]) {
    
    int output_h = (image_h + 2 * p_h - k_h) / s_h + 1;
    int output_w = (image_w + 2 * p_w - k_w) / s_w + 1;
    int output_c = c_out;

    double paddedImage[(c_in * (image_h + 2 * p_h) * (image_w + 2 * p_w))];

    // New image size after padding
    int paddedHeight = image_h + 2 * p_h; 
    int paddedWidth = image_w + 2 * p_w;
    
    pad_image(flattenedImage, paddedImage, image_w, image_h, c_in, p_h, p_w);

    for (int k = 0; k < c_out; ++k) {

        for (int i = 0; i < output_h; ++i) {

            for (int j = 0; j < output_w; ++j) {

                // Initialize output with bias for the current kernel
                output[(k * output_h * output_w) + (i * output_w) + j] = biases[k];

                
                for (int kc = 0; kc < c_in; ++kc) {

                    for (int ki = 0; ki < k_h; ++ki) {

                        for (int kj = 0; kj < k_w; ++kj) {

                            output[(k * output_h * output_w) + (i * output_w) + j] += paddedImage[(kc * paddedHeight * paddedWidth) + ((ki + (i * s_h)) * paddedWidth) + (kj + (j * s_w))] * kernels[(k * (k_h * k_w * c_in)) + (kc * (k_h * k_w)) + (ki * k_w) + kj];
                        }
                    }
                }

            }
        }
    }
}

/// Transpose Convolution Layer
// Input: flattenedImage, kernels, biases, imageWidth, imageHeight, imageDepth, k_h, k_w, numKernels, stride, padding, group
// Output: output
// Description of arguments:
// flattenedImage: Flattened image of size imageWidth * imageHeight * imageDepth
// kernels: Weights of the kernels of size numKernels * k_h * k_w * imageDepth
// biases: Biases of the kernels of size numKernels
// imageWidth: Width of the image
// imageHeight: Height of the image
// imageDepth: Depth of the image
// k_h: Height of the kernel
// k_w: Width of the kernel
// numKernels: Number of kernels
// stride: Stride of the convolution
// padding: Padding of the convolution
// group: Number of group

// Function to add padding in middle of the image
// This function would take a image as input and add z number of zeros in the middle of each column and row
// The number of zeros to be added is given by the padding parameter z

void add_padding_middle(const double *input, double *output, int imageWidth, int imageHeight, int imageDepth, int z_h, int z_w) {

    int paddedWidth = imageWidth + (z_h * (imageWidth - 1));
    int paddedHeight = imageHeight + (z_w * (imageHeight - 1));

    for (int i = 0; i < paddedHeight; ++i) {
        for (int j = 0; j < paddedWidth; ++j) {
            for (int k = 0; k < imageDepth; ++k) {
                if (i % (z_h + 1) == 0 || j % (z_w + 1) == 0) {
                    output[(k * paddedHeight * paddedWidth) + (i * paddedWidth) + j] = input[(k * imageHeight * imageWidth) + (i / (z_h + 1)) * imageWidth + (j / (z_w + 1))];
                } 
                
                else {
                    output[k * paddedHeight * paddedWidth + i * paddedWidth + j] = 0;
                }
            }
        }
    }
}

void transposed_convolution(int c_in, int c_out, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int image_h, int image_w, const double flattenedImage[], const double kernels[], const double biases[], double output[]) {

    // Calculating the parameters
    // Number of zeros to be added in the middle of the image
    int z_h = s_h - 1;   
    int z_w = s_w - 1;

    // New padding
    int p_h_0 = k_h - p_h - 1;
    int p_w_0 = k_w - p_w - 1;

    // New stride
    int s_h_0 = 1;
    int s_w_0 = 1;


    // Adding zeros in the middle of the image
    // size of the new image
    int image_h_0 = image_h + (z_h * (image_h - 1));
    int image_w_0 = image_w + (z_w * (image_w - 1));
    int image_c_0 = c_in;

    double paddedImage[(c_in * (image_h_0) * (image_w_0))];
    add_padding_middle(flattenedImage, paddedImage, image_w, image_h, c_in, z_h, z_w);

    // Using the convolution function with new parameters'
    convolution(c_in, c_out, k_h, k_w, s_h_0, s_w_0, p_h_0, p_w_0, image_h_0, image_w_0, paddedImage, kernels, biases, output);
}

// ############################## Pooling Layers ##############################

// Max Pooling Layer
void maxPooling(const double flattenedImage[], double output[], int image_h, int image_w, int numChannels, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w) {
    
    int output_h = (image_h + 2 * p_h - k_h) / s_h + 1;
    int output_w = (image_w + 2 * p_w - k_w) / s_w + 1;
    int output_c = numChannels;

    double paddedImage[(numChannels * (image_h + 2 * p_h) * (image_w + 2 * p_w))];

    // New image size after padding
    int paddedHeight = image_h + 2 * p_h; 
    int paddedWidth = image_w + 2 * p_w;
    
    pad_image(flattenedImage, paddedImage, image_w, image_h, numChannels, p_h, p_w);


    for (int c = 0; c < numChannels; ++c) {
        for (int i = 0; i < output_h; ++i) {
            for (int j = 0; j < output_w; ++j) {

                // Initiallising the maximum value
                double max_val = 0.0;
                // Finding the maximum value in the kernel
                for (int ki = 0; ki < k_h; ++ki) {

                    for (int kj = 0; kj < k_w; ++kj) {

                        max_val = fmax(max_val, paddedImage[(c * paddedHeight * paddedWidth) + ((ki + (i * s_h)) * paddedWidth) + (kj + (j * s_w))]);
                    }
                }

                // Assigning the maximum value to the output
                output[c * (output_h * output_w) + i * output_w + j] = max_val;
            }
        }
    }
}

// Average Pooling Layer
void avgPooling(const double flattenedImage[], double output[], int image_h, int image_w, int numChannels, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w) {
    
    int output_h = (image_h + 2 * p_h - k_h) / s_h + 1;
    int output_w = (image_w + 2 * p_w - k_w) / s_w + 1;
    int output_c = numChannels;

    double paddedImage[(numChannels * (image_h + 2 * p_h) * (image_w + 2 * p_w))];

    // New image size after padding
    int paddedHeight = image_h + 2 * p_h; 
    int paddedWidth = image_w + 2 * p_w;
    
    pad_image(flattenedImage, paddedImage, image_w, image_h, numChannels, p_h, p_w);


    for (int c = 0; c < numChannels; ++c) {
        for (int i = 0; i < output_h; ++i) {
            for (int j = 0; j < output_w; ++j) {

                // Initiallising the average value
                double avg_val = 0.0;

                // Finding the average value in the kernel
                for (int ki = 0; ki < k_h; ++ki) {

                    for (int kj = 0; kj < k_w; ++kj) {

                        avg_val += paddedImage[(c * paddedHeight * paddedWidth) + ((ki + (i * s_h)) * paddedWidth) + (kj + (j * s_w))];
                    }
                }

                // Assigning the maximum value to the output
                output[c * (output_h * output_w) + i * output_w + j] = avg_val/(k_h*k_w);
            }
        }
    }
}

// ############################### Normalisation Layers ###############################

// ############################## Fully Connected Layers ##############################
void linear(const double input[], double output[], const double weights[], const double bias[], int inputSize, int outputSize) {

	for (int i = 0; i < outputSize; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = bias[i];
        for (int j = 0; j < inputSize; ++j) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
            output[i] += input[j] * weights[i * inputSize + j];
        }
    }
}

// ############################## Dropout Layer ##############################



// For Testing:-
// int main() {

//         static const double test_image[28*28*1]= {  
// 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.7254902, 0.62352943, 0.5921569, 0.23529412, 0.14117648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87058824, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.94509804, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.6666667, 0.20392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2627451, 0.44705883, 0.28235295, 0.44705883, 0.6392157, 0.8901961, 0.99607843, 0.88235295, 0.99607843, 0.99607843, 0.99607843, 0.98039216, 0.8980392, 0.99607843, 0.99607843, 0.54901963, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666667, 0.25882354, 0.05490196, 0.2627451, 0.2627451, 0.2627451, 0.23137255, 0.08235294, 0.9254902, 0.99607843, 0.41568628, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3254902, 0.99215686, 0.81960785, 0.07058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08627451, 0.9137255, 1.0, 0.3254902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5058824, 0.99607843, 0.93333334, 0.17254902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23137255, 0.9764706, 0.99607843, 0.24313726, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52156866, 0.99607843, 0.73333335, 0.019607844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03529412, 0.8039216, 0.972549, 0.22745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49411765, 0.99607843, 0.7137255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29411766, 0.9843137, 0.9411765, 0.22352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07450981, 0.8666667, 0.99607843, 0.6509804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764706, 0.79607844, 0.99607843, 0.85882354, 0.13725491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14901961, 0.99607843, 0.99607843, 0.3019608, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12156863, 0.8784314, 0.99607843, 0.4509804, 0.003921569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52156866, 0.99607843, 0.99607843, 0.20392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23921569, 0.9490196, 0.99607843, 0.99607843, 0.20392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098, 0.99607843, 0.99607843, 0.85882354, 0.15686275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098, 0.99607843, 0.8117647, 0.07058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


// };
//     // Testing the padding layer
//     // double paddedImage[55*55*1];
//     // add_padding_middle(test_image, paddedImage, 28, 28, 1, 1, 1);
//     // for (int i = 0; i < 55; ++i) {
//     //     for (int j = 0; j < 55; ++j) {
//     //         for (int k = 0; k < 1; ++k) {
//     //             cout << paddedImage[k * 55 * 55 + i * 55 + j] << ",";
//     //         }
//     //     }
//     // }
//     // Testing the avgpooling
//     double output[14*14*1];
//     maxPooling(test_image, output, 28, 28, 1, 2, 2, 2, 2, 0, 0);
//     for (int i = 0; i < 14; ++i) {
//         for (int j = 0; j < 14; ++j) {
//             for (int k = 0; k < 1; ++k) {
//                 cout << output[k * 14 * 14 + i * 14 + j] << ",";
//             }
//         }
//     }

// }
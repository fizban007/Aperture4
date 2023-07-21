#include <memory>
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <assert.h>
#include <random>

#include "core/multi_array.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"

namespace Aperture {

// using Matrix = buffer<float>;
using Matrix = multi_array<float, 2>;

 // void randomize(multi_array<)
class NNLayer {
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;
	virtual Matrix& forward(Matrix& A) = 0;
	std::string getName() { return this->name; };

};

inline NNLayer::~NNLayer() {}

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	Matrix Y;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();
	Matrix& forward(Matrix& X);
	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};

NeuralNetwork::NeuralNetwork(float learning_rate) :
	learning_rate(learning_rate)
{ }

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix& NeuralNetwork::forward(Matrix& X) {
    // if (Y.extent() != X.extent()) {
    //     Y.resize(X.extent());
    // }
    Matrix* ptr = &X;
	for (auto layer : layers) {
		auto& Y = layer->forward(*ptr);
        ptr = &Y;
	}
	return *ptr;
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}

using Shape = index_t<2>;

class LinearLayer : public NNLayer {
// private:
public:
	const float weights_init_threshold = 0.01;
	Matrix W;
	Matrix b;
	Matrix Z;
	Matrix A;
	void initializeBiasWithZeros();
	void initializeWeightsRandomly();
	void computeAndStoreLayerOutput(Matrix& A);
	void loadWeightsAndBiasFromFile(std::string filename);
    Shape layer_shape; // Specifically the shape of W

// public:
	LinearLayer(std::string name, Shape W_shape);
	LinearLayer(std::string name, Shape W_shape, std::string filename);
	~LinearLayer();
	Matrix& forward(Matrix& A);
	int getXDim() const;
	int getYDim() const;
	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;
};

LinearLayer::LinearLayer(std::string name, Shape W_shape) : W(W_shape[0], W_shape[1]), b(W_shape[1], 1)
{
    layer_shape = W_shape;
	this->name = name;
	initializeBiasWithZeros();
	initializeWeightsRandomly();
}

LinearLayer::LinearLayer(std::string name, Shape W_shape, std::string filename) : W(W_shape[0], W_shape[1]), b(W_shape[1], 1)
{
    layer_shape = W_shape;
	this->name = name;
	loadWeightsAndBiasFromFile(filename);
}

LinearLayer::~LinearLayer()
{ }

void LinearLayer::initializeWeightsRandomly() {
	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);
	int count = 0;
	for (int y = 0; y < W.extent()[1]; y++) {
	    for (int x = 0; x < W.extent()[0]; x++) {
			W[y * layer_shape[0] + x] = normal_distribution(generator) * weights_init_threshold;
			// std::cout << "Element set: " << y * W.shape.x + x << std::endl;
			count++;
		}
	}
	std::cout << "Weights set: " << count << std::endl;
	W.copy_to_device();
}

void LinearLayer::initializeBiasWithZeros() {
	int count = 0;
	for (int x = 0; x < layer_shape[1]; x++) {
		b[x] = 0;
		count++;
	}
	std::cout << "Bias set: " << count << std::endl;
    b.copy_to_device();
}

void LinearLayer::loadWeightsAndBiasFromFile(std::string filename) {
	std::ifstream file(filename);
	std::cout << "W shape x: " << W.extent()[0] << std::endl;
	std::cout << "W shape y: " << W.extent()[1] << std::endl;
	std::cout << "b shape x: " << W.extent()[1] << std::endl;
	std::cout << "b shape y: " << 1 << std::endl;
	int count = 0;
	if (file.is_open()) {
		for (int y = 0; y < layer_shape[1]; y++) {
			std::string line;
			std::getline(file, line);
			std::vector<std::string> tokens;
			std::string token;
			std::stringstream token_stream(line);
            // std::cout << line << std::endl;
			while (std::getline(token_stream, token, ' ')) {
				tokens.push_back(token);
			}
            std::cout << tokens[0] << std::endl;
			// Set values in W matrix
			for (int x = 0; x < layer_shape[0]; x++) {
				// W[y * layer_shape[0] + x] = std::stof(tokens[x]);
                W(x, y) = std::stof(tokens[x]);
				// W[y * W.shape.x + x] = 0;
				// std::cout << W[y * W.shape.x + x] << std::endl;
				// std::cout << "Element set: " << y * W.shape.x + x << std::endl;
				count++;
			}
		}
		std::cout << "Weights set: " << count << std::endl;

		count = 0;

		for (int x = 0; x < layer_shape[1]; x++){
			std::string line;
			std::getline(file, line);
			b[x] = std::stof(line);
			// b[x] = 0;
			// std::cout << b[x] << std::endl;
			// std::cout << "Element set: " << x << std::endl;
			count++;
		}
		std::cout << "Bias set: " << count << std::endl;
		file.close();
	}
	else {
		throw std::runtime_error("Cannot open file with weights and bias.");
	}
	W.copy_to_device();
	b.copy_to_device();
}

Matrix& LinearLayer::forward(Matrix& A) {
	assert(W.extent()[0] == A.extent()[1]);
    std::cout << A.extent()[0] << ", " << A.extent()[1] << std::endl;
	// this->A.copy_from(A);
	// Shape Z_shape(A.shape.x, W.shape.y);
    extent_t<2> Z_ext(A.extent()[0], W.extent()[1]);
	// Z.allocateMemoryIfNotAllocated(Z_shape);
    if (Z.extent() != Z_ext) {
        Z.resize(Z_ext);
    }
    std::cout << Z.extent()[0] << ", " << Z.extent()[1] << std::endl;
	computeAndStoreLayerOutput(A);
    // Throw exception if layer output computation failed
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Cannot compute layer output.");
    }
	return Z;
}

__device__ 
float sigmoid(float x) {
	return 1.0f / (1 + exp(-x));
}

// __global__ 
// void linearLayerForward( float* W, float* A, float* Z, float* b, int W_x_dim, int W_y_dim, int A_x_dim, int A_y_dim) {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	int Z_x_dim = A_x_dim;
// 	int Z_y_dim = W_y_dim;
// 	float Z_value = 0;
// 	if (row < Z_y_dim && col < Z_x_dim) {
// 		for (int i = 0; i < W_x_dim; i++) {
// 			Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
// 		}
// 		Z[row * Z_x_dim + col] = sigmoid(Z_value + b[row]);
// 	}
// }

void LinearLayer::computeAndStoreLayerOutput(Matrix& A) {
    kernel_launch([] LAMBDA (auto Wptr, auto Aptr, auto Zptr, auto bptr,
    auto Wx, auto Wy, auto Ax, auto Ay) {
        int Z_x_dim = Ax; //1
        int Z_y_dim = Wy; //64
        for (auto id : grid_stride_range(0, Z_x_dim * Z_y_dim)) {
            float Z_value = 0;
            int col = id % Z_x_dim;
            int row = id / Z_x_dim;
            if (row < Z_y_dim && col < Z_x_dim) {
                for (int i = 0; i < Wx; i++) {
                    Z_value += Wptr[row * Wx + i] * Aptr[i * Ax + col];
                }
                Zptr[id] = sigmoid(Z_value + bptr[row]);
            }
            printf("Z[%d] = %f\n", id, Zptr[id]);
        }
    }, 
    W.dev_ptr(), A.dev_ptr(), Z.dev_ptr(), b.dev_ptr(),
    W.extent()[0], W.extent()[1], A.extent()[0], A.extent()[1]);

	// dim3 block_size(8, 8);
	// dim3 num_of_blocks(	(Z.shape.x + block_size.x - 1) / block_size.x, (Z.shape.y + block_size.y - 1) / block_size.y);
	// linearLayerForward<<<num_of_blocks, block_size>>>( W.data_device.get(),
	// 												   A.data_device.get(),
	// 												   Z.data_device.get(),
	// 												   b.data_device.get(),
	// 												   W.shape.x, W.shape.y, A.shape.x, A.shape.y);
}

int LinearLayer::getXDim() const {
	return W.extent()[0];
}

int LinearLayer::getYDim() const {
	return W.extent()[1];
}

// const Matrix& LinearLayer::getWeightsMatrix() const {
// 	return W;
// }

// Matrix LinearLayer::getBiasVector() const {
// 	return b;
// }

// class SigmoidActivation : public NNLayer {
// private:
// 	Matrix A;
// 	Matrix Z;

// public:
// 	SigmoidActivation(std::string name);
// 	~SigmoidActivation();
// 	Matrix& forward(Matrix& Z);
// };

// __global__
// void sigmoidActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < Z_x_dim * Z_y_dim) {
//         A[index] = sigmoid(Z[index]);
//     }
// }

// SigmoidActivation::SigmoidActivation(std::string name) {
//     this->name = name;
// }

// SigmoidActivation::~SigmoidActivation() { }

// Matrix& SigmoidActivation::forward(Matrix& Z) {
//     this->Z = Z;
//     A.allocateMemoryIfNotAllocated(Z.shape);
//     dim3 block_size(256);
//     dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
// 	sigmoidActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);
//     // Throw exception if any errors occurred during kernel execution
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
//         throw std::runtime_error("Cannot compute layer output.");
//     }
// 	return A;
// }


}

using namespace Aperture;

int main() {
    NeuralNetwork net;
    net.addLayer(new LinearLayer("linear_1", Shape(1, 64), "./net1_layer0.txt"));
    // net.addLayer(new LinearLayer("linear_1", Shape(1, 64)));
    // net.addLayer(new LinearLayer("linear_2", Shape(64, 64), "./net1_layer2.txt"));
    // net.addLayer(new LinearLayer("linear_3", Shape(64, 1), "./net1_layer4.txt"));

    // LinearLayer linear1("linear_1", Shape(1, 64), "./net1_layer0.txt");
    // Matrix w = linear1.getWeightsMatrix();
    // for (int i = 0; i < 64; i++)
    // {
    //     std::cout << w[i] << " " << std::endl;
    // }
    // Matrix b = linear1.getBiasVector();
    // for (int i = 0; i < 64; i++)
    // {
    //     std::cout << b[i] << " " << std::endl;
    // }

    Matrix test_input(1,1);
    // Set test input
    // test_input.allocateMemory();
    test_input[0] = 0.5;
    test_input.copy_to_device();

    Matrix& test_out = net.forward(test_input);
    test_out.copy_to_host();
    std::cout << "NN input: " << test_input[0] << std::endl; // Should be "0.5"
    std::cout << "NN output: " << test_out[0] << std::endl;

    return 0;
}

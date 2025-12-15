#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <cstdint>

class NeuralNetwork {
private:
    // Tensor arena 16-byte aligned
    alignas(16) uint8_t tensor_arena[35000];
    
public:
    NeuralNetwork();
    ~NeuralNetwork();
    
    void predict();
    float* getInputBuffer();
    float* getOutputBuffer();
    int getInputSize();
    int getOutputSize();
};

#endif

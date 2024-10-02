#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
    {5, 10},
    {6, 12},
    {7, 14},
    {8, 16},
};
#define TRAIN_SIZE sizeof(train) / sizeof(train[0])

// ReLU activation function
float relu(float x) {
    return fmaxf(0.f, x);
}

// Random float generator within a range
float rand_float(float s, float e) {
    return s + ((float)rand() / (float)RAND_MAX) * (e - s);
}

// Cost function with bias
float cost(float w, float bias) {
    float result = 0.0f;
    for (size_t i = 0; i < TRAIN_SIZE; ++i) {
        float x = train[i][0];
        float y = relu(x * w + bias);
        float distance = y - train[i][1];
        result += distance * distance; // Mean squared error
    }
    result /= TRAIN_SIZE; // Average error
    return result;
}

int main() {
    printf("%d",sizeof(size_t));
    return 0;
    srand(time(0));
    float w = rand_float(1, 10); // Initial random weight
    float bias = rand_float(1, 6); // Initial random bias
    float eps = 1e-3;              // Small change for finite difference calculation
    float learningRate = 1e-3;     // Learning rate

    // Gradient descent loop
    for (int i = 0; i < 10000; i++) {
        printf("weight: %f, bias: %f, cost: %f\n", w, bias, cost(w, bias));
        float gradientW = (cost(w + eps, bias) - cost(w, bias)) / eps; // Approximate gradient for weight
        float gradientB = (cost(w, bias + eps) - cost(w, bias)) / eps; // Approximate gradient for bias
        w -= gradientW * learningRate;  // Update weight
        bias -= gradientB * learningRate; // Update bias
    }

    return 0;
}

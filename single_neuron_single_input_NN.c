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

float relu(float x) // non linear activation function
{
    return fmaxf(0.f, x);
}

float rand_float(float s, float e)
{
    return s + ((float)rand() / (float)RAND_MAX) * (e - s);
}
// y = x*w
float cost(float w, float bias) // drawback: less accurate
{
    float result = 0.0f;
    for (size_t i = 0; i < TRAIN_SIZE; ++i)
    {
        float x = train[i][0];
        float y = relu(x * w + bias);
        float distance = y - train[i][1];
        result += distance * distance; // mean sqare distance
    }
    result /= TRAIN_SIZE; // measure of how badly the model preforms, if the value is big, the model is bad
    return result;
}
float derivativeCost(float w) // drawback:derivatives need to be calculated for every cost function, which is now feasible
{
    float result = 0.0f; // derivative of summation of mean sqare distance
    for (size_t i = 0; i < TRAIN_SIZE; ++i)
    {
        float x = train[i][0];
        float y = x * w;
        float distance = 2 * (y - train[i][1]) * w; // derivative of mean sqare distance
        result += distance;
    }
    result /= TRAIN_SIZE; // measure of how badly the model preforms, if the value is big, the model is bad
    return result;
}

void run_derivative(int epochs, float weight, float learningRate)
{
    for (int i = 0; i < epochs; i++)
    {
        // printf("%f\n", derivativeCost(w1));
        float distanceOfCost = derivativeCost(weight); // exact derivative of cost function
        weight -= distanceOfCost * learningRate;

        // printf("%f\n", derivativeCost(w1));
    }
}

int main()
{
    srand(time(0));
    float w = rand_float(1, 10); // what should be multiplied to x to get y
    float bias = rand_float(1, 6);
    float eps = 1e-3;          // nudge the value of w towards minima of cost function;
    float learningRate = 1e-3; // to bring distanceOfCost to a workable range;

    for (int i = 0; i < 10000; i++)
    {
        printf("weight: %f, bias: %f, cost: %f\n", w, bias, cost(w, bias));
        float distanceOfCostforW = (cost(w + eps, bias) - cost(w, bias)) / eps; // finite distance, to approximate the derivative
        float distanceOfCostforBias = (cost(w, bias + eps) - cost(w, bias)) / eps;
        w -= distanceOfCostforW * learningRate;
        bias -= distanceOfCostforBias * learningRate;
    }

    // for 10k epochs without bias
    //  1.999501 -- finite distance    with bias-> weight: 1.998605, bias: 0.005080    with relu -> 
    //  2.000007 -- derivative/gradient descent

    // errors without bias
    // finite distance: 0.000000, derivative: 0.000000
    // finite distance: -0.000499, derivative: 0.000007
    // finite distance: -0.000998, derivative: 0.000015
    // finite distance: -0.001496, derivative: 0.000022
    // finite distance: -0.001995, derivative: 0.000030
    // finite distance: -0.002494, derivative: 0.000037
    // finite distance: -0.002993, derivative: 0.000044
    // finite distance: -0.003491, derivative: 0.000051
    // finite distance: -0.003990, derivative: 0.000059

    // printf("%f\n", w1);

    // for (int i = 0; i < TRAIN_SIZE; i++)
    // {
    //     float res1 = train[i][0] * w - train[i][1];
    //     float res2 = train[i][0] * w1 - train[i][1];
    //     printf("finite distance: %f, derivative: %f\n", res1, res2);
    // }

    return 0;
}
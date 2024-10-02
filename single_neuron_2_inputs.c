#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

#define TRAIN_SIZE sizeof(train) / sizeof(train[0])

float sigmoidf(float x)
{   
    return 1.f / (1.f + expf(-x));
}

float rand_float(float s, float e)
{
    return s + ((float)rand() / (float)RAND_MAX) * (e - s);
}

// y = x1*w1 +x2*w2 + b;
float cost(float w1, float w2, float bias)
{
    float result = 0.0f;
    for (size_t i = 0; i < TRAIN_SIZE; ++i)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1 * w1 + x2 * w2 + bias);
        float distance = y - train[i][2];
        result += distance * distance;
    }
    result /= TRAIN_SIZE;
    return result;
}

int main()
{
    srand(time(0));
    float w1 = rand_float(0, 10);
    float w2 = rand_float(0, 10);
    float bias = rand_float(0, 10);
    float eps = 1e-3;
    float learningRate = 1e-2;

    for (int i = 0; i < 100000; i++)
    {
        float c = cost(w1, w2, bias);
        // printf("weight1: %f, weight2: %f, bias: %f, cost: %f\n", w1, w2, bias, c);
        float distanceOfCostforW1 = (cost(w1 + eps, w2, bias) - c) / eps;
        float distanceOfCostforW2 = (cost(w1, w2 + eps, bias) - c) / eps;
        float distanceOfCostforBias = (cost(w1, w2, bias + eps) - c) / eps;
        w1 -= distanceOfCostforW1 * learningRate;
        w2 -= distanceOfCostforW2 * learningRate;
        bias -= distanceOfCostforBias * learningRate;
    }

    printf("weight1: %f, weight2: %f, bias: %f, cost: %f\n", w1, w2, bias, cost(w1, w2, bias));
    printf("---------------------------\n");
    for (int i = 0; i < TRAIN_SIZE; i++)
    {
        printf("%f | %f = %f, expected: %f\n", train[i][0], train[i][1], sigmoidf(train[i][0] * w1 + train[i][1] * w2 + bias), train[i][2]);
    }

    return 0;
}
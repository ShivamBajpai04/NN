#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
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
float cost(float w1, float w2, float b1, float w3, float w4, float b2, float w5, float w6, float b3)
{
    float result = 0.0f;
    for (size_t i = 0; i < TRAIN_SIZE; ++i)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y1 = sigmoidf(x1 * w1 + x2 * w2 + b1);
        float y2 = sigmoidf(x1 * w3 + x2 * w4 + b2);
        float y3 = sigmoidf(y1 * w5 + y2 * w6 + b3);
        float d3 = y3 - train[i][2];
        result += d3 * d3;
    }
    result /= TRAIN_SIZE;
    return result;
}

int main()
{
    srand(time(0));
    float w1 = rand_float(0, 1);
    float w2 = rand_float(0, 1);
    float b1 = rand_float(0, 1);
    float w3 = rand_float(0, 1);
    float w4 = rand_float(0, 1);
    float b2 = rand_float(0, 1);
    float w5 = rand_float(0, 1);
    float w6 = rand_float(0, 1);
    float b3 = rand_float(0, 1);
    float eps = 1e-3;
    float learningRate = 1e-2;

    for (int i = 0; i < 1000000; i++)
    {
        float c = cost(w1, w2, b1, w3, w4, b2, w5, w6, b3);
        // printf("weight1: %f, weight2: %f, bias: %f, cost: %f\n", w1, w2, bias, c);
        float dw1 = (cost(w1 + eps, w2, b1, w3, w4, b2, w5, w6, b3) - c) / eps;
        float dw2 = (cost(w1, w2 + eps, b1, w3, w4, b2, w5, w6, b3) - c) / eps;
        float db1 = (cost(w1, w2, b1 + eps, w3, w4, b2, w5, w6, b3) - c) / eps;
        float dw3 = (cost(w1, w2, b1, w3 + eps, w4, b2, w5, w6, b3) - c) / eps;
        float dw4 = (cost(w1, w2, b1, w3, w4 + eps, b2, w5, w6, b3) - c) / eps;
        float db2 = (cost(w1, w2, b1, w3, w4, b2 + eps, w5, w6, b3) - c) / eps;
        float dw5 = (cost(w1, w2, b1, w3, w4, b2, w5 + eps, w6, b3) - c) / eps;
        float dw6 = (cost(w1, w2, b1, w3, w4, b2, w5, w6 + eps, b3) - c) / eps;
        float db3 = (cost(w1, w2, b1, w3, w4, b2, w5, w6, b3 + eps) - c) / eps;
        w1 -= dw1;
        w2 -= dw2;
        b1 -= db1;
        w3 -= dw3;
        w4 -= dw4;
        b2 -= db2;
        w5 -= dw5;
        w6 -= dw6;
        b3 -= db3;
    }

//XOR = INH(AND(a,b),OR(a,b))
//INH = AND(!a, b)
//INH stands for inhibition or Not-if or Reverse Implication

    printf("w1: %f, w2: %f, b1: %f, w3: %f, w4: %f, b2: %f, w5: %f, w6: %f, b3: %f, cost: %f\n", w1, w2, b1, w3, w4, b2, w5, w6, b3, cost(w1, w2, b1, w3, w4, b2, w5, w6, b3));

    // printf("---------------------------\n");
    // for (int i = 0; i < TRAIN_SIZE; i++)
    // {
    //     float y1 = sigmoidf(w1 * train[i][0] + w2 * train[i][1] + b1);
    //     printf("%f ^ %f = %f, expected: %f\n", train[i][0], train[i][1], y1, train[i][2]);
    // }

    // printf("---------------------------\n");
    // for (int i = 0; i < TRAIN_SIZE; i++)
    // {
    //     float y1 = sigmoidf(train[i][0] * w3 + train[i][1] * w4 + b2);
    //     printf("%f ^ %f = %f, expected: %f\n", train[i][0], train[i][1], y1, train[i][2]);
    // }
    // printf("---------------------------\n");
    // for (int i = 0; i < TRAIN_SIZE; i++)
    // {
    //     float y1 = sigmoidf(train[i][0] * w5 + train[i][1] * w6 + b3);
    //     printf("%f ^ %f = %f, expected: %f\n", train[i][0], train[i][1], y1, train[i][2]);
    // }
    printf("---------------------------\n");
    for (int i = 0; i < TRAIN_SIZE; i++)
    {
        float y1 = sigmoidf(w1 * train[i][0] + w2 * train[i][1] + b1);
        float y2 = sigmoidf(w3 * train[i][0] + w4 * train[i][1] + b2);
        printf("%f ^ %f = %f, expected: %f\n", train[i][0], train[i][1], sigmoidf((y1 * w5 + y2 * w6 + b3)), train[i][2]);
    }

    return 0;
}
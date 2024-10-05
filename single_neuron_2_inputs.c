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

#define TRAIN_SIZE (sizeof(train) / sizeof(train[0]))

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float rand_float(float s, float e)
{
    return s + ((float)rand() / (float)RAND_MAX) * (e - s);
}

// y = x1*w1 + x2*w2 + b
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
    result /= TRAIN_SIZE; // Mean squared error
    return result;
}

int main()
{
    srand(time(0));
    float w1 = rand_float(0, 1); // Start with smaller random weights
    float w2 = rand_float(0, 1);
    float bias = rand_float(0, 1);
    float eps = 1e-5;             // Smaller epsilon for better gradient approximation
    float learningRate = 1e-3;    // Smaller learning rate

    for (int i = 0; i < 10000000; i++)
    {
        float c = cost(w1, w2, bias);

        if (i % 100000 == 0) // Print every 1000 iterations to avoid clutter
        {
            printf("Iteration %d: weight1 = %f, weight2 = %f, bias = %f, cost = %f\n", i, w1, w2, bias, c);
        }

        // Approximate the gradients
        float dcost_w1 = (cost(w1 + eps, w2, bias) - c) / eps;
        float dcost_w2 = (cost(w1, w2 + eps, bias) - c) / eps;
        float dcost_bias = (cost(w1, w2, bias + eps) - c) / eps;

        // Update weights and bias
        w1 -= learningRate * dcost_w1;
        w2 -= learningRate * dcost_w2;
        bias -= learningRate * dcost_bias;
    }

    // Final weights and cost
    printf("Final: weight1 = %f, weight2 = %f, bias = %f, cost = %f\n", w1, w2, bias, cost(w1, w2, bias));

    // Check model on training data
    printf("---------------------------\n");
    for (int i = 0; i < TRAIN_SIZE; i++)
    {
        float prediction = sigmoidf(train[i][0] * w1 + train[i][1] * w2 + bias);
        printf("%f | %f = %f, expected: %f\n", train[i][0], train[i][1], prediction, train[i][2]);
    }

    return 0;
}

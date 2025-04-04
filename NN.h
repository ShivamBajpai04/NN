#ifndef NN_H_
#define NN_H_
#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert

#endif // NN_ASSERT

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

/**
 * @brief Generates a random float between 0 and 1.
 * @return Random float value.
 */
float rand_float(void);

/**
 * @brief Computes the sigmoid function for a given input.
 * @param x Input value.
 * @return Sigmoid of the input.
 */
float sigmoidf(float x);

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *data;
} Mat;

#define MAT_AT(m, i, j) (m).data[(i) * (m).stride + (j)]

/**
 * @brief Allocates memory for a matrix with the specified dimensions.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return Allocated matrix.
 */
Mat mat_alloc(size_t rows, size_t cols);

/**
 * @brief Fills a matrix with a specified value.
 * @param m Matrix to fill.
 * @param val Value to fill the matrix with.
 */
void mat_fill(Mat m, float val);

/**
 * @brief Randomizes the elements of a matrix within a specified range.
 * @param m Matrix to randomize.
 * @param s Start of the range.
 * @param e End of the range.
 */
void mat_rand(Mat m, float s, float e);

/**
 * @brief Extracts a specific row from a matrix as a new matrix.
 * @param m Source matrix.
 * @param row Index of the row to extract.
 * @return A matrix representing the specified row.
 */
Mat mat_row(Mat m, size_t row);

/**
 * @brief Copies the contents of one matrix to another.
 * @param dest Destination matrix.
 * @param src Source matrix.
 */
void mat_copy(Mat dest, Mat src);

/**
 * @brief Computes the dot product of two matrices and stores the result in a destination matrix.
 * @param a First operand matrix.
 * @param b Second operand matrix.
 * @param dest Destination matrix to store the result.
 */
void mat_dot(Mat a, Mat b, Mat dest);

/**
 * @brief Adds two matrices element-wise and stores the result in the destination matrix.
 * @param dest Destination matrix to store the result.
 * @param b Matrix to add to the destination matrix.
 */
void mat_sum(Mat dest, Mat b);

/**
 * @brief Prints the contents of a matrix to the console.
 * @param m Matrix to print.
 * @param name Name of the matrix (for display purposes).
 */
void mat_print(Mat m, const char *name, size_t padding);

/**
 * @brief Applies the sigmoid function to each element of a matrix.
 * @param m Matrix to apply the sigmoid function to.
 */
void mat_sigmoid(Mat m);

#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct
{
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activations is count + 1
} NN;

/**
 * @brief Allocates memory for a neural network based on the given architecture.
 * @param arch Array representing the number of neurons in each layer.
 * @param arch_count Number of layers in the architecture.
 * @return Allocated neural network.
 */
NN nn_alloc(size_t *arch, size_t arch_count);

/**
 * @brief Prints the details of a neural network, including weights and biases.
 * @param nn Neural network to print.
 * @param name Name of the neural network (for display purposes).
 */
void nn_print(NN nn, const char *name);

#define NN_PRINT(nn) nn_print(nn, #nn)

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = (float *)malloc(sizeof(*m.data) * rows * cols);
    NN_ASSERT(m.data != NULL);
    return m;
}
void mat_dot(Mat a, Mat b, Mat dest)
{
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(a.rows == dest.rows);
    NN_ASSERT(b.cols == dest.cols);

    for (size_t i = 0; i < dest.rows; ++i)
    {
        for (size_t j = 0; j < dest.cols; ++j)
        {
            float s = 0.0f;
            for (size_t k = 0; k < b.rows; ++k)
            {
                s += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
            MAT_AT(dest, i, j) = s;
        }
    }
}
void mat_sum(Mat dest, Mat b)
{
    NN_ASSERT(dest.rows == b.rows);
    NN_ASSERT(dest.cols == b.cols);
    for (size_t i = 0; i < dest.rows; ++i)
    {
        for (size_t j = 0; j < b.cols; ++j)
        {
            MAT_AT(dest, i, j) += MAT_AT(b, i, j);
        }
    }
}
void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; ++i)
    {
        printf("%*s    ", (int)padding, "");
        for (size_t j = 0; j < m.cols; ++j)
        {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

void mat_rand(Mat m, float s, float e)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            MAT_AT(m, i, j) = rand_float() * (e - s) + s;
        }
    }
}

void mat_fill(Mat m, float val)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            MAT_AT(m, i, j) = val;
        }
    }
}

void mat_sigmoid(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .data = &MAT_AT(m, row, 0),
    };
}

void mat_copy(Mat dest, Mat src)
{
    NN_ASSERT(dest.rows == src.rows);
    NN_ASSERT(dest.cols == src.cols);
    for (size_t i = 0; i < dest.rows; ++i)
    {
        for (size_t j = 0; j < dest.cols; ++j)
        {
            MAT_AT(dest, i, j) = MAT_AT(src, i, j);
        }
    }
}

// size_t arch[] = {2,2,1};
// NN nn = nn_alloc(arch,ARRAY_LEN(arch));

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);
    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as) * nn.count + 1);
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);
    for (size_t i = 1; i < arch_count; ++i)
    {
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        nn.as[i] = mat_alloc(1, arch[i]);
    }
    return nn;
}

void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; ++i)
    {
        snprintf(buf, sizeof(buf), "ws[%zu]", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs[%zu]", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

#endif // NN_IMPLEMENTATION
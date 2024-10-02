#ifndef NN_H_
#define NN_H_
#include <stddef.h>
#include <stdio.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert

#endif // NN_ASSERT

typedef struct
{
    size_t rows;
    size_t cols;
    // size_t stride;
    float *data;
} Mat;

#define MAT_AT(m, i, j) (m).data[(i) * (m).cols + (j)]

float rand_float(void);

Mat mat_alloc(size_t rows, size_t cols); // allocate memory to the matrix dynamically
void mat_fill(Mat m, float val);         // fill the matrix with val
void mat_rand(Mat m, float s, float e);  // randomize the matrixx
void mat_dot(Mat a, Mat b, Mat dest);    // dest is output, and a and b are operands; No memory allocation outside of mat_alloc()
void mat_sum(Mat dest, Mat b);           // add two matrices
void mat_print(Mat m);                   // print the contents of the matrix
#endif                                   // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = malloc(sizeof(*m.data) * rows * cols);
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
void mat_print(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
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
#endif // NN_IMPLEMENTATION
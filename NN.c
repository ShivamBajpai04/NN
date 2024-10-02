#define NN_IMPLEMENTATION
// #define NN_MALLOC my_malloc
#include "NN.h"
#include <time.h>

float train_input[] = {0, 0, 1, 0, 0, 1, 1, 1};
float train_output[] = {0, 1, 1, 0};

int main()
{
    srand(time(0));
    Mat w1 = mat_alloc(2, 2), b1 = mat_alloc(1, 2); // one layer with 2 neurons
    Mat w2 = mat_alloc(2, 1), b2 = mat_alloc(1, 1); // one layer with 1 neurons

    mat_rand(w1,0,10);
    mat_rand(b1,0,10);
    mat_rand(w2,0,10);
    mat_rand(b2,0,10);

    Mat input = {.rows = 4, .cols = 2, .data = train_input};
    Mat output = {.rows = 1, .cols = 4, .data = train_output};
    MAT_PRINT(w1);
    MAT_PRINT(b1);
    MAT_PRINT(w2);
    MAT_PRINT(b2);
}
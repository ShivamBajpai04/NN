#define NN_IMPLEMENTATION
// #define NN_MALLOC my_malloc
#include "NN.h"
#include<time.h>
int main()
{
    srand(time(0));
    Mat m = mat_alloc(1, 2);
    Mat m1 = mat_alloc(2, 3);
    Mat m2 = mat_alloc(1, 3);
    // mat_rand(m, 0, 1);
    // mat_rand(m1, 0, 9);
    mat_fill(m,2);
    mat_fill(m1,2);
    mat_print(m);
    mat_print(m1);
    // mat_print(m2);
    printf("\n");
    mat_dot(m,m1,m2);
    mat_print(m2);
}
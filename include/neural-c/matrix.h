#include "./vector.h"

typedef struct {
        vector_t vectors;
        size_t   vector_count;
} matrix_t;

matrix_t matrix_new(size_t x, size_t y);

/**
 * Multiply matrix A and B and return the result.
 *
 * @return The product of A and B; matrix of dimensions AX by BY
 */
matrix_t **matrix_mult(matrix_t A, size_t ax, size_t ay, 
                       matrix_t B, size_t bx, size_t by
                       );

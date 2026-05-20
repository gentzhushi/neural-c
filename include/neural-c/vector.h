#include <stddef.h>

typedef struct {
        double *data;
        double count;
} vector_t;

/**
 * Create a new empty vector of a specified size
 */
vector_t *vector_new(size_t init_capacity);

/**
 * Calculate and return the scalar (dot) product of 2 vectors
 */
double vector_scalar(const vector_t *x, const vector_t *y, size_t vsize);

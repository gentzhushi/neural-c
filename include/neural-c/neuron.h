#include <stddef.h>

#include "./vector.h"

typedef struct {
        vector_t params;
} neuron_t;

neuron_t *neuron_new(double* params, size_t psize);
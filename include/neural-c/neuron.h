#include <stddef.h>

typedef struct {
        double *data;
        size_t data_count;
} neuron_t;

neuron_t *neuron_new(double* params, size_t psize);

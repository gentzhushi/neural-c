#include <stddef.h>
#include "./neuron.h"

typedef struct {
        neuron_t *neurons;
        size_t   size;
} neuron_layer_t;

/**
 * Create new neuron layer
 */
neuron_layer_t neuron_layer_new(size_t size);

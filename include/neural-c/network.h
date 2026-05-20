#include "./layer.h"

typedef struct {
        neuron_layer_t *layers;
        size_t         layer_count;
} neural_network_t;

double *forward(const neural_network_t *net, double *input);

void train(neural_network_t *net);

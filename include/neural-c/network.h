#include "./layer.h"

typedef struct {
        neuron_layer_t *layers;
        size_t         layer_count;
} neural_network_t;

neural_network_t neural_network_new(const neuron_layer_t* layers, size_t layer_count);

double *neural_network_forward(const neural_network_t *net, double *input);

void neural_network_train(neural_network_t *net);

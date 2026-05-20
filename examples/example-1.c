#include <stdio.h>

#include "../include/neural-c/network.h"

#define LAYER_COUNT 4
 
int main(void)
{
        int neuron_counts[LAYER_COUNT] = {784, 16, 16, 10};

        neuron_layer_t layers[LAYER_COUNT];

        for (int i = 0; i < LAYER_COUNT; i++)
                layers[i] = neuron_layer_new(neuron_counts[i]);

        neural_network_t network = neural_network_new(layers, LAYER_COUNT);

        neural_network_train(&network);

        double *input;

        double *output = neural_network_forward(&network, input);

        for (int i = 0; i < neuron_counts[LAYER_COUNT - 1]; i++)
                printf("Neuron %d: %lf\n", i + 1, output[i]);

        return 0;
}

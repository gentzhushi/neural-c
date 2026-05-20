#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FEATURE_COUNT 3

double dataset[][FEATURE_COUNT] = {
	{0, 0,  0},
	{1, 0,  2},
	{0, 1,  2},
	{1, 1,  4},

	{2, 0,  4},
	{2, 1,  6},
	{1, 2,  6},
	{2, 2,  8},

	{3, 0,  6},
	{3, 1,  8},
	{3, 2, 10},
	{3, 3, 12},
	{0, 3,  6},
	{1, 3,  8},
	{2, 3, 10}
};

#define DATASET_SIZE (sizeof(dataset)/sizeof(dataset[0]))

double	learningRate	= 1e-4;
int		trainingEpochs	= 1e+6;
double	epsilon			= 1e-1; /* Used to calculate derivative */

/**
 * This returns the dot product of theta and input, it expects x to be an array
 * of inputs without the synthetic x_0 = 1.
 */
double infer(const double *theta, double *input)
{
	double sum = theta[0]; // * input[0] == 1

	for (int i = 0; i < FEATURE_COUNT - 1; i++)
		sum += theta[i+1] * input[i];

	return sum;
}

/**
 * Mean squared error loss function. ∑(y-predictedY)²
 */
double loss(const double *theta)
{
	double loss = 0;

	for (int i = 0; i < DATASET_SIZE; i++) {
		double 	  actualY = dataset[i][FEATURE_COUNT-1];
		double predictedY = infer(theta, dataset[i]);
		loss += pow(actualY - predictedY, 2.0);
	}

	return loss;
}

/**
 * Partial derivative of the loss function (MSE) with respect to theta[i];
 */
double loss_pd_wrt(const double *theta, int i) 
{
	if (i >= FEATURE_COUNT) {
		fprintf(stderr, "loss_pd_wrt: provided i is out of bounds.\n");
		exit(EXIT_FAILURE);
	}
	double thetaCopy[FEATURE_COUNT];
	memcpy(thetaCopy, theta, FEATURE_COUNT * sizeof(double));

	thetaCopy[i] += epsilon;
	double forward = loss(thetaCopy);

	thetaCopy[i] -= 2 * epsilon;
	double backward = loss(thetaCopy);

	return (forward - backward)/2 * epsilon;
}

/**
 * Stochastic gradient descent of the MSE loss function
 */
void trainModel(double *theta)
{
	for (int k = 0; k < trainingEpochs; k++) {
		for (int i = 0; i < FEATURE_COUNT; i++) {
			// Θ_j := Θ_j - ⍺*∂/∂Θ_j(J)
			theta[i] -= learningRate * loss_pd_wrt(theta, i);
			double newLoss = loss(theta);
		}
	}
}

int main(int argc, char** argv)
{
	if (argc < FEATURE_COUNT) {
		fprintf(stderr, "Insufficient inputs.\n");
		exit(EXIT_FAILURE);
	} else if (argc > FEATURE_COUNT) {
		fprintf(stderr, "Surplus of inputs, ignoring the extras.\n");
	}

	double theta[FEATURE_COUNT] = {0};
	double input[FEATURE_COUNT - 1] = {0};

	trainModel(theta);

	// Populate the input array
	for (int i = 0; i < FEATURE_COUNT - 1; i++)
		input[i] = atof(argv[i+1]);

	double output = infer(theta, input);
	
	// Print '[INFERENCE] (x0, ..., xN) -> y' 
	printf("[INFERENCE] (");
	for (int i = 0; i < FEATURE_COUNT - 2; i++)
		printf("%lf, ", input[i]);
	printf("%lf) -> %lf\n", input[FEATURE_COUNT - 2], output);

	return 0;
}

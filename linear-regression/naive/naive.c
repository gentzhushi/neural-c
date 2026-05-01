#include <stdio.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FEATURE_COUNT 	2ul

double dataset[][FEATURE_COUNT] = { // x and y
	{0,  0},
	{1,  2},
	{2,  4},
	{3,  6},
	{4,  8},
	{5, 10},
	{6, 12},
	{7, 14},
	{8, 16}
};

#define DATASET_SIZE 	(sizeof(dataset)/sizeof(dataset[0]))
#define LEARNING_RATE 	0.000545
#define TRAINING_EPOCHS 1000
#define EPSILON 		0.01 /* Used to calculate derivative */

double
infer(const double *theta, double input)
{
	return theta[0] + theta[1] * input;
}

double
loss(const double *theta)
{
	double loss = 0;

	for (int i = 0; i < DATASET_SIZE; i++)
	{
		double 	  actualY = dataset[i][1];
		double predictedY = infer(theta, dataset[i][0]);
		loss += pow(actualY - predictedY, 2.0);
	}
	return loss;
}

// Partial derivative of the loss function (MSE) with respect to theta[i];
double
loss_pd_wrt(const double *theta, int i) 
{
	if (i >= FEATURE_COUNT)
	{
		fprintf(stderr, "loss_pd_wrt: provided i is out of bounds.\n");
	}
	double thetaCopy[FEATURE_COUNT];
	memcpy(thetaCopy, theta, FEATURE_COUNT * sizeof(double));
	thetaCopy[i] += EPSILON;
	double derivative = (loss(thetaCopy) - loss(theta)) / EPSILON;
	// printf("Derivative for theta[%d] is: %f\n", i, derivative);
	return derivative;
}

void
trainModel(double *theta)
{
	for (int k = 0; k < TRAINING_EPOCHS; k++)
	{
		double currentLoss = loss(theta);
		for (int i = 0; i < FEATURE_COUNT; i++)
		{
			// -= ALPHA * PARTIAL_DERIVATIVE(LOSS, THETA[i])
			theta[i] -= LEARNING_RATE * loss_pd_wrt(theta, i);
			double newLoss = loss(theta);
		}
	}
}

void
printModel(const char* description, const double *theta)
{
	printf( "---------------------------------------------------------------\n"
			"%s:\n\tTheta (weights): ", description);
	for (int i = 0; i < FEATURE_COUNT; i++)
	{
		printf("w[%d]: %f ", i, theta[i]);
	}
	printf( "\n\tLoss: %f\n"
			"---------------------------------------------------------------\n"
			, loss(theta));
}

int
main(int argc, char** argv)
{
	double theta[FEATURE_COUNT] = {0};

	printModel("Model BEFORE training", theta);
	trainModel(theta);
	printModel("Model AFTER training", theta);
	
	for (int i = 1; i < argc; i++)
	{
		double input = atof(argv[i]);
		double output = infer(theta, input);
		printf("[INFERENCE] %f -> %f\n", input, output);
	}

	return 0;
}

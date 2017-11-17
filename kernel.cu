
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Network.h"
#include <stdio.h>
#include <cuda.h>

int main()
{

	ErrorHandler *error = &ErrorHandler::getInstance();
	Network train;
	float learningrate = 0.05;
	int trainiter = 1;
	train.train(false,false, learningrate, trainiter);
	train.test(false);

	system("Pause");
}

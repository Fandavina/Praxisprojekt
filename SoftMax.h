#pragma once
#include "Layer.h"
class SoftMax:public Layer
{
public:
	void printDev(int dimension);
	void printGrad(int dimension);
	int getTypeId();

};


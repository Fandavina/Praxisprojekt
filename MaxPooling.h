#pragma once
#include "Layer.h"


class MaxPoolLayer:public Layer
{
public:

	int size, stride;

	cudnnPoolingDescriptor_t Descr = NULL;

	void createDescr();
	void destroyDescr();

	MaxPoolLayer(int size_, int stride_, int batchSize);

	void set(int size_, int stride_, int batchSize);
	
	MaxPoolLayer() { createDescr(); };
	void setDescr();
	void printDev(int dimension);
	void printGrad(int dimension);

	int getTypeId();
};
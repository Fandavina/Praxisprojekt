#pragma once
#include "Layer.h"
/*ActivationLayer extends Layer*/
class Activation:public Layer
{
public:

	cudnnActivationDescriptor_t Descr = NULL;
	Activation();
	int getTypeId();
	~Activation();
	void createDescr();
	void destroyDescr();
	void setDescr();
	Activation(int batchsize, Layer * prevLayer);
	
	void printDev(int dimension);
	void printGrad(int dimension);

};


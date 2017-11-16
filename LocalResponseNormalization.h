#pragma once
#include "Layer.h"

class LocalResponseNormalization:public Layer
{
public:
	cudnnLRNDescriptor_t Descr=NULL;
	LocalResponseNormalization();
	~LocalResponseNormalization();
	void createDescr();
	void destroyDescr();
	void setDescr();
	void printDev(int dimension);
	void printGrad(int dimension);
	int getTypeId();
};


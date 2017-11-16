#pragma once
#include "Layer.h"
class FullyConnectedLayer:public Layer
{
public:
    std::vector<float> pneurons, pbias;
	
	//DEVICE DATA
	float*ptrtoDevBias = nullptr;
	float*ptrtoDevNeuron = nullptr;

	float* ptrToGradDevNeuron = nullptr;
	float* ptrToGradDevBias = nullptr;

	FullyConnectedLayer() { Layer::createDescr(); };
	~FullyConnectedLayer() { Layer::destroyDescr(); }
	void setDescr();

	FullyConnectedLayer(int outputs_, int batchSize);

	void set(int outputs_, int batchSize);

	void initLayer(const char *fileprefix);
	void saveLayer(const char *fileprefix);
	void copyToDevFwD();
	void copyToDevBwD();
	void initRandom();
	void copyToHost();
	void printDevNB(int dimension);
	void printHostNB(int dimension);
	void printDev(int dimension);

	void printGrad(int dimension);

	int getTypeId();

	void printGradCB(int dimension);

	void mallocDevBwD();
	void mallocDevFwD();
	void freeDevFwDData();
	void freeDevBwDData();
};
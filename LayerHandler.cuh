#pragma once

#include "Conv.h"
#include "MaxPooling.h"
#include "FullyConnected.h"
#include "LocalResponseNormalization.h"
#include "Activation.h"
#include "SoftMax.h"
#include "DataLayer.h"
#include "LblLayer.cuh"

#include <map>


class LayerHandler{
public:
	//LAYER
	Layer*firstLayer = &datalayer;
	Layer*lastLayer = &lbllayer;
	DataLayer datalayer;
	std::vector<ConvLayer>convlayers;
	std::vector<MaxPoolLayer>poollayers;
	std::vector<FullyConnectedLayer> fulls;
	std::vector<Activation> activs;
	std::vector<LocalResponseNormalization> lrns;
	std::vector<SoftMax> sfts;
	LblLayer lbllayer;

	LayerHandler() { };
	~LayerHandler() { };
	void freeDevBwD();
	void setLayer(int channels, int width, int height, int batchSize);
	void setDescr(size_t & workspacesize, bool withWorkspace);
	void initLayer(bool pretrained);
	void saveLayer();
	void copyLayertoDev();
	void copytoLayertoDevBwD();
	void copyLayertoHost();
	void copytoDevData(float * imagesfloat, int imageID);
	void copytoHostData(float *& imagesfloat);
	void copytoDevDiff(float * labelsfloat);
	void copytoHostLabelwComp(float *& labelsfloat);
	void printDev(int dimension);
	void printGrad(int dimension);
	void freeDevFwD();
	void freeDev();
	void init();
private:
	//Controlbool 
	bool devMallocFwD = false;
	bool devMallocBwD = false;
	bool layerSet = false;
	int baSize;
	int inputsize;
	int outputsize;
	std::map<int, int> outcfull;
	std::map<int, int> outcconv;

	ErrorHandler *error = &ErrorHandler::getInstance();
	std::string GetCurrentWorkingDir(void);

};
#pragma once
#include "Layer.h"
class DataLayer:public Layer
{
public:

	void set(int channels, int width, int height, int batchSize);
	void copytoDevData(float * imagesfloat, int imageID);

	void copytoHostData(float *& imagesfloat);

	void printDev(int dimension);

	int getTypeId();   

};


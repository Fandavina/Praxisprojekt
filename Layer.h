#pragma once
#include <cudnn.h>
#include <cuda.h>
#include <cublas.h>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <vector>
#include <iostream>
#include <random>

#include "handle.cuh"
#include "Error.h"

const cudnnTensorFormat_t tensorformat =	 CUDNN_TENSOR_NCHW;
const cudnnDataType_t dataType =			 CUDNN_DATA_FLOAT;
const cudnnNanPropagation_t propagateNan =	 CUDNN_PROPAGATE_NAN;

const cudnnConvolutionMode_t convMode =		 CUDNN_CROSS_CORRELATION;
const cudnnPoolingMode_t poolMode =			 CUDNN_POOLING_MAX;
const cudnnActivationMode_t activationmode = CUDNN_ACTIVATION_SIGMOID;

const unsigned int  lrnN =		 5;
const double		lrnAlpha =	 0.0001;
const double		lrnBeta =	 0.75;
const double		lrnK =		 1.0;

/*Minimum Layerstruct*/
class Layer
{
public:
	/*OutputTensor*/
	cudnnTensorDescriptor_t DstTensor = NULL;
	/*OutputTensor of Prev Layer*/
	cudnnTensorDescriptor_t* SrcTensor = nullptr;
	
	/*Outputdimension*/
	int outchannel, outwidth, outheight;
	int outputsize = 0;
	/*Inputdimension*/
	int inchannel, inwidth, inheight;
	int inputsize = 0;
	/*Batch Size*/
	int baSize;

	/*Outputdata*/
	float *ptrToOutData = nullptr;
	/*Reverse Calculated InputData*/
	float *ptrToGradData = nullptr;

	/*Layer before*/
	Layer*prevLayer = nullptr;
	/*Layer after*/
	Layer*nextLayer = nullptr;

	Layer();
	void set(int batchsize);
	void initpre(Layer * preLayer);	
	~Layer();	

	virtual void freeDevFwD();
	virtual void freeDevBwD();

	virtual void createDescr();
	virtual void setDescr();
	virtual void destroyDescr();

	virtual void mallocDevFwD();
	virtual void mallocDevBwD();

	/*Print OutputData*/
	virtual void printDev(int dimension) {};
	/*Print Calculated InputData*/
	virtual void printGrad(int dimension) {};
	
	/*Identify childLayers*/
	virtual int getTypeId();
	enum LayerID { ParentLayer = 0, Activation = 1, Conv = 2, DataLayer = 3, FullyConnectedLayer = 4, LblLayer = 5, LocalResponseNormalization = 6, MaxPooling = 7, SoftMax = 8 };

protected:
	ErrorHandler *error = &ErrorHandler::getInstance();
	Handle *handle;
	/*Printfunctions*/
	void printDev(int dimension, std::string name);
	void printGrad(int dimension, std::string name);
	/*HelperPrintfunctions*/
	void printptrDev(std::string name, float * devptr, int dimension, size_t size);
	void printHostptr(std::string name, float * ptr, int dimension, size_t size);
	void printHost(std::string name, std::vector<float> hostdata, int dimension, size_t size);
	void randomGenerator(size_t sizerand, size_t sizevec,std::vector<float>&res);
	float RandomFloat(float min, float max);
	void set(int batchsize, int outc, int outh, int outw);
};


#pragma once

#include "Layer.h"

class ConvLayer:public Layer
{
public:
	int kernel_size;

	std::vector<float> pconv, pbias;
	//DEVICE DATA
	float* ptrToDevConv=nullptr;
	float* ptrToDevBias = nullptr;
	float* ptrToGradDevConv = nullptr;
	float* ptrToGradDevBias = nullptr;
	
	//CUDNN DESCRIPTOR
	cudnnTensorDescriptor_t BiasTensorDescr = NULL;
	cudnnFilterDescriptor_t FilterDescr = NULL;
	cudnnConvolutionDescriptor_t Descr = NULL;
	cudnnConvolutionFwdAlgo_t AlgoFwd;
	cudnnConvolutionBwdFilterAlgo_t AlgoBwd;
	cudnnConvolutionBwdDataAlgo_t AlgoDataBwd;

	bool dataAlgo=false;

	ConvLayer();
	~ConvLayer();

	ConvLayer(int out_channels_, int kernel_size_, int batchsize);

	void set(int out_channels_, int kernel_size_, int batchsize);
	
	void initLayer(const char *fileprefix);
	void saveLayer(const char *fileprefix);
	
	void createDescr();
	void destroyDescr();

	void setDescr(size_t & workspace, bool setDataAlgo, bool withWorkspace);

	
	void initRandom();
	void copyToDevFwD();
	void copyToHost();

	void printDevCB(int dimension);
	void printDev(int dimension);

	void printGrad(int dimension);
	int getTypeId();
	void printGradCB(int dimension);

	void printHostCB(int dimension);

	void mallocDevFwD();
	void mallocDevBwD();

	void freeDevFwDData();
	void freeDevBwDData();
private:
	size_t SetFwdConvolutionTensors(bool withWorkspace);
	size_t SetBwdConvolutionTensors(bool setDataAlgo, bool withWorkspace);

};
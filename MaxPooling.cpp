#include "MaxPooling.h"
	void MaxPoolLayer::createDescr(){
		Layer::createDescr();
	 	if(Descr==NULL)error->checkError(cudnnCreatePoolingDescriptor(&Descr));
	}
	void MaxPoolLayer::destroyDescr(){
		Layer::destroyDescr();
	 	if(Descr!=NULL){ error->checkError(cudnnDestroyPoolingDescriptor(Descr)); Descr=NULL;}
	}
    MaxPoolLayer::MaxPoolLayer(int size_, int stride_,int batchSize)  {
		set(size_,stride_,batchSize);
		createDescr();
	}
	void MaxPoolLayer::set(int size_, int stride_,int batchSize){
		size=size_;
		stride=stride_;
		Layer::set(batchSize, prevLayer->outchannel, prevLayer->outheight / stride, prevLayer->outwidth / stride);
	}
		
	void MaxPoolLayer::setDescr(){
		Layer::setDescr();
		error->checkError(cudnnSetPooling2dDescriptor(Descr, poolMode, propagateNan,size, size, 0, 0, stride, stride));
	}

	void MaxPoolLayer::printDev(int dimension) {
		Layer::printDev(dimension, "Pool ");
	}
	void MaxPoolLayer::printGrad(int dimension) {
		Layer::printGrad(dimension, "Pool ");
	}
	int MaxPoolLayer::getTypeId()
	{
		return LayerID::MaxPooling;
	}
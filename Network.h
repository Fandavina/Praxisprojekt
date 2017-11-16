#pragma once
#include "ImageHandler.h"
#include "PropagationHelper.cuh"
class Network {
public:
	/*Standard*/
	Network();
	/*Constructor*/
	Network(int workgpu, int batchsize, dim3 imagedim, bool withWorkspace);
	/*Init Network and all its layers*/
	void set(int workgpu, int batchsize, dim3 imagedim, bool withWorkspace);
	
	/*
	Init Network Random depending on pretrained
	pretrained = false => initRandom
	*/
	void initNetwork(bool pretrained);
	/*Save NetworkLayer(Conv,Full) to Saved File*/
	void saveNetwork();
	/*pretrained = false => initRandom*/
	void train(bool pretrained, bool withworkspace, float learningrate, int inter);
	/*Predict the Class of the Image */
	float * ForwardPropagation(float * imagesfloat);
	/*Destroys the Network and reset the Grafikcard*/
	~Network();

private:
	size_t workspaceSize=0;
	int numgpu = 0, workinggpu, baSize;
	dim3 imagedimension;
	int imagesize;
	LayerHandler layerhandle;
	ErrorHandler *error = &ErrorHandler::getInstance();
	PropagationHelper propa;

	/*Propagates Forward*/
	float * FwdPropa(bool controloutput);
	/*Propagates Backward*/
	void BwdPropa(bool controloutput);
	/*Predict the Class of the Image beginning at ID*imagesize without Copying the Layer to Dev(must be completed)*/
	float * ForwardPropagationwCopy(float * imagesfloat, int ID, bool controloutput);
	/*Goes UpsideDown through the layerstruct*/
	void BackwardPropagation(float * labelfloat, bool controloutput);
	/*Changes Weigth depending on results of Forward-&Backwardpropagation*/
	void UpdateWeigth(float learningrate, bool output);
	/*makes an floatptr out of the data in images*/
	float * makeImageFloat(std::vector<Image> images);
	/*make an floatptr out of the label in images*/
	float * makeLabelfloat(std::vector<Image> images);
	/*Print the ptr*/
	void printptr(float * ptr, int size);
	/*Copy the ptr to Host and than print it*/
	void printDevptr(float * ptr, int size);	
};
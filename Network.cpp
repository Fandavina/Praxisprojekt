#include "Network.h"

Network::Network() {
	error->checkError(cudaGetDeviceCount(&numgpu));
	int cudnnversion = (int)cudnnGetVersion();
	int cudartversion = (int)cudnnGetCudartVersion();
	if (cudnnversion < 7003)error->warning("Cudnn Version < 7003");
	if (cudartversion < 9000)error->warning("Cudart Version < 9000");
}

Network::Network(int workgpu, int batchsize, dim3 imagedim, bool withWorkspace) {
	Network();
	set(workgpu, batchsize, imagedim, withWorkspace);
}

void Network::set(int workgpu, int batchsize, dim3 imagedim, bool withWorkspace) {
	if (workgpu >= numgpu) {
		std::string errorstr = "Es ist keine GPU NR." + ((char)workgpu - (char)0);
		errorstr += " vorhanden";
		error->throwError(errorstr);
	}
	/*cudaDeviceProp prop; //TODO Used Memory > GPU Memory
	cudaGetDeviceProperties(&prop, 0);*/
	baSize = batchsize;
	workinggpu = workgpu;
	error->checkError(cudaSetDevice(workinggpu));
	imagedimension = imagedim;
	imagesize = imagedim.x*imagedim.y*imagedim.z;
	layerhandle.setLayer(imagedimension.x, imagedimension.y, imagedimension.z, batchsize);
	layerhandle.setDescr(workspaceSize, withWorkspace);
}

void Network::initNetwork(bool pretrained) {
	layerhandle.initLayer(pretrained);
}
void Network::saveNetwork() {
	layerhandle.saveLayer();
}

void Network::train(bool pretrained,bool withworkspace,float learningrate,int iter)
{
	ImageHandler imagehandle;
	imagehandle.loadImages(1);
	int batchSize = imagehandle.images.size() / 2;
	dim3 imagedim;
	imagedim.y = imagehandle.images.at(0).InfoHeader.biWidth;
	imagedim.z = imagehandle.images.at(0).InfoHeader.biHeight;
	imagedim.x = imagehandle.images.at(0).InfoHeader.biSizeImage;
	imagedim.x /= imagedim.y;
	imagedim.x /= imagedim.z;

	set(0, batchSize, imagedim, withworkspace);
	initNetwork(pretrained);
	bool outputfwd = true; bool outputbwd = true; bool outputwupdate = false; bool outputwupdatesum = true;

	float * labelfloat = makeLabelfloat(imagehandle.images);
	float * imagesfloat = makeImageFloat(imagehandle.images);
	float * predict;

	layerhandle.copyLayertoDev();
	layerhandle.copytoLayertoDevBwD();
	/*Training*/
	for(int i=0;i<iter;i++){
		error->checkError(cudaSetDevice(workinggpu));
		for (int ID = 0; ID < imagehandle.images.size(); ID+=batchSize) {
			predict = ForwardPropagationwCopy(imagesfloat, ID, outputfwd);
			std::cout << "Predict " << predict[0] << " , ";
			printptr(&labelfloat[ID], batchSize); std::cout << std::endl;
			if (outputfwd &&outputbwd)std::cout << "________________________________________________________________________________________________________________" << std::endl;			
			BackwardPropagation(labelfloat, outputbwd);
			if (outputfwd || outputbwd)std::cout << "________________________________________________________________________________________________________________" << std::endl;
			if (outputfwd || outputbwd)std::cout << std::endl;
			UpdateWeigth(learningrate,outputwupdate);
		}
	}
	/*Controloutput*/
	if (outputwupdatesum) {
		for (int i = 0; i < layerhandle.convlayers.size(); i++) {
			layerhandle.convlayers[i].printHostCB(5);
			layerhandle.convlayers[i].printDevCB(5);
			std::cout << "________________________________________________" << std::endl;
		}
		std::cout << std::endl;
		for (int i = 0; i < layerhandle.fulls.size(); i++) {
			layerhandle.fulls[i].printHostNB(5);
			layerhandle.fulls[i].printDevNB(5);
			std::cout << "________________________________________________" << std::endl;
		}
		std::cout << std::endl;
	}
	layerhandle.copyLayertoHost();	
	saveNetwork();
	layerhandle.freeDev();
	error->checkError(cudaDeviceReset());

}

float* Network::ForwardPropagation(float*imagesfloat) {
	layerhandle.copyLayertoDev();
	layerhandle.copytoDevData(imagesfloat, 0);
	float* res = FwdPropa(false);
	return res;
}

void Network::UpdateWeigth(float learningrate,bool output) {
	for(unsigned int i=0;i<layerhandle.convlayers.size();i++){
		propa.UpdateWeightsConv(learningrate, &layerhandle.convlayers[i],output);
	}
	for (unsigned int i = 0; i<layerhandle.fulls.size(); i++) {
		propa.UpdateWeightsFull(learningrate, &layerhandle.fulls[i],output);
	}
}

float* Network::ForwardPropagationwCopy(float*imagesfloat, int ID, bool controloutput) {
	layerhandle.copytoDevData(imagesfloat, ID);
	float* res = FwdPropa(controloutput);
	return res;
}

float * Network::FwdPropa(bool controloutput) {
	float*predict = new float[baSize];
	propa.set(baSize, (int)workspaceSize, workinggpu);
	Layer*nxtLayer = layerhandle.firstLayer->nextLayer;
	ConvLayer *conv; MaxPoolLayer *pool;FullyConnectedLayer *full; Activation *act; SoftMax *sft; LocalResponseNormalization *lrn;
	do {
		switch (nxtLayer->getTypeId())
		{
		case Layer::Conv:
			if (controloutput)std::cout << "Conv, ";
			conv = (ConvLayer*)nxtLayer;
			propa.convForward(conv);
			break;
		case Layer::MaxPooling:
			if (controloutput)std::cout << "MaxPool, ";
			pool = (MaxPoolLayer*)nxtLayer;
			propa.poolForward(pool);
			break;
		case Layer::FullyConnectedLayer:
			if (controloutput)std::cout << "Fully, ";
			full =(FullyConnectedLayer*) nxtLayer;
			propa.fullyConnectedForward(full);
			break;
		case Layer::Activation:
			if (controloutput)std::cout << "Activ, ";
			act=(Activation*)nxtLayer;
			propa.activationForward(act);
			break;
		case Layer::SoftMax:
			if (controloutput)std::cout << "Sfw, ";
			sft = (SoftMax*)nxtLayer;
			propa.softmaxForward(sft);
			break;
		case Layer::LocalResponseNormalization:
			if (controloutput)std::cout << "Lrn, ";
			lrn = (LocalResponseNormalization*)nxtLayer;
			propa.lrnForward(lrn);
			break;
		case Layer::ParentLayer:
			error->throwError("ParentLayer not allowed to be used for Propagate");
			break;
		default:
			break;
		}
		nxtLayer = nxtLayer->nextLayer;
	} while (nxtLayer->getTypeId() != Layer::LblLayer);

	if (controloutput)	std::cout << std::endl << std::endl;
	if (controloutput) { layerhandle.printDev(5); std::cout << std::endl; }

	int max_digits = layerhandle.lastLayer->outchannel;
	float *result = new float[max_digits*baSize];
	error->checkError(cudaMemcpy(result, layerhandle.sfts[layerhandle.sfts.size()-1].ptrToOutData, baSize*max_digits * sizeof(float), cudaMemcpyDeviceToHost));
	for (int batch = 0; batch < baSize; batch++)
	{
		predict[batch] = 0;
		for (int i = 1; i < max_digits; i++)
		{
			if ((result[(int)predict[batch]]) < (result[i])) predict[batch] = (float)i;

		}
	}
	return predict;
}

void Network::BackwardPropagation(float*labelfloat, bool controloutput) {
	layerhandle.copytoDevDiff(labelfloat);
	BwdPropa(controloutput);
}

void Network::BwdPropa(bool controloutput) {
	propa.set(baSize, (int)workspaceSize, workinggpu);

	error->checkError(cudaDeviceSynchronize());
	Layer*preLayer = layerhandle.lastLayer->prevLayer;

	ConvLayer *conv; MaxPoolLayer *pool; FullyConnectedLayer *full; Activation *act; SoftMax *sft; LocalResponseNormalization *lrn;
	do {
		switch (preLayer->getTypeId())
		{
		case Layer::Conv:
			if(controloutput)	std::cout << "Conv, ";
			conv = (ConvLayer*)preLayer;
			propa.convBackward(conv);
			break;
		case Layer::MaxPooling:
			if (controloutput)std::cout << "Pool, ";
			pool = (MaxPoolLayer*)preLayer;
			propa.poolBackward(pool);
			break;
		case Layer::FullyConnectedLayer:
			if (controloutput)std::cout << "Fully, ";
			full = (FullyConnectedLayer*)preLayer;
			propa.fullyConnectedBackward(full);
			break;
		case Layer::Activation:
			if (controloutput)std::cout << "Activ, ";
			/*preLayer->ptrToGradData = preLayer->nextLayer->ptrToGradData;*/
			act = (Activation*)preLayer;
			propa.activationBackward(act);
			break;
		case Layer::SoftMax:
			if (controloutput)std::cout << "Sfw, ";
			preLayer->ptrToGradData = preLayer->nextLayer->ptrToGradData;
			/*sft = (SoftMax*)preLayer;
			propa.softmaxBackward(sft);*/
			break;
		case Layer::LocalResponseNormalization:
			if (controloutput)std::cout << "Lrn, ";
			preLayer->ptrToGradData = preLayer->nextLayer->ptrToGradData;
			/*lrn = (LocalResponseNormalization*)preLayer;
			propa.lrnBackward(lrn);*/
			break;
		case Layer::ParentLayer:
			error->throwError("ParentLayer not allowed to be used for Propagate");
			break;
		default:
			break;
		}
		preLayer = preLayer->prevLayer;
	} while (preLayer != nullptr);
	if (controloutput)std::cout << std::endl << std::endl;
	if (controloutput) { layerhandle.printGrad(5); std::cout << std::endl; }
	error->checkError(cudaDeviceSynchronize());
}

Network::~Network() {
	error->checkError(cudaDeviceReset());
}

float* Network::makeImageFloat(std::vector<Image> images) {
	float* fl = new float[imagesize*images.size()];
	for (int i = 0; i < images.size(); i++) {
		std::vector<float> akt = images[i].data;
		for (int j = 0; j < imagesize; j++) {
			fl[i*imagesize + j] = akt[j];
		}
	}
	return fl;
}

float* Network::makeLabelfloat(std::vector<Image> images) {
	float* fl = new float[1 * images.size()];
	for (int i = 0; i < images.size(); i++) {
		fl[i] = (float)images[i].label;
	}
	return fl;
}

void Network::printptr(float*ptr, int size) {
	for (int i = 0; i < size; i++) {
		std::cout << ptr[i] << " ";
	}
}

void Network::printDevptr(float*ptr, int size) {
	float* temp = new float[size];
	error->checkError(cudaMemcpy(&temp[0], ptr, size, cudaMemcpyDeviceToHost));
	for (int i = 0; i < size; i++) {
		std::cout << temp[i] << " ";
	}
	delete [] temp;
}
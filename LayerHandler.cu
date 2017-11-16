#include "LayerHandler.cuh"

void LayerHandler::freeDevFwD() {
	if (!devMallocFwD)return;
	//DEVICE MEM
	Layer*nxtLayer = firstLayer;
	do {
		nxtLayer->freeDevFwD();
		nxtLayer = nxtLayer->nextLayer;
	} while (nxtLayer != nullptr);

}
void LayerHandler::freeDevBwD() {
	if (!devMallocBwD)return;
	Layer*nxtLayer = firstLayer;
	do {
		nxtLayer->freeDevBwD();
		nxtLayer = nxtLayer->nextLayer;
	} while (nxtLayer != nullptr);
}
void LayerHandler::freeDev() {
	freeDevFwD();
	freeDevBwD();
	for (unsigned int i = 0; i < convlayers.size(); i++) {
		convlayers[i].freeDevBwDData();
		convlayers[i].freeDevFwDData();
	}
	for (unsigned int i = 0; i < fulls.size(); i++) {
		fulls[i].freeDevFwDData();
		fulls[i].freeDevBwDData();
	}
}
void LayerHandler::init() {
	convlayers.resize(2); poollayers.resize(2); fulls.resize(2); activs.resize(2);
	lrns.resize(1); sfts.resize(1);
	outcconv.insert_or_assign(0, 20);
	outcconv.insert_or_assign(1, 50);

	outcfull.insert_or_assign(0, 500);
	outcfull.insert_or_assign(1, 10);

	Layer*prevLayer = nullptr; Layer*nLayer = &convlayers[0];
	datalayer.initpre(prevLayer); prevLayer = &datalayer;
	convlayers[0].initpre(prevLayer); prevLayer = &convlayers[0];
	poollayers[0].initpre(prevLayer); prevLayer = &poollayers[0];
	convlayers[1].initpre(prevLayer); prevLayer = &convlayers[1];
	poollayers[1].initpre(prevLayer); prevLayer = &poollayers[1];

	fulls[0].initpre(prevLayer); prevLayer = &fulls[0];
	activs[0].initpre(prevLayer);prevLayer = &activs[0];
	lrns[0].initpre(prevLayer);prevLayer = &lrns[0];
	
	fulls[1].initpre(prevLayer); prevLayer = &fulls[1];
	activs[1].initpre(prevLayer);prevLayer = &activs[1];
	sfts[0].initpre(prevLayer); prevLayer = &sfts[0];
	lbllayer.initpre(prevLayer);
}
void LayerHandler::setLayer(int channels, int width, int height, int batchSize) {
	layerSet = true;
	datalayer.set(channels, width, height, batchSize);
	init();
	baSize = batchSize;
	int kernel = 5; int size = 2; int stride = 2;
	
	Layer*nLayer = firstLayer->nextLayer;
	ConvLayer*conv; FullyConnectedLayer*full; MaxPoolLayer *pool; int value = 0;
	int indexc = 0; int indexf = 0;
	do {
		switch (nLayer->getTypeId())
		{
		case Layer::Conv:
			conv = (ConvLayer*)nLayer;
			value = outcconv.at(indexc);
			indexc++;
			conv->set(value, kernel, batchSize);
			break;
		case Layer::FullyConnectedLayer:
			full = (FullyConnectedLayer*)nLayer;
			value = outcfull.at(indexf);
			indexf++;
			full->set(value, batchSize);
			break;
		case Layer::MaxPooling:
			pool = (MaxPoolLayer*)nLayer;
			pool->set(size, stride, batchSize);
			break;
		default:
			nLayer->set(batchSize);
			break;
		}
		nLayer = nLayer->nextLayer;
	} while (nLayer != nullptr);
	
	inputsize = batchSize*firstLayer->inchannel*firstLayer->inwidth*firstLayer->inheight;
	outputsize = batchSize*lastLayer->inchannel*lastLayer->inwidth*lastLayer->inheight;
}

void LayerHandler::setDescr(size_t&workspacesize, bool withWorkspace) {
	datalayer.setDescr();
	
	ConvLayer *conv;
	Layer*nxtLayer = firstLayer;
	do {
		switch (nxtLayer->getTypeId())
		{
		case Layer::Conv:
			conv = (ConvLayer*)nxtLayer;
			if(conv->prevLayer->getTypeId()==Layer::DataLayer)conv->setDescr(workspacesize, false, withWorkspace);
			else conv->setDescr(workspacesize, true, withWorkspace);
			nxtLayer = nxtLayer->nextLayer;
			break;
		default:
			nxtLayer->setDescr();
			nxtLayer = nxtLayer->nextLayer;
			break;
		}
	} while (nxtLayer != nullptr);
	error->checkError(cudaDeviceSynchronize());
}
void LayerHandler::initLayer(bool pretrained) {
	if (layerSet == false) {
		error->throwError("InitLayer without LayerSet before");
	}
	if (pretrained) {
		std::string pre = GetCurrentWorkingDir() + "\\Saved\\";
		for (unsigned int i = 0; i < convlayers.size(); i++) {
			std::string temp = "conv";
			temp += char(i) + '0';
			convlayers[i].initLayer((pre + temp).c_str());
		}
		for (unsigned int i = 0; i < fulls.size(); i++) {
			std::string temp = "full";
			temp += char(i) + '0';
			fulls[i].initLayer((pre + temp).c_str());
		}
	}
	else {
		for (unsigned int i = 0; i < convlayers.size(); i++) {
			convlayers[i].initRandom();
		}
		for (unsigned int i = 0; i < fulls.size(); i++) {
			fulls[i].initRandom();
		}
	}
}
void LayerHandler::saveLayer() {
	std::string pre = GetCurrentWorkingDir() + "\\Saved\\";
	for (int i = 0; i < convlayers.size(); i++) {
		std::string temp = "conv";
		temp += char(i)+'0';
		convlayers[i].saveLayer((pre + temp).c_str());
	}
	for (int i = 0; i < fulls.size(); i++) {
		std::string temp = "full";
		temp += char(i) + '0';
		fulls[i].saveLayer((pre + temp).c_str());
	}
}
void LayerHandler::copyLayertoDev()
{
	Layer*nLayer = firstLayer;
	ConvLayer*conv; FullyConnectedLayer *full;
	do {
		nLayer->mallocDevFwD();
		switch (nLayer->getTypeId())
		{
		case Layer::Conv:
			conv = (ConvLayer*)nLayer;
			conv->copyToDevFwD();
			break;
		case Layer::FullyConnectedLayer:
			full = (FullyConnectedLayer*)nLayer;
			full->copyToDevFwD();
			break;
		default:
			break;
		}	
		nLayer = nLayer->nextLayer;
	} while (nLayer != nullptr);
}
void LayerHandler::copytoLayertoDevBwD() {
	Layer*nLayer = firstLayer; FullyConnectedLayer *full;
	do {
		nLayer->mallocDevBwD();
		switch (nLayer->getTypeId())
		{
		case Layer::FullyConnectedLayer:
			full = (FullyConnectedLayer*)nLayer;
			full->copyToDevBwD();
			break;
		default:	
			break;
		}
		nLayer = nLayer->nextLayer;
	} while (nLayer != nullptr);
}
void LayerHandler::copyLayertoHost()
{
	for (unsigned int i = 0; i < convlayers.size(); i++) {
		convlayers[i].copyToHost();
	}
	for (unsigned int i = 0; i < fulls.size(); i++) {
		fulls[i].copyToHost();
	}
}
void LayerHandler::copytoDevData(float* imagesfloat, int imageID) {
	datalayer.copytoDevData(imagesfloat, imageID);
}
void LayerHandler::copytoHostData(float* &imagesfloat) {
	datalayer.copytoHostData(imagesfloat);
}
void LayerHandler::copytoDevDiff(float*labelsfloat) {
	lbllayer.copytoDevDiff(labelsfloat);
}
void LayerHandler::copytoHostLabelwComp(float* &labelsfloat) {
	lbllayer.copytoHostLabelwComp(labelsfloat);
}
void LayerHandler::printDev(int dimension) {
	std::cout.precision(5);
	std::cout.setf(std::ios::fixed, std::ios::floatfield);

	Layer*nxtLayer = firstLayer;
	do {
		nxtLayer->printDev(dimension);
		nxtLayer = nxtLayer->nextLayer;
		std::cout << std::endl;
	} while (nxtLayer->getTypeId() != Layer::LblLayer);
}
void LayerHandler::printGrad(int dimension) {
	std::cout.precision(5);
	std::cout.setf(std::ios::fixed, std::ios::floatfield);

	Layer*nxtLayer = lastLayer;
	do {
		nxtLayer->printGrad(dimension);
		nxtLayer = nxtLayer->prevLayer;
		std::cout << std::endl;
	} while (nxtLayer->prevLayer->getTypeId() != Layer::DataLayer);
}
std::string LayerHandler::GetCurrentWorkingDir(void) {
	char buff[FILENAME_MAX];
	_getcwd(buff, FILENAME_MAX);
	std::string current_working_dir(buff);
	return current_working_dir;
}
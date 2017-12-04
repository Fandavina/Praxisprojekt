#include "FullyConnected.h"

void FullyConnectedLayer::mallocDevFwD()
{
	Layer::mallocDevFwD();
	error->checkError(cudaMalloc(&ptrToDevNeuron, sizeof(float) * pneurons.size()));
	error->checkError(cudaMalloc(&ptrToDevBias, sizeof(float) * pbias.size()));
}
void FullyConnectedLayer::mallocDevBwD()
{
	Layer::mallocDevBwD();
	error->checkError(cudaMalloc(&ptrToGradDevNeuron, pneurons.size() * sizeof(float)));
	error->checkError(cudaMalloc(&ptrToGradDevBias,pbias.size() * sizeof(float)));
	error->checkError(cudaDeviceSynchronize());
}

void FullyConnectedLayer::freeDevFwDData() {
	if (ptrToDevNeuron != nullptr) {
		error->checkError(cudaFree(ptrToDevNeuron));
		ptrToDevNeuron = nullptr;
	}
	if (ptrToDevBias != nullptr) {
		error->checkError(cudaFree(ptrToDevBias));
		ptrToDevBias = nullptr;
	}
}
void FullyConnectedLayer::freeDevBwDData() {
	if (ptrToGradDevNeuron != nullptr) {
		error->checkError(cudaFree(ptrToGradDevNeuron));
		ptrToGradDevNeuron = nullptr;
	}
	if (ptrToGradDevBias != nullptr) {
		error->checkError(cudaFree(ptrToGradDevBias));
		ptrToGradDevBias = nullptr;
	}
}
void FullyConnectedLayer::setDescr() {
	Layer::setDescr();
}
FullyConnectedLayer::FullyConnectedLayer(int outputs_, int batchSize) {
	set(outputs_, batchSize);

}
void FullyConnectedLayer::set( int outputs_, int batchSize) {
	Layer::set(batchSize,outputs_,1,1);
	pneurons.resize(inputsize* outputsize);
	pbias.resize(outputs_);
}
void FullyConnectedLayer::initLayer(const char *fileprefix) {
	std::string ssf = "";
	std::string ssbf = "";
	ssf = fileprefix;
	ssf += ".bin";
	ssbf = fileprefix;
	ssbf += ".bias.bin";

	// Read weights file
	FILE *fp = fopen(ssf.c_str(), "rb");
	if (!fp)
	{
		error->throwError("Keine gespeicherte FullLayer unter " + ssf + "gefunden");
	}
	float *neurontemp = new float[pneurons.size()];
	fread(neurontemp, sizeof(float), pneurons.size(), fp);
	fclose(fp);
	for (unsigned int i = 0; i < pneurons.size(); i++) {
		pneurons[i] = neurontemp[i];
	}
	fclose(fp);

	// Read bias file
	fp = fopen(ssbf.c_str(), "rb");
	if (!fp)
	{
		error->throwError("Keine gespeicherte FullLayer unter " + ssbf + "gefunden");
	}
	float *biastemp = new float[pbias.size()];
	fread(biastemp, sizeof(float), pbias.size(), fp);
	fclose(fp);
	for (unsigned int i = 0; i < pbias.size(); i++) {
		pbias[i] = biastemp[i];
	}
	fclose(fp);
}
void FullyConnectedLayer::initRandom() {
	if (pneurons.size() == 0) {
		error->throwError("FullInit failed. Reason: pneuronssize =0");
	}
	if (pbias.size() == 0) {
		error->throwError("FullInit failed. Reason: pbiassize =0");
	}
	// Create random network
	randomGenerator(inchannel*outchannel, pneurons.size(), pneurons);
	randomGenerator(inchannel*outchannel, pbias.size(), pbias);

}
void FullyConnectedLayer::saveLayer(const char *fileprefix) {
	std::string ssf = "";
	std::string ssbf = "";
	ssf = fileprefix;
	ssf += ".bin";
	ssbf = fileprefix;
	ssbf += ".bias.bin";

	// Write weights file
	FILE *fp = fopen(ssf.c_str(), "wb");
	if (!fp)
	{
		printf("ERROR: Cannot open file %s\n", ssf.c_str());
		exit(2);
	}
	for each(float var in pneurons) {
		fwrite(&var, sizeof(float), 1, fp);
	}
	fclose(fp);

	// Write bias file
	fp = fopen(ssbf.c_str(), "wb");
	if (!fp)
	{
		printf("ERROR: Cannot open file %s\n", ssbf.c_str());
		exit(2);
	}
	for each(float var in pbias) {
		fwrite(&var, sizeof(float), 1, fp);
	}
	fclose(fp);
	fclose(fp);
}
void FullyConnectedLayer::copyToDevFwD() {
	error->checkError(cudaMemcpyAsync(ptrToDevNeuron, &pneurons[0], sizeof(float) * pneurons.size(), cudaMemcpyHostToDevice));
	error->checkError(cudaMemcpyAsync(ptrToDevBias, &pbias[0], sizeof(float) * pbias.size(), cudaMemcpyHostToDevice));
}
void FullyConnectedLayer::copyToDevBwD() {
	error->checkError(cudaMemcpyAsync(&ptrToGradDevNeuron[0], ptrToDevNeuron, sizeof(float) * pneurons.size(), cudaMemcpyDeviceToDevice));
	error->checkError(cudaMemcpyAsync(&ptrToGradDevBias[0], ptrToDevBias, sizeof(float) * pbias.size(), cudaMemcpyDeviceToDevice));

}
void FullyConnectedLayer::copyToHost() {
	error->checkError(cudaMemcpy(&pneurons[0], ptrToDevNeuron, sizeof(float) * pneurons.size(), cudaMemcpyDeviceToHost));
	error->checkError(cudaMemcpy(&pbias[0], ptrToDevBias, sizeof(float) * pbias.size(), cudaMemcpyDeviceToHost));
}
void FullyConnectedLayer::printDevNB(int dimension) {
	printptrDev("FullData Device", ptrToDevNeuron, dimension, pneurons.size());
	printptrDev("BiasData Device", ptrToDevBias, dimension, pbias.size());
}
void FullyConnectedLayer::printHostNB(int dimension) {
	printHost("FullData Host", pneurons, dimension, pneurons.size());
	printHost("BiasData Host", pbias, dimension, pbias.size());
}
void FullyConnectedLayer::printDev(int dimension) {
	Layer::printDev(dimension, "Fully ");
}
void FullyConnectedLayer::printGrad(int dimension) {
	Layer::printGrad(dimension, "Fully ");
}
int FullyConnectedLayer::getTypeId()
{
	return LayerID::FullyConnectedLayer;
}
void FullyConnectedLayer::printGradCB(int dimension) {
	printptrDev("FullGradNeuron Device", ptrToGradDevNeuron, dimension, pneurons.size());
	printptrDev("FullGradBias Device", ptrToGradDevBias, dimension, pbias.size());
}

#include "FullyConnected.h"

void FullyConnectedLayer::mallocDevFwD()
{
	Layer::mallocDevFwD();
	error->checkError(cudaMalloc(&ptrtoDevNeuron, sizeof(float) * pneurons.size()));
	error->checkError(cudaMalloc(&ptrtoDevBias, sizeof(float) * pbias.size()));
}
void FullyConnectedLayer::mallocDevBwD()
{
	Layer::mallocDevBwD();
	error->checkError(cudaMalloc(&ptrToGradDevNeuron, pneurons.size() * sizeof(float)));
	error->checkError(cudaMalloc(&ptrToGradDevBias,pbias.size() * sizeof(float)));
	error->checkError(cudaDeviceSynchronize());
}

void FullyConnectedLayer::freeDevFwDData() {
	if (ptrtoDevNeuron != nullptr) {
		error->checkError(cudaFree(ptrtoDevNeuron));
		ptrtoDevNeuron = nullptr;
	}
	if (ptrtoDevBias != nullptr) {
		error->checkError(cudaFree(ptrtoDevBias));
		ptrtoDevBias = nullptr;
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
	int inputs_ = prevLayer->outchannel;
	pneurons.resize(inputs_* outputs_);
	pbias.resize(outputs_);
	Layer::set(batchSize,outputs_,1,1);
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
	std::random_device rd;
	std::mt19937 gen;

	// Xavier weight filling
	float wfc1 = sqrt(3.0f / (inchannel * outchannel));
	std::uniform_real_distribution<> dActi(-wfc1, wfc1);

	for (int i = 0; i < pneurons.size(); i++)
		pneurons[i] = static_cast<float>(dActi(gen));
	for (int i = 0; i < pbias.size(); i++)
		pbias[i] = static_cast<float>(dActi(gen));

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
	error->checkError(cudaMemcpyAsync(ptrtoDevNeuron, &pneurons[0], sizeof(float) * pneurons.size(), cudaMemcpyHostToDevice));
	error->checkError(cudaMemcpyAsync(ptrtoDevBias, &pbias[0], sizeof(float) * pbias.size(), cudaMemcpyHostToDevice));
}
void FullyConnectedLayer::copyToDevBwD() {
	error->checkError(cudaMemcpyAsync(&ptrToGradDevNeuron[0], ptrtoDevNeuron, sizeof(float) * pneurons.size(), cudaMemcpyDeviceToDevice));
	error->checkError(cudaMemcpyAsync(&ptrToGradDevBias[0], ptrtoDevBias, sizeof(float) * pbias.size(), cudaMemcpyDeviceToDevice));

}
void FullyConnectedLayer::copyToHost() {
	error->checkError(cudaMemcpy(&pneurons[0], ptrtoDevNeuron, sizeof(float) * pneurons.size(), cudaMemcpyDeviceToHost));
	error->checkError(cudaMemcpy(&pbias[0], ptrtoDevBias, sizeof(float) * pbias.size(), cudaMemcpyDeviceToHost));
}
void FullyConnectedLayer::printDevNB(int dimension) {
	printptrDev("FullData Device", ptrtoDevNeuron, dimension, pneurons.size());
	printptrDev("BiasData Device", ptrtoDevBias, dimension, pbias.size());
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

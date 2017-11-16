#include "Network.h"

Network::Network() {
	error->checkError(cudaGetDeviceCount(&numgpu));
	int cudnnversion = (int)cudnnGetVersion();
	int cudartversion = (int)cudnnGetCudartVersion();
	if (cudnnversion < 7003)error->warning("Cudnn Version < 7003");
	if (cudartversion < 9000)error->warning("Cudart Version < 9000");
}
Network::Network(int workgpu, int batchsize, dim3 imagedim,bool withWorkspace) {
	error->checkError(cudaGetDeviceCount(&numgpu));
	int cudnnversion = (int)cudnnGetVersion();
	int cudartversion = (int)cudnnGetCudartVersion();
	if (cudnnversion < 7003)error->warning("Cudnn Version < 7003");
	if (cudartversion < 9000)error->warning("Cudart Version < 9000");
	set(workgpu, batchsize, imagedim, withWorkspace);
}
void Network::set(int workgpu, int batchsize, dim3 imagedim,bool withWorkspace) {
	if (workgpu >= numgpu) {
		std::string errorstr = "Es ist keine GPU NR." + ((char)workgpu - (char)0);
		errorstr += " vorhanden";
		error->checkError(errorstr);
	}
	/*cudaDeviceProp prop; //TODO Used Memory > GPU Memory
	cudaGetDeviceProperties(&prop, 0);*/
	baSize = batchsize;
	workinggpu = workgpu;
	error->checkError(cudaSetDevice(workinggpu));
	imagedimension = imagedim;
	imagesize = imagedim.x*imagedim.y*imagedim.z;
	layer.setLayer(imagedimension.x, imagedimension.y, imagedimension.z, batchsize);
	layer.setDescr(workspaceSize,withWorkspace);
}
void Network::initNetwork(bool pretrained) {
	layer.initLayer(pretrained);
}
void Network::saveNetwork() {
	layer.saveLayer();
}
void Network::train(bool pretrained)
{
	ImageHandler imagehandle;
	imagehandle.loadImages(1);
	int batchSize = 1;// imagehandle.images.size() / 2;
	dim3 imagedim;
	imagedim.y = imagehandle.images.at(0).InfoHeader.biWidth;
	imagedim.z = imagehandle.images.at(0).InfoHeader.biHeight;
	imagedim.x = imagehandle.images.at(0).InfoHeader.biSizeImage;
	imagedim.x /= imagedim.y;
	imagedim.x /= imagedim.z;

	set(0, batchSize, imagedim,false);
	initNetwork(pretrained);
	int ID = 0;
	//float * imagesfloat = makeImageFloat(imagehandle.images);
	//int res = ForwardPropagation(imagesfloat, ID,false);
	//int * labelfloat = makeLabelInt(imagehandle.images);
	//BackwardPropagation(labelfloat, ID,true);
	layer.freeDev();
}

int Network::ForwardPropagation(float*imagesfloat, int ID,bool controloutput) {
	layer.copyLayertoDev();
	layer.mallocDev();
	layer.copytoDevData(imagesfloat, ID);
	propa.set(baSize);
	int res = FwdPropa(controloutput);
	return res;
}
int Network::FwdPropa(bool controloutput) {
	int id = 0;
	error->checkError(cudaDeviceSynchronize());
	propa.convForward(layer.dataTensor, layer.ptrToDevData, workspaceSize, layer.conv1);
	propa.poolForward(layer.conv1.TensorDescr, layer.conv1.ptrToOutData, layer.pool1);
	propa.convForward(layer.pool1.TensorDescr, layer.pool1.ptrToOutData, workspaceSize, layer.conv2);
	propa.poolForward(layer.conv2.TensorDescr, layer.conv2.ptrToOutData, layer.pool2);

	dim3 inputdim; inputdim.x = layer.pool2.outchannel; inputdim.y = layer.pool2.outheight; inputdim.z = layer.pool2.outwidth;
	propa.fullyConnectedForward(inputdim, layer.pool2.ptrToOutData, layer.full1);

	propa.activationForward(layer.full1.TensorDescr, layer.full1.ptrToOutData, layer.activation1);
	propa.lrnForward(layer.activation1.TensorDescr, layer.activation1.ptrToOutData, layer.lrn1);

	inputdim.x = layer.lrn1.outchannel; inputdim.y = layer.lrn1.outheight; inputdim.z = layer.lrn1.outwidth;
	propa.fullyConnectedForward(inputdim, layer.lrn1.ptrToOutData,layer.full2);

	propa.softmaxForward(layer.full2.TensorDescr, layer.full2.ptrToOutData, layer.softmax1);
	error->checkError(cudaDeviceSynchronize());
	if(controloutput)layer.printDev(5);

	const int max_digits = 10;
	float result[max_digits];
	error->checkError(cudaMemcpy(result, layer.softmax1.ptrToOutData, max_digits * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 1; i < max_digits; i++)
	{
		if (result[id] < result[i]) id = i;
	}
	return id;

}
void Network::BackwardPropagation(int*labelfloat, int ID,bool controloutput) {
	layer.copyLayertoDev();
	layer.mallocDev();
	layer.copytoDevLabel(labelfloat, ID);
	int res = BwdPropa(controloutput);
}
int Network::BwdPropa(bool controloutput) {
	propa.set(baSize);
	dim3 inputdim;
	error->checkError(cudaDeviceSynchronize());
	propa.softmaxBackward(layer.ptrToDevLabel, layer.full2.TensorDescr, layer.softmax1);

	inputdim.x = layer.full2.tensorsizec; inputdim.y = layer.full2.tensorsizeh; inputdim.z = layer.full2.tensorsizew;
	propa.fullyConnectedBackward(inputdim, layer.softmax1.ptrToGradData, layer.full2);

	propa.lrnBackward(layer.full2.ptrToGradData,layer.activation1.TensorDescr,layer.activation1.ptrToOutData, layer.lrn1);
	
	propa.activationBackward(layer.lrn1.ptrToGradData, layer.full1.TensorDescr, layer.full1.ptrToOutData, layer.activation1);
	
	inputdim.x = layer.activation1.outchannel; inputdim.y = layer.activation1.outheight; inputdim.z = layer.activation1.outwidth;
	propa.fullyConnectedBackward(inputdim, layer.activation1.ptrToOutData, layer.full1);
	
	propa.poolBackward(layer.full1.ptrToGradData, layer.conv2.TensorDescr, layer.conv2.ptrToOutData, layer.pool2);
	
	propa.convBackward(layer.pool2.ptrToGradData, layer.pool1.TensorDescr, layer.pool1.ptrToOutData, workspaceSize, true,layer.conv2);
	
	propa.poolBackward(layer.conv2.ptrToGradData, layer.conv1.TensorDescr, layer.conv1.ptrToOutData, layer.pool1);

	propa.convBackward(layer.pool1.ptrToGradData, layer.dataTensor, layer.ptrToDevData, workspaceSize,false, layer.conv1);
	if(controloutput)layer.printGrad(5);
	error->checkError(cudaDeviceSynchronize());
	return 0;
}
Network::~Network() {
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
int* Network::makeLabelInt(std::vector<Image> images) {
	int* fl = new int[1 * images.size()];
	for (int i = 0; i < images.size(); i++) {
		fl[i] = images[i].label;
	}
	return fl;
}
void Network::printptr(float*ptr, int size) {
	for (int i = 0; i < size; i++) {
		std::cout << ptr[i] << " ";
	}
}
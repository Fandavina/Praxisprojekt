#include "Activation.h"


Activation::Activation()
{
	createDescr();
}

int Activation::getTypeId()
{
	return LayerID::Activation;
}
Activation::~Activation()
{
	destroyDescr();
}

void Activation::createDescr() {
	if (Descr == NULL) error->checkError(cudnnCreateActivationDescriptor(&Descr));
}

void Activation::destroyDescr() {
	if (Descr != NULL) { error->checkError(cudnnDestroyActivationDescriptor(Descr)); Descr = NULL; }
}

void Activation::setDescr() {
	Layer::setDescr();
	error->checkError(cudnnSetActivationDescriptor(Descr, activationmode, propagateNan, 0.0));
}

Activation::Activation(int batchsize, Layer * prevLayer) {
	set(batchsize);
	initpre(prevLayer);
}

void Activation::printDev(int dimension) {
	Layer::printDev(dimension,"Acti ");
}
void Activation::printGrad(int dimension) {
	Layer::printGrad(dimension, "Acti ");
}
#include "LocalResponseNormalization.h"



LocalResponseNormalization::LocalResponseNormalization()
{
	createDescr();
}

LocalResponseNormalization::~LocalResponseNormalization()
{
	destroyDescr();
}
void LocalResponseNormalization::createDescr() {
	Layer::createDescr();
	if (Descr == NULL) error->checkError(cudnnCreateLRNDescriptor(&Descr));
}

void LocalResponseNormalization::destroyDescr() {
	Layer::destroyDescr();
	if (Descr != NULL) { error->checkError(cudnnDestroyLRNDescriptor(Descr)); Descr = NULL; }
}

void LocalResponseNormalization::setDescr() {
	error->checkError(cudnnSetLRNDescriptor(Descr,
		lrnN,
		lrnAlpha,
		lrnBeta,
		lrnK));
	Layer::setDescr();
}
void LocalResponseNormalization::printDev(int dimension) {
	Layer::printDev(dimension, "LRN ");
}
void LocalResponseNormalization::printGrad(int dimension) {
	Layer::printGrad(dimension, "LRN ");
}
int LocalResponseNormalization::getTypeId() {
	return Layer::LocalResponseNormalization;
}
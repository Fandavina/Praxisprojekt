#include "SoftMax.h"

void SoftMax::printDev(int dimension) {
	Layer::printDev(dimension, "SfwMax ");
}
void SoftMax::printGrad(int dimension) {
	Layer::printGrad(dimension, "SfwMax ");
}
int SoftMax::getTypeId()
{
	return LayerID::SoftMax;
}
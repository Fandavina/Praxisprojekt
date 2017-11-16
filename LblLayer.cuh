#include "Layer.h"
class LblLayer :public Layer {
public:
	void copytoDevData(float * imagesfloat, int imageID);
	void copytoHostData(float *& imagesfloat);
	void printGrad(int dimension);
	int getTypeId();
	void copytoDevDiff(float * labelsfloat);
	void copytoHostLabelwComp(float *& labelsfloat);
};
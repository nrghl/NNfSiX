/**
 * Creates A Basic Neuron With 3 Inputs
 * Associated YT NNFS tutorial: https://www.youtube.com/watch?v=Wo5dMEP_BbI
 */

#include <u.h>
#include <libc.h>

void
main()
{
	double inputs[] = {1.0, 2.0, 3.0};
	double weights[] = {3.1, 2.1, 8.7};
	double bias = 3.0;

	double output;

	output = inputs[0] * weights[0] + inputs[1] * weights[1] +
						inputs[2] * weights[2] + bias;

	print("%f\n", output);

	exits(nil);
}

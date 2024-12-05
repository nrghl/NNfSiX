/**
 *  P.6 Softmax Activation
 * Associated YT NNFS tutorial: https://www.youtube.com/watch?v=omz_NdFgWyU
 */

#include <u.h>
#include <libc.h>

#define RAND_MAX 5
#define RAND_HIGH_RANGE (0.10)
#define RAND_MIN_RANGE (-0.10)
#define INIT_BIASES (0.0)

#define NET_BATCH_SIZE 300
#define NET_INPUT_LAYER_1_SIZE 2 /* Can be replaced with (sizeof(var)/sizeof(double)) */
#define NET_INPUT_LAYER_2_SIZE 3 /* Can be replaced with (sizeof(var)/sizeof(double)) */
#define NET_OUTPUT_LAYER_SIZE 3 /* Can be replaced with (sizeof(var)/sizeof(double)) */

/* Callback function template definition */
typedef void (*actiavtion_callback)(double * output);

typedef struct{
    double *weights;    /* Neural layer network weights */
    double *biases;     /* Neural layer network biases */
    double *output;     /* Output of the neural layer */
    int input_size;     /* Size of the input layer */
    int output_size;    /* Size of the output layer */
	actiavtion_callback callback; /* Pionter to the callbacb used for the activation function */
}layer_dense_t;

typedef struct{
    double *x; /* Holds the x y axis data. Data is formated x y x y x y */
    double *y; /* Holds the group the data belongs too. Two steps of x is a single step of y */
}spiral_data_t;

/**@brief Get the dot product of a neuron and add the bias.
 *
 * @param[in]   input   	Pointer to the first address of the inputs.
 * @param[in]   weights  	Pointer to the first address of the weights.
 * @param[in]   bias	 	Pointer to the value of the neurons bias.
 * @param[in]   input_size  Number of neurons in the input layer.
 * @param[in]   callback    Pionter to the activation callback function.
 * @retval[out] output		The dot product of the neuron.
 */
double
dot_product(double *input,double *weights,double *bias,int input_size,actiavtion_callback callback)
{
    double output = 0.0;
    for(int i = 0;i<input_size;i++){
        output += input[i]*weights[i];
    }
	if(callback != 0){
        callback(&output);
    }
    output += *bias;
    return output;
}

/**@brief Get the dot products of each neuron and add the bias and store it in an output array.
 *
 * @param[in]   	input   		Pointer to the first address of the inputs.
 * @param[in]   	weights  		Pointer to the first address of the weights.
 * @param[in]   	bias	 		Pointer to the first address of the neuron bases.
 * @param[in]   	input_size  	Number of neurons in the input layer.
 * @param[in/out]   outputs  		Pointer to the first address of the outputs array.
 * @param[in]   	output_size  	Number of neurons in the output layer.
 * @param[in]       callback        Pionter to the activation callback function.
 */
void
layer_output(double *input,double *weights,double *bias,int input_size,double *outputs,int output_size,actiavtion_callback callback)
{
    int offset = 0;
    for(int i = 0; i < output_size; i++){
        outputs[i] = dot_product(input,weights + offset,&bias[i],input_size,callback);
        offset+=input_size;
    }
}

double  /* Generate a random floating point number from min to max */
rand_range(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

/**@brief Setup a layer with random weights and bais as well as allocating memory for the storage buffers.
 *
 * @param[in]   	layer   		Pointer to an empty layer with no values.
 * @param[in]   	intput_size  	Size of the input layer.
 * @param[in]   	output_size	 	Size of the output layer.
 */
void
layer_init(layer_dense_t *layer,int intput_size,int output_size)
{
    layer->input_size = intput_size;
    layer->output_size = output_size;

    /* create data as a flat 1-D dataset */

    layer->weights = malloc(sizeof(double) * intput_size * output_size);
    if(layer->weights == nil){
        print("weights mem error\n");
        return;
    }
    layer->biases   = malloc(sizeof(double) * output_size);
	if(layer->biases == nil){
        print("biases mem error\n");
        return;
    }
    layer->output = malloc(sizeof(double) * output_size);
    if(layer->output == nil){
        print("output mem error\n");
        return;
    }
    for(int i = 0; i < (output_size); i++){
           layer->biases[i] = INIT_BIASES;
    }
    for(int i = 0; i < (intput_size*output_size); i++){
           layer->weights[i] = rand_range(RAND_MIN_RANGE,RAND_HIGH_RANGE);
    }
}

void  /* free the memory allocated by a layer */
deloc_layer(layer_dense_t *layer)
{
    if(layer->weights != nil){
        free(layer->weights);
    }
    if(layer->biases != nil){
        free(layer->biases);
    }
    if(layer->biases != nil){
        free(layer->output);
    }
}

/**@brief Does a forward pass in the network from one layer to the next.
 *
 * @param[in]   	previos_layer   		Pointer the previos layer struct.
 * @param[in]   	next_layer  			Pointer the next layer struct.
 */
void
forward(layer_dense_t *previos_layer,layer_dense_t *next_layer)
{
    layer_output((previos_layer->output),next_layer->weights,next_layer->biases,next_layer->input_size,(next_layer->output),next_layer->output_size,next_layer->callback);
}

double  /* sigmoid activation function */
activation_sigmoid(double x)
{
     double result;
     result = 1 / (1 + exp(-x));
     return result;
}

double  /* ReLU activation function */
activation_ReLU(double x)
{
    if(x < 0.0){
       x = 0.0;
    }
    return x;
}

/**@brief Callback to apply a activation function to the output of a node.
 *
 * @param[in]   output   	Pointer to the dot product output.
 */
void
actiavtion1(double *output)
{
    *output = activation_ReLU(*output);
    /* *output = sigmoid(*output); */
}

/**@brief Generate a randome range in a uniform distribution.
 * @Note Code was lifted from here https://stackoverflow.com/questions/11641629/generating-a-uniform-distribution-of-integers-in-c
 *
 * @param[in]   rangeLow   	Lowest value that in the range that can be genarated.
 * @param[in]   rangeHigh  	Highest value that in the range that can be genarated.
 * @retval[out] rng_scaled	Random mumber that has been generated.
 */
double
uniform_distribution(double rangeLow, double rangeHigh)
{
    double rng = rand()/(1.0 + RAND_MAX);
    double range = rangeHigh - rangeLow + 1;
    double rng_scaled = (rng * range) + rangeLow;

    return rng_scaled;
}

/**@brief Generate a randome range in a uniform distribution.
 * @Note Credit to shreeviknesh (#106) saved alot of time.
 *
 * @param[in]   points   	Number of points to generate per class.
 * @param[in]   classes  	Number of classes to generate.
 * @param[out]  data	    Structure holding the generated spiral data.
 */
void
spiral_data(int points,int classes,spiral_data_t *data)
{
    data->x = (double*)malloc(sizeof(double)*points*classes*2);
    if(data->x == nil){
        print("data mem error\n");
        return;
    }
    data->y = (double*)malloc(sizeof(double)*points*classes);
    if(data->y == nil){
        print("pionts mem error\n");
        return;
    }

    int ix = 0;
    int iy = 0;

    for(int class_number = 0; class_number < classes; class_number++) {
		double r = 0;
		double t = class_number * 4;

		while(r <= 1 && t <= (class_number + 1) * 4) {
			/* adding some randomness to t */
			double random_t = t + uniform_distribution(-1.0,1.0) * 0.2;

			/* converting from polar to cartesian coordinates */
			data->x[ix] = r * sin(random_t * 2.5);
			data->x[ix+1] = r * cos(random_t * 2.5);

			data->y[iy] = class_number;

			/* the below two statements achieve linspace-like functionality */
			r += 1.0f / (points - 1);
			t += 4.0f / (points - 1);

			/* increment index */
            iy++;
			ix+=2;
		}
	}
}

/**@brief Free the allocated memory for the spiral data.
 *
 * @param[in]  data	    Structure holding the generated spiral data.
 */
void
deloc_spiral(spiral_data_t *data)
{
    if(data->x != nil){
        free(data->x);
    }
	if(data->y != nil){
        free(data->y);
    }
}

/**@brief Gets the sum of the output layer and normalizes each output value.
 *
 * @note C can not do inline summation of arrays. This can only be done after a forward pass has been done.
 *
 * @param[in/out]   	output_layer   		Pointer the output layer struct.
 */
void
activation_softmax(layer_dense_t *output_layer)
{
    double sum = 0.0;
    double maxu = 0.0;
    maxu = output_layer->output[0];
    for(int i = 1; i < output_layer->output_size;i++){
        if(output_layer->output[i] > maxu){
            maxu = output_layer->output[i];
        }
    }
    for(int i = 0; i < output_layer->output_size;i++){
        output_layer->output[i] = exp(output_layer->output[i] - maxu);
        sum += output_layer->output[i];
    }
    for(int i = 0; i < output_layer->output_size;i++){
        output_layer->output[i] = output_layer->output[i] / sum;
    }
}

/**@brief Test function. Sums the output after activation_softmax has run on the output layer. Correct output is 1.0.
 *
 * @param[in]   	output_layer   		Pointer the output layer struct.
 */
double
sum_softmax_layer_output(layer_dense_t *output_layer)
{
    double sum = 0.0;
    for(int i = 0; i < output_layer->output_size;i++){
        sum += output_layer->output[i];
    }
    return sum;
}

int
main()
{
    srand(0);  /* seed the random values */
    spiral_data_t X_data;

    layer_dense_t X;
    layer_dense_t dense1;
    layer_dense_t dense2;

    spiral_data(100,3,&X_data);
    if(X_data.x == nil){
        print("data nil\n");
        return 0;
    }

    X.callback = 0;

    dense1 .callback = actiavtion1;
    dense2.callback = 0;

    layer_init(&dense1 ,NET_INPUT_LAYER_1_SIZE,NET_INPUT_LAYER_2_SIZE);
    layer_init(&dense2,NET_INPUT_LAYER_2_SIZE,NET_OUTPUT_LAYER_SIZE);

    for(int i = 0; i < NET_BATCH_SIZE;i++){
        X.output = &X_data.x[i*2];
        forward(&X,&dense1);
        print("\nbatch: %d layer1_output: ",i);
        for(int j = 0; j < dense1 .output_size; j++){
            print("%f ",dense1 .output[j]);
        }
        forward(&dense1,&dense2);
        print("\nbatch: %d layer2_output: ",i);
        for(int j = 0; j < dense2.output_size; j++){
            print("%f ",dense2.output[j]);
        }
        activation_softmax(&dense2);
        print("\nbatch: %d layer2_softmax: ",i);
        for(int j = 0; j < dense2.output_size; j++){
            print("%f ",dense2.output[j]);
        }
        /* print("\nbatch: %d layer2_normalize_sum: %f\n",i,sum_softmax_layer_output(&dense2)); */
		print("\n--");
    }

    deloc_layer(&dense1);
    deloc_layer(&dense2);
    deloc_spiral(&X_data);

    exits(nil);
}

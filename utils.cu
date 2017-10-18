#include <curand.h>
#include <curand_kernel.h>

__device__ void setSeed(curandState_t *state){
	curand_init((unsigned long long)clock(), 0, 0, state);
}

__device__ double randomDouble(curandState_t *state, double max){
    return max * curand_uniform_double(state);
}

__device__ int randomInt(curandState_t *state, int max){
    return curand(state) % max;
}

__device__ double randomBetween(curandState_t *state, double min, double max){
    return min + (max - min) * curand_uniform_double(state);
}

__device__ double fitness(double x, double y){
    return 100.00 * sqrt(fabs(y - 0.01*x*x)) + 0.01*fabs(x+10);
}

__device__ int checkBoundaries(Cromosome *individual){
	return minX <= individual->x && individual->x <= maxX
	 	&& minY <= individual->y && individual->y <= maxY;
}

__device__ void copy(Cromosome *to, Cromosome *from, int adnSize){
	to->x = from->x;
	to->y = from->y;
	to->fitness = from->fitness;
	for(int i=0; i<adnSize;i++){
		to->adn[i] = from->adn[i];
	}
}

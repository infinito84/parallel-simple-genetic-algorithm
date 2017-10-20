#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>
#include "structures.c"
#include "constants.c"
#include "utils.cu"
#include "cromosome.cu"
#include "crossover.cu"

__global__ void initialize(Population *populations, int N, int N_COUPLES, int ELITISM, int adnSize, int nDecimals, int yBits){
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	curandState_t state;
	setSeed(&state, id);
	populations[id].race = id;
	populations[id].generation = 0;
	populations[id].individuals = new Cromosome[N];
	populations[id].couples = new Couple[N_COUPLES];
	for(int i=0;i<N;i++){
		populations[id].individuals[i] = randomCromosome(&state, adnSize, nDecimals, yBits);
	}
	for(int i=0;i<N_COUPLES;i++){
		populations[id].couples[i].parent1 = randomCromosome(&state, adnSize, nDecimals, yBits);
		populations[id].couples[i].parent2 = randomCromosome(&state, adnSize, nDecimals, yBits);
	}
	populations[id].optimal.race = id;
	populations[id].optimal.individual.fitness = DBL_MAX;
}

__global__ void calcBest(Population *populations, int adnSize){
	Optimal optimal;
	optimal.individual.fitness = DBL_MAX;
	for(int i=0; i<THREADS; i++){
		if(populations[i].optimal.individual.fitness < optimal.individual.fitness){
			optimal = populations[i].optimal;
		}
	}
	printf("\x1B[32mGanador: Raza: %d, Generación: %d\n", optimal.race, optimal.generation);
	showCromosome(&optimal.individual, adnSize);
}

__global__ void createPopulation(Population *populations, int N, int N_COUPLES, int ELITISM, int adnSize, int nDecimals, int yBits){
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	curandState_t state;
	setSeed(&state, id);
	Cromosome tempCromosome;
	Cromosome *population = populations[id].individuals;
	Couple *parents = populations[id].couples;

	populations[id].generation++;
    double min = DBL_MAX, max = -DBL_MAX, total = 0;
    for(int i=0;i<N;i++){
		// Si no está en el rango, creamos uno nuevo
		if(!checkBoundaries(&population[i])){
			generateAdn(&state, &population[i], adnSize, nDecimals, yBits);
		}
		//calculamos el fitness
		population[i].fitness = fitness(population[i].x, population[i].y);
		total += population[i].fitness;
        if(population[i].fitness < min){
            min = population[i].fitness;
            tempCromosome = population[i];
        }
        if(population[i].fitness > max){
            max = population[i].fitness;
        }
    }
	if(min < populations[id].optimal.individual.fitness){
		populations[id].optimal.generation = populations[id].generation;
		populations[id].optimal.individual = tempCromosome;
		printf("\x1B[0mPopulation #%d, Generation #%d, min: %f, avg: %f, good: %f\n",
			id, populations[id].optimal.generation, min, total/N, populations[id].optimal.individual.fitness);
	}
    // Se realiza cálculo de la ruleta (minimización)
    float totalRoulette = 0;
    for(int i=0;i<N;i++){
		population[i].before = totalRoulette;
		population[i].fitness = max - population[i].fitness + min;
		totalRoulette += population[i].fitness;
        population[i].roulette = totalRoulette;
		population[i].before = totalRoulette - population[i].before;
    }
    // Se seleccionan N_COUPLES
    for(int i=0; i<N_COUPLES; i++){
        double n1 = randomDouble(&state, totalRoulette);
        double n2 = randomDouble(&state, totalRoulette);
        for(int j=0;j<N;j++){
            if(population[j].roulette >= n1 && n1 != -1){
                copy(&parents[i].parent1, &population[j], adnSize);
                n1 = -1;
            }
            if(population[j].roulette >= n2 && n2 != -1){
                copy(&parents[i].parent2, &population[j], adnSize);
                n2 = -1;
            }
        }
    }
    // Se cruzan los padres los dos que están seguidos (se crea nueva generación)
    int child = ELITISM;
	int bitSplitter = randomInt(&state, adnSize);
    for(int i=0;i<N_COUPLES;i++){
        crossover(&state, bitSplitter, &parents[i], &population[child++], &population[child++], adnSize, nDecimals, yBits);
    }
}

int main(){
	N = N / THREADS;
	N_COUPLES = (N - ELITISM) / 2;
    calcSizes();
	cudaError_t err = cudaSuccess;
	Population *populations;
	cudaMalloc((void **)&populations, THREADS * sizeof(Population));
	int threadsPerBlock = THREADS/BLOCKS;
	initialize<<<BLOCKS, threadsPerBlock>>>(populations, N, N_COUPLES, ELITISM, adnSize, nDecimals, yBits);
	for(int i=0; i<GENERATIONS; i++){
		createPopulation<<<BLOCKS, threadsPerBlock>>>(populations, N, N_COUPLES, ELITISM, adnSize, nDecimals, yBits);
		err = cudaGetLastError();
	    if (err != cudaSuccess){
	        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }
		cudaDeviceSynchronize();
	}
	calcBest<<<1, 1>>>(populations, adnSize);
	cudaDeviceSynchronize();
}

// Compile: /usr/local/cuda-9.0/bin/nvcc main-cuda.cu -o bin/main-cuda -lm
// Execute: time ./bin/main-cuda
/* threads   tiempo
1	188.504
2	188.504
4	188.504
8	188.504
50	22.752
100	10.530
200 5.975
500 3.975
1000 2.591
*/

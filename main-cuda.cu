#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>
#include "structures.c"
#include "constants.c"
#include "utils.cu"
#include "cromosome.cu"
#include "crossover.cu"

__global__ void createPopulation(Optimal best, int N, int N_COUPLES, int ELITISM, int adnSize, int nDecimals, int yBits){
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id >= THREADS) return;
	curandState_t state;
	Cromosome tempCromosome;
	Optimal optimal;
	int generation = 0;
	setSeed(&state);
	// Se crea población inicial
	Cromosome *population = new Cromosome[N];
	Couple *parents = new Couple[N_COUPLES];
	for(int i=0;i<N;i++){
		population[i] = randomCromosome(&state, adnSize, nDecimals, yBits);
	}
	optimal.race = id;
	optimal.individual.fitness = DBL_MAX;
	while(generation < GENERATIONS){
		generation++;
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
		if(min < optimal.individual.fitness){
			optimal.generation = generation;
			optimal.individual = tempCromosome;
			printf("\x1B[32mPopulation #%d, Generation #%d, min: %f, avg: %f, good(%d): %f\n",
				id, generation, min, total/N, optimal.generation, optimal.individual.fitness);
		}
		else{
			printf("\x1B[0mPopulation #%d, Generation #%d, min: %f, avg: %f, good(%d): %f\n",
				id, generation, min, total/N, optimal.generation, optimal.individual.fitness);
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
        int i=0;
        while(i<N_COUPLES){
            double n1 = randomDouble(&state, totalRoulette);
            double n2 = randomDouble(&state, totalRoulette);
            for(int j=0;j<N;j++){
                if(population[j].roulette >= n1 && n1 != -1){
                    parents[i].parent1 = population[j];
                    n1 = -1;
                }
                if(population[j].roulette >= n2 && n2 != -1){
                    parents[i].parent2 = population[j];
                    n2 = -1;
                }
            }
            i++;
        }

        // Se cruzan los padres los dos que están seguidos (se crea nueva generación)
        int child = ELITISM;
        for(int i=0;i<N_COUPLES;i++){
			int bitSplitter = randomInt(&state, adnSize);
            crossover(&state, bitSplitter, &parents[i], &population[child++], &population[child++], adnSize, nDecimals, yBits);
        }
	}
}

int main(){
    Optimal best;
	best.individual.fitness = DBL_MAX;
	N = N / THREADS;
	N_COUPLES = (N - ELITISM) / 2;

    calcSizes();
	cudaError_t err = cudaSuccess;
	int blocks = 6;
	int threadsPerBlock = THREADS/blocks;
	createPopulation<<<blocks, threadsPerBlock>>>(best, N, N_COUPLES, ELITISM, adnSize, nDecimals, yBits);
	err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	cudaDeviceSynchronize();
	printf("\x1B[32mGanador: Raza: %d, Generación: %d\n", best.race, best.generation);
    //showCromosome(&best.individual, adnSize);
}

// Compile: /usr/local/cuda-9.0/bin/nvcc main-cuda.cu -o bin/main-cuda -lm
// Execute: time ./bin/main-cuda
/* threads   tiempo
1	188.429
2	58.058
3	40.571
4	30.894
5	26.768
6	23.155
7	23.493
8	24.828
16	18.900
32	13.670
50	10.631
100	9.659
*/

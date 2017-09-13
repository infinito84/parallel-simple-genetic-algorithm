#include <stdio.h>
#include <omp.h>
#include <float.h>
#include "structures.c"
#include "constants.c"
#include "utils.c"
#include "cromosome.c"
#include "crossover.c"

int main(){
    setSeed();
    calcSizes();
	Optimal optimal = {randomCromosome(), 0, 0};
	optimal.individual.fitness = DBL_MAX;
	N = N / THREADS;
	N_COUPLES = (N - ELITISM) / 2;
	omp_set_num_threads(THREADS);
	#pragma omp parallel
	{
		Cromosome tempCromosome;
		int generation = 0;
		int id = omp_get_thread_num();
		// Se crea población inicial
		Cromosome population[N];
		Couple parents[N_COUPLES];
	    for(int i=0;i<N;i++){
	        population[i] = randomCromosome();
	    }

	    while(generation < GENERATIONS){
	        generation++;
	        double min = DBL_MAX, max = -DBL_MAX, total = 0, total2 = 0;
	        for(int i=0;i<N;i++){
				// Si no está en el rango, lo castigamos con un fitness alto
				total2 += population[i].fitness;
				if(!checkBoundaries(&population[i])){
					population[i].fitness = 300;
				}
	            total += population[i].fitness;
	            if(population[i].fitness < min){
	                min = population[i].fitness;
	                tempCromosome = population[i];
	            }
	        }
			if(min < optimal.individual.fitness){
	            optimal.race = id;
	            optimal.generation = generation;
	            optimal.individual = tempCromosome;
				printf("\x1B[32mRaza #%d, Generation #%d, min: %f, avg: %f, global(%d#%d): %f\n",
					id, generation, min, total2/N, optimal.race, optimal.generation, optimal.individual.fitness);
	        }
			else{
	        	printf("\x1B[0mRaza #%d, Generation #%d, min: %f, avg: %f, global(%d#%d): %f\n",
					id, generation, min, total2/N, optimal.race, optimal.generation, optimal.individual.fitness);
			}

			// Ordenamos para dejar los mejores al principio
			qsort (population, N, sizeof(Cromosome), compare);

	        // Se realiza cálculo de la ruleta (minimización)
	        float totalRoulette = 0;
	        for(int i=0;i<N;i++){
				totalRoulette += (max - population[i].fitness + min) / max;
	            population[i].roulette = totalRoulette;
	        }

	        // Se seleccionan N_COUPLES
	        int i=0;
	        while(i<N_COUPLES){
	            double n1 = randomDouble(totalRoulette);
	            double n2 = randomDouble(totalRoulette);
	            if(n1 != n2){
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
	        }

	        // Se cruzan los padres los dos que están seguidos (se crea nueva generación)
	        int child = 0;
			int bitSplitter = randomInt(adnSize);
	        for(int i=0;i<N_COUPLES;i++){
	            crossover(bitSplitter, &parents[i], &population[child++], &population[child++]);
	        }

			// Aplicamos mutación
			int toMutate = N * MUTATION;
			for(int i=0;i<toMutate;i++){
				int who = randomInt(N - ELITISM);
				mutate(&population[who + ELITISM]);
			}
	    }
	}

    printf("\x1B[32mGanador: Raza: %d, Generación: %d\n", optimal.race, optimal.generation);
    showCromosome(&optimal.individual);
}

// Compile: gcc main-openmp.c -o bin/main-openmp -lm -fopenmp
// Execute: time ./bin/main-openmp
// 4 threads: 32.204s
// 3 threads: 40.934s
// 2 threads: 60.535s
// 1 threads: 189.882s

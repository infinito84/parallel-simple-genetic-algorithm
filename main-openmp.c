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
	        double min = DBL_MAX, max = -DBL_MAX, total = 0;
	        for(int i=0;i<N;i++){
				// Si no está en el rango, creamos uno nuevo
				if(!checkBoundaries(&population[i])){
					generateAdn(&population[i]);
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
				optimal.race = id;
				optimal.generation = generation;
				optimal.individual = tempCromosome;
				printf("\x1B[32mPopulation #%d, Generation #%d, min: %f, avg: %f, global(%d#%d): %f\n",
					id, generation, min, total/N, optimal.race, optimal.generation, optimal.individual.fitness);
			}
			else{
				printf("\x1B[0mPopulation #%d, Generation #%d, min: %f, avg: %f, global(%d#%d): %f\n",
					id, generation, min, total/N, optimal.race, optimal.generation, optimal.individual.fitness);
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
	            double n1 = randomDouble(totalRoulette);
	            double n2 = randomDouble(totalRoulette);
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
				int bitSplitter = randomInt(adnSize);
	            crossover(bitSplitter, &parents[i], &population[child++], &population[child++]);
	        }
	    }
	}

    printf("\x1B[32mGanador: Raza: %d, Generación: %d\n", optimal.race, optimal.generation);
    showCromosome(&optimal.individual);
}

// Compile: gcc main-openmp.c -o bin/main-openmp -lm -fopenmp
// Execute: time ./bin/main-openmp
/* threads  tiempo
1	189.882
2	60.535
3	40.934
4	32.204
5	26.439
6	23.135
7	23.026
8	24.555
16	19.579
32	13.769
50	10.641  #hasta aquí bien
100	8.285
*/

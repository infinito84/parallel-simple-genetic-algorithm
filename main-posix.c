#include <stdio.h>
#include <omp.h>
#include <float.h>
#include <pthread.h>
#include "structures.c"
#include "constants.c"
#include "utils.c"
#include "cromosome.c"
#include "crossover.c"

Optimal optimal;

void *createRace(void *arg){
	Cromosome tempCromosome;
	int generation = 0;
	int id = *(int*) arg;
	// Se crea población inicial
	Cromosome population[N];
	Couple parents[N_COUPLES];
	for(int i=0;i<N;i++){
		population[i] = randomCromosome();
	}
	for(int i=0;i<N_COUPLES;i++){
		parents[i].parent1 = randomCromosome();
		parents[i].parent2 = randomCromosome();
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
					copy(&parents[i].parent1, &population[j], adnSize);
					n1 = -1;
				}
				if(population[j].roulette >= n2 && n2 != -1){
					copy(&parents[i].parent1, &population[j], adnSize);
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

int main(){
    int id[THREADS];
    pthread_t threads[THREADS];
    setSeed();
    calcSizes();

	optimal.individual = randomCromosome();
	optimal.individual.fitness = DBL_MAX;
	N = N / THREADS;
	N_COUPLES = (N - ELITISM) / 2;

	for(int i=0; i <THREADS; i++){
		id[i] = i;
		pthread_create(&threads[i], NULL, createRace, &id[i]);
	}

	for(int i=0; i <THREADS; i++){
		pthread_join(threads[i], NULL);
	}

	printf("\x1B[32mGanador: Raza: %d, Generación: %d\n", optimal.race, optimal.generation);
    showCromosome(&optimal.individual);
}

// Compile: gcc main-posix.c -o bin/main-posix -lm -lpthread
// Execute: time ./bin/main-posix
/* threads   tiempo
1	192.611
2	55.789
4	29.466
8	19.559
50	8.937
100	8.155
200 8.117
500 8.265
1000 8.621
*/

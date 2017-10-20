__device__ void binary2x(curandState_t *state, Cromosome *child, int adnSize, int nDecimals, int yBits){
    double xCount = 0;
    double x = 0;
    for(int i=adnSize - yBits - 1; i>=0; i--){
        if(child->adn[i]){
            x += powf(2, xCount);
        }
        xCount++;
    }
    child->x = x / nDecimals + minX;
}

__device__ void binary2y(curandState_t *state, Cromosome *child, int adnSize, int nDecimals, int yBits){
    int yCount = 0;
    double y = 0;
    for(int i=adnSize - 1; i>=0; i--){
        if(child->adn[i] && yCount < yBits){
            y += powf(2, yCount);
        }
        yCount++;
    }
    child->y = y / nDecimals + minY;
}

__device__ void mutate(curandState_t *state, Cromosome *individual, int adnSize){
	int bitMutation = randomInt(state, adnSize);
	individual->adn[bitMutation] = !individual->adn[bitMutation];
}

__device__ void crossover(curandState_t *state, int bitSplitter, Couple *couple, Cromosome *child1, Cromosome *child2, int adnSize, int nDecimals, int yBits){
	for(int i=0; i<adnSize; i++){
        if(i < bitSplitter){
            child1->adn[i] = couple->parent1.adn[i];
            child2->adn[i] = couple->parent2.adn[i];
        }
        else{
            child1->adn[i] = couple->parent2.adn[i];
            child2->adn[i] = couple->parent1.adn[i];
        }
    }
	if(randomDouble(state, 1) < MUTATION){
		mutate(state, child1, adnSize);
	}
	if(randomDouble(state, 1) < MUTATION){
		mutate(state, child2, adnSize);
	}
    binary2x(state, child1, adnSize, nDecimals, yBits);
    binary2y(state, child1, adnSize, nDecimals, yBits);
    binary2x(state, child2, adnSize, nDecimals, yBits);
    binary2y(state, child2, adnSize, nDecimals, yBits);
}

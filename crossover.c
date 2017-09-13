void binary2x(Cromosome *child){
    double xCount = 0;
    double x = 0;
    for(int i=adnSize - yBits - 1; i>=0; i--){
        if(child->adn[i]){
            x += pow(2, xCount);
        }
        xCount++;
    }
    child->x = x / nDecimals + minX;
}

void binary2y(Cromosome *child){
    int yCount = 0;
    double y = 0;
    for(int i=adnSize - 1; i>=0; i--){
        if(child->adn[i] && yCount < yBits){
            y += pow(2, yCount);
        }
        yCount++;
    }
    child->y = y / nDecimals + minY;
}

void crossover(int bitSplitter, Couple *couple, Cromosome *child1, Cromosome *child2){
    child1->adn = (int *)malloc(adnSize * sizeof(int));
    child2->adn = (int *)malloc(adnSize * sizeof(int));
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
    binary2x(child1);
    binary2y(child1);
    binary2x(child2);
    binary2y(child2);
    child1->fitness = fitness(child1->x, child1->y);
    child2->fitness = fitness(child2->x, child2->y);
}

void mutate(Cromosome *individual){
	int bitMutation = randomInt(adnSize);
	individual->adn[bitMutation] = !individual->adn[bitMutation];
	binary2x(individual);
    binary2y(individual);
    individual->fitness = fitness(individual->x, individual->y);
}

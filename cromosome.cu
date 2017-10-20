int adnSize, xSize, ySize, xBits = 0, yBits = 0;
double nDecimals;

void calcSizes(){
    nDecimals = pow(10, DECIMALS);
    xSize = (maxX - minX) * nDecimals;
    ySize = (maxY - minY) * nDecimals;
    double bits = 1, temp;
    while(!xBits || !yBits){
        temp = pow(2,bits);
        if(xSize < temp && !xBits){
            xBits = bits;
        }
        if(ySize < temp && !yBits){
            yBits = bits;
        }
        bits++;
    }
    adnSize = xBits + yBits;
    printf("Bits: adn(%d), x(%d), y(%d)\n", adnSize, xBits, yBits);
}

__device__ void generateAdn(curandState_t *state, Cromosome *temp, int adnSize, int nDecimals, int yBits){
	temp->x = randomBetween(state, minX, maxX);
    temp->y = randomBetween(state, minY, maxY);
	int x = (int)(nDecimals * (temp->x - minX));
	int y = (int)(nDecimals * (temp->y - minX));
	int yCount = 0;
    for(int i = adnSize - 1; i >= 0; i--){
        if(yCount < yBits){
            temp->adn[i] = (int)y % 2;
            y = y / 2;
            yCount++;
        }
        else{
            temp->adn[i] = (int)x % 2;
            x = x / 2;
        }
    }
}

__device__ __host__ void showCromosome(Cromosome *temp, int adnSize){
    printf("Cromosome: adn(");
    for(int i=0;i<adnSize;i++) printf("%d", temp->adn[i]);
    printf("), x(%.2f), y(%.2f), fitness(%f), ruleta(%f)\n", temp->x, temp->y, temp->fitness, temp->before);
}

__device__ Cromosome randomCromosome(curandState_t *state, int adnSize, int nDecimals, int yBits){
    Cromosome temp;
    temp.adn = new int[adnSize];
	generateAdn(state, &temp, adnSize, nDecimals, yBits);
	//showCromosome(&temp, adnSize);
    return temp;
}

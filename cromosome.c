int minX = -15, maxX = 0;
int minY = -3, maxY = 3;
int decimals = 2;
int adnSize, xSize, ySize, xBits = 0, yBits = 0;
double nDecimals;

void calcSizes(){
    nDecimals = pow(10,decimals);
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

struct Cromosome randomCromosome(){
    struct Cromosome temp;
    double x = randomInt(xSize + 1);
    double y = randomInt(ySize + 1);
    temp.x = (x / nDecimals) + minX;
    temp.y = (y / nDecimals) + minY;
    temp.adn = (int *)malloc(adnSize * sizeof(int));
    temp.fitness = fitness(temp.x, temp.y);
    int yCount = 0;
    for(int i = adnSize - 1; i >= 0; i--){
        if(yCount <= yBits){
            temp.adn[i] = (int)y % 2;
            y = y / 2;
            yCount++;
        }
        else{
            temp.adn[i] = (int)x % 2;
            x = x / 2;
        }
    }
    return temp;
}

void showCromosome(struct Cromosome *temp){
    printf("Cromosome: adn(");
    for(int i=0;i<adnSize;i++) printf("%d", temp->adn[i]);
    printf("), x(%.2f), y(%.2f)\n", temp->x, temp->y);
}

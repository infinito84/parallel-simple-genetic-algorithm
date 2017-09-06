const N = 2000;
const N_COUPLES = N / 2;
const GENERATIONS = 100;
const MUTATION = 10; //among 100

var population = [];
var parents;
var generation = 0;
var minGlobal = Number.MAX_VALUE;
var minGeneration = 0;
var minCromosome;

var minX = -15, maxX = 0;
var minY = -3, maxY = 3;
var decimals = 2;
var xBits = 0, yBits = 0;

var nDecimals = Math.pow(10,decimals);
var xSize = (maxX - minX) * nDecimals;
var ySize = (maxY - minY) * nDecimals;
var bits = 1, temp;
while(!xBits || !yBits){
    temp = Math.pow(2,bits);
    if(xSize < temp && !xBits){
        xBits = bits;
    }
    if(ySize < temp && !yBits){
        yBits = bits;
    }
    bits++;
}
var adnSize = xBits + yBits;
console.log("Bits: adn(%d), x(%d), y(%d)\n", adnSize, xBits, yBits);

var fitness = function(x, y){
    return 100.00 * Math.sqrt(Math.abs(y - 0.01*x*x)) + 0.01*Math.abs(x+10);
}

var main = function(){
    // Se crea población inicial
    for(var i=0;i<N;i++){
        population.push(randomCromosome());
    }
    while(generation < GENERATIONS){
        generation++;
        var min = Number.MAX_VALUE, max = Number.MIN_VALUE, total = 0;
        var tempCromosome;
        population.forEach(function(individual){
            total += individual.fitness;
            if(individual.fitness < min){
                min = individual.fitness;
                tempCromosome = individual;
            }
            if(individual.fitness > max){
                max = individual.fitness;
            }
        })
        if(min < minGlobal){
            minGlobal = min;
            minGeneration = generation;
            minCromosome = tempCromosome;
        }
        console.log('Generation #'+generation+', min: '+min+', avg: '+(total/N)+', global('+minGeneration+'): '+minGlobal);

        // Se realiza cálculo de la ruleta (minimización)
        var totalRoulette = 0;
        population.forEach(function(individual){
            totalRoulette = max - min - individual.fitness + totalRoulette;
            individual.roulette = totalRoulette;
        });

        // Se seleccionan N_COUPLES
        parents = [];
        for(var i = 0; i < N_COUPLES; i++){
            var n1 = Math.random()*(totalRoulette);
            var n2 = Math.random()*(totalRoulette);
            var couple = {};
            population.forEach(function(individual){
                if(individual.roulette >= n1 && !couple.parent1){
                    couple.parent1 = individual;
                }
                if(individual.roulette >= n2 && !couple.parent2){
                    couple.parent2 = individual;
                }
            });
            if(!couple.parent2){
                console.log(couple, n1, n2, totalRoulette);
            }
            parents.push(couple);
        }

        // Se cruzan los padres los dos que están seguidos (se crea nueva generación)
        population = [];
        parents.forEach(crossover);
    }

    console.log('Ganador: ', minCromosome);
}

var randomCromosome = function(){
    var temp = {};
    var x = parseInt(Math.random()*(xSize + 1));
    var y = parseInt(Math.random()*(ySize + 1));
    temp.x = (x / nDecimals) + minX;
    temp.y = (y / nDecimals) + minY;
    temp.adn = new Array(adnSize);
    temp.fitness = fitness(temp.x, temp.y);
    var yCount = 0;
    for(var i = adnSize - 1; i >= 0; i--){
        if(yCount <= yBits){
            temp.adn[i] = parseInt(y % 2);
            y = y / 2;
            yCount++;
        }
        else{
            temp.adn[i] = parseInt(x % 2);
            x = x / 2;
        }
    }
    return temp;
}

var binary2x = function(child){
    var xCount = 0, x = 0;
    for(var i=adnSize - yBits - 2; i>=0; i--){
        if(child.adn[i]){
            x += Math.pow(2, xCount);
        }
        xCount++;
    }
    child.x = x / nDecimals + minX;
}

var binary2y = function(child){
    var yCount = 0, y = 0;
    for(var i=adnSize - 1; i>=0; i--){
        if(child.adn[i] && yCount <= yBits){
            y += Math.pow(2, yCount);
        }
        yCount++;
    }
    child.y = y / nDecimals + minY;
}

var crossover = function(couple){
    var bitSplitter = parseInt(Math.random()*adnSize);
    var mutation1 = parseInt(Math.random()*adnSize);
    var mutation2 = parseInt(Math.random()*adnSize);
    var doMutation = parseInt(Math.random()*100);
    var child1 = {adn: new Array(adnSize)};
    var child2 = {adn: new Array(adnSize)};

    for(var i=0; i<adnSize; i++){
        if(i < bitSplitter){
            child1.adn[i] = couple.parent1.adn[i];
            child2.adn[i] = couple.parent2.adn[i];
        }
        else{
            child1.adn[i] = couple.parent2.adn[i];
            child2.adn[i] = couple.parent1.adn[i];
        }
        if(doMutation < MUTATION){
            if(i == mutation1){
                child1.adn[i] = !child1.adn[i];
            }
            if(i == mutation2){
                child2.adn[i] = !child2.adn[i];
            }
        }
    }
    binary2x(child1);
    binary2y(child1);
    binary2x(child2);
    binary2y(child2);
    child1.fitness = fitness(child1.x, child1.y);
    child2.fitness = fitness(child2.x, child2.y);
    population.push(child1, child2);
}

main();

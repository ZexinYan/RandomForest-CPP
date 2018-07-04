#include "include/DecisionTree.h"
#include "include/Data.h"
#include "include/RandomForest.h"

int main() {
    Data trainData(true, 1719692);
    trainData.read("../data/train.txt");

    RandomForest randomForest(100, "gini", "log2", -1, 150, 1, 1000000, 8);

    randomForest.fit(trainData);

    auto results = randomForest.predictProba(trainData);
    writeDataToCSV(results, trainData, "../results/trainResults.csv");
    return 0;
}

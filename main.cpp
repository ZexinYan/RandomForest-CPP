#include "include/DecisionTree.h"
#include "include/Data.h"
#include "include/RandomForest.h"

int main() {
    Data trainData(true, 1719692);
    trainData.read("../data/train.txt");
//
    RandomForest randomForest(100, "gini", "log2", -1, 150, 1, 1000000, 8);

    randomForest.fit(trainData);

    Data testData(true, 429923);
    testData.read("../data/test.txt");

    auto results = randomForest.predictProba(testData);
    writeDataToCSV(results, testData, "../results/trainResults.csv", false);
    return 0;
}

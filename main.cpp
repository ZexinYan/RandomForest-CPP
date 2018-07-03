#include "include/DecisionTree.h"
#include "include/Data.h"
//#include "include/randomForest.h"

int main() {
    Data trainData(true, 1719692);
    trainData.read("../data/train.txt");

    DecisionTree tree("gini", 1, 100, 1, -1, "log2");

    tree.fit(trainData);

//    vector<double> results(trainData.getSampleSize(), 0);
//    tree.predictProba(trainData, results);
//    writeDataToCSV(results, trainData, "../results/train.csv");
    return 0;
}

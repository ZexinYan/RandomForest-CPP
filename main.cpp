#include "include/DecisionTree.h"
#include "include/Data.h"
//#include "include/randomForest.h"

int main() {
    Data trainData(true, 1719692);
    trainData.read("../data/train.txt");

    for (int i = 0; i < 1000000; i++) {
        for (int j = 0; j < 201; j++) {
            trainData.readFeature(i, j);
//            cout << i << " " << j << " " << trainData.readFeature(i, j) << endl;
        }
    }

//    DecisionTree tree("gini", 10, 100, 1, -1, "log2");
//
//    tree.fit(trainData);
//
//    vector<double> results(trainData.getSampleSize(), 0);
//    tree.predictProba(trainData, results);
//    writeDataToCSV(results, trainData, "../results/train.csv");
    return 0;
}

//
// Created by Aster on 04/07/2018.
//

#include "../include/RandomForest.h"

void RandomForest::fit(Data &trainData) {
//    ThreadPool pool(this->nJobs);
//    std::vector<std::future<DecisionTree>> results;
//    for (int i = 0; i < nEstimators; i++) {
//        results.emplace_back(pool.enqueue([&] {
//            DecisionTree tree(criterion, maxDepth, minSamplesSplit, minSamplesLeaf,
//                              eachTreeSamplesNum, maxFeatures);
//            tree.fit(trainData);
//            return tree;
//        }));
//    }
//
//    for (auto &&each : results) {
//        decisionTrees.push_back(each.get());
//    }
    for (int i = 0; i < nEstimators; i++) {
        DecisionTree tree(criterion, maxDepth, minSamplesSplit, minSamplesLeaf,
                          eachTreeSamplesNum, maxFeatures);
        tree.fit(trainData);
        decisionTrees.push_back(tree);
        cout << "Tree: " << i << endl;
    }
}

void RandomForest::norm(vector<double> &total) {
    for (double &i : total) { i /= nEstimators; }
}

//void vecAdd(vector<double> &total, vector<double> &part) {
//    for (int i = 0; i < total.size(); i++) {
//        total[i] += part[i];
//    }
//}

vector<double> RandomForest::predictProba(Data &Data) {
    vector<double> results(Data.getSampleSize(), 0);
    for (int i = 0; i < nEstimators; i++) {
        decisionTrees[i].predictProba(Data, results);
    }
    norm(results);
    return results;
}

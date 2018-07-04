//
// Created by Aster on 04/07/2018.
//

#include "../include/RandomForest.h"

void RandomForest::fit(Data &trainData) {
    ThreadPool pool(this->nJobs);
    std::vector<std::future<DecisionTree>> results;
    for (int i = 0; i < nEstimators; i++) {
        results.emplace_back(pool.enqueue([&, i] {
            DecisionTree tree(criterion, maxDepth, minSamplesSplit, minSamplesLeaf,
                              eachTreeSamplesNum, maxFeatures);
            tree.fit(trainData);
            cout << "Fitted Tree: " << i << endl;
            return tree;
        }));
    }

    for (auto &&each : results) {
        decisionTrees.push_back(each.get());
    }
}

void RandomForest::norm(vector<double> &total) {
    for (double &i : total) { i /= nEstimators; }
}

void vecAdd(vector<double> &total, vector<double> &part) {
    for (int i = 0; i < total.size(); i++) {
        total[i] += part[i];
    }
}

vector<double> RandomForest::predictProba(Data &Data) {
    ThreadPool pool(this->nJobs);
    vector<future<vector<double>>> poolResults;
    for (int i = 0; i < nEstimators; i++) {
        poolResults.emplace_back(pool.enqueue([&, i] {
            vector<double> results(Data.getSampleSize(), 0);
            decisionTrees[i].predictProba(Data, results);
            cout << "Predict Tree: " << i << endl;
            return results;
        }));
    }
    vector<double> results(Data.getSampleSize(), 0);
    for (auto &&each : poolResults) {
        auto tmpResults = each.get();
        vecAdd(results, tmpResults);
    }
    norm(results);
    return results;
}

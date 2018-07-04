//
// Created by Aster on 25/06/2018.
//

#ifndef RANDOMFOREST_RANDOMFOREST_H
#define RANDOMFOREST_RANDOMFOREST_H

#include "DecisionTree.h"
#include "Data.h"
#include "ThreadsPool.h"

class RandomForest {
private:
    vector<DecisionTree> decisionTrees;
    int nEstimators;
    string criterion;
    string maxFeatures;
    int maxDepth;
    int minSamplesSplit;
    int minSamplesLeaf;
    int eachTreeSamplesNum;
    int nJobs;

    void norm(vector<double> &total);

public:
    /**
     *
     * @param nEstimators The number of trees in the forest.
     * @param criterion The function to measure the quality of a split.
     * Supported criteria are “gini” for the Gini impurity and “entropy” for
     * the information gain.
     * @param maxFeatures
     *          * If “auto”, then max_features=sqrt(n_features).
     *          * If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
     *          * If “log2”, then max_features=log2(n_features).
     *          * If "None", then max_features=n_features.
     * @param maxDepth The maximum depth of the tree.
     * @param minSamplesSplit The minimum number of samples required to split
     * an internal node.
     * @param minSamplesLeaf The minimum number of samples required to be at
     * a leaf node.
     * @param nJobs The number of jobs to run in parallel for both fit and
     * predict. If -1, then the number of jobs is set to the number of cores.
     */
    RandomForest(int nEstimators = 10, string criterion = "gini",
                 string maxFeatures = "auto", int maxDepth = -1,
                 int minSamplesSplit = 2, int minSamplesLeaf = 1,
                 int eachTreeSamplesNum = 1000000,
                 int nJobs = 1) : nEstimators(nEstimators),
                                  criterion(std::move(criterion)),
                                  maxFeatures(maxFeatures),
                                  maxDepth(maxDepth),
                                  minSamplesSplit(minSamplesSplit),
                                  minSamplesLeaf(minSamplesLeaf),
                                  nJobs(nJobs),
                                  eachTreeSamplesNum(eachTreeSamplesNum) {
        decisionTrees.reserve(nEstimators);
    }

    void fit(Data &trainData);

    vector<double> predictProba(Data &Data);
};

#endif //RANDOMFOREST_RANDOMFOREST_H

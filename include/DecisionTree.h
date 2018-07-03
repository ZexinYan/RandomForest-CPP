//
// Created by Aster on 25/06/2018.
//

#ifndef RANDOMFOREST_DECISIONTREE_H
#define RANDOMFOREST_DECISIONTREE_H

#include "Data.h"
#include <memory>
#include <functional>
#include <set>
#include <utility>
#include <cmath>

using namespace std;

double computeTargetProb(vector<int> &samplesVec, Data &Data);

//double computeEntropy(vector<int> &samples, Data &Data);

double computeGini(int &, int &);

//double computeInformationGain(vector<int> &samples,
//                              vector<int> &samplesLeft,
//                              vector<int> &samplesRight,
//                              Data &Data);

double computeGiniIndex(int &, int &, int &, int &);

int _sqrt(int num);

int _log2(int num);

int _none(int num);

class DecisionTree {
private:
    struct Node {
        int featureIndex;
        shared_ptr<Node> left;
        shared_ptr<Node> right;
        double threshold;
        bool isLeaf;
        int depth;
        double prob;

        Node() {
            left = nullptr;
            right = nullptr;
            isLeaf = false;
        }
    };

    int featureNum;
    int maxDepth;
    int minSamplesSplit;
    int minSamplesLeaf;
    int sampleNum;
    function<double(int&, int&, int&, int&)> criterionFunc;
    function<int(int)> maxFeatureFunc;
    shared_ptr<Node> root;

    set<double> getValuesRange(int &featureIndex,
                               vector<int> &samplesVec,
                               Data &Data);

    void splitSamplesVec(int &featureIndex, double &threshold,
                         vector<int> &samplesVec, vector<int> &samplesLeft,
                         vector<int> &samplesRight, Data &Data);

    void chooseBestSplitFeatures(shared_ptr<Node> &node,
                                 vector<int> &samplesVec,
                                 Data &Data);

    shared_ptr<Node> constructNode(vector<int> &sampleVec,
                                   Data &trainData,
                                   int depth);

public:
    /**
     *
     * @param criterion The function to measure the quality of a split.
     * Supported criteria are “gini” for the Gini impurity and “entropy"
     * for the information gain.
     * @param maxDepth The maximum depth of the tree. If 0, then nodes are
     * expanded until all leaves are pure or until
     * all leaves contain less than min_samples_split samples.
     * @param minSamplesSplit The minimum number of samples
     * required to split an internal node
     * @param minSamplesLeaf The minimum number of samples
     * required to be at a leaf node
     * @param sampleNum The number of samples to consider when constructing
     * tree.
     * @param maxFeatures The number of features to consider when looking for
     * the best split.
     */
    explicit DecisionTree(const string &criterion = "gini",
                          int maxDepth = -1,
                          int minSamplesSplit = 2,
                          int minSamplesLeaf = 1,
                          int sampleNum=-1,
                          const string &maxFeatures = "auto");

    void fit(Data &trainData);

    double computeProb(int sampleIndex, Data &Data);

    void predictProba(Data &Data, vector<double> &results);
};

#endif //RANDOMFOREST_DECISIONTREE_H

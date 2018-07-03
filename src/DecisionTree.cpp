//
// Created by Aster on 02/07/2018.
//

#include "../include/DecisionTree.h"

int computeTure(vector<int> &samplesVec, Data &Data) {
    int total = 0;
    for (auto each : samplesVec) {
        total += Data.readTarget(each);
    }
    return total;
}

double computeTargetProb(vector<int> &samplesVec, Data &Data) {
    double num = 0;
    int total = 0;
    for (auto i : samplesVec) {
        if (i != -1) {
            num += Data.readTarget(i);
            total++;
        }
    }
    return num / (total + 0.000000001);
}

double getSize(vector<int> &samples) {
    double num = 0;
    for (auto i : samples) {
        if (i != -1) { num++; }
    }
    return num;
}

//double computeEntropy(vector<int> &samples, Data &Data) {
//    double trueProb = computeTargetProb(samples, Data);
//    return -1 * (trueProb * log2(trueProb)
//                 + (1 - trueProb) * log2((1 - trueProb)));
//}

double computeGini(int& sideTrue, int& sideSize) {
    double trueProb = (sideTrue * 1.0) / (sideSize + 0.00000001);
    return (1 - trueProb * trueProb - (1 - trueProb) * (1 - trueProb));
}

//double computeInformationGain(vector<int> &samples,
//                              vector<int> &samplesLeft,
//                              vector<int> &samplesRight,
//                              Data &Data) {
//    return -1 * computeEntropy(samples, Data)
//           + ( getSize(samplesLeft) / getSize(samples))
//             * computeEntropy(samplesLeft, Data)
//           + (getSize(samplesRight) / getSize(samples))
//             * computeEntropy(samplesRight, Data);
//}

double computeGiniIndex(int& leftTrue, int& leftSize,
                        int& rightTrue, int& rightSize) {
    double leftProb = (leftSize * 1.0) / (leftSize + rightSize);
    double rightprob = (rightSize * 1.0) / (leftSize + rightSize);
    return leftProb * computeGini(leftTrue, leftSize)
           + rightprob * computeGini(rightTrue, rightSize);
}

int _sqrt(int num) {
    return int(sqrt(num));
}

int _log2(int num) {
    return int(log2(num));
}

int _none(int num) {
    return num;
}

set<double> DecisionTree::getValuesRange(int &featureIndex,
                                         vector<int> &samplesVec,
                                         Data &Data) {
    set<double> featureRange;
    for (auto sampleIndex : samplesVec) {
        featureRange.insert(
                Data.readFeature(sampleIndex, featureIndex));
    }
    return featureRange;
}

void DecisionTree::splitSamplesVec(int &featureIndex, double &threshold,
                                   vector<int> &samplesVec,
                                   vector<int> &samplesLeft,
                                   vector<int> &samplesRight, Data &Data) {
    samplesLeft.clear();
    samplesRight.clear();
    for (auto samplesIndex : samplesVec) {
        if (Data.readFeature(samplesIndex, featureIndex) >
            threshold) {
            samplesRight.push_back(samplesIndex);
        } else {
            samplesLeft.push_back(samplesIndex);
        }
    }
}

void DecisionTree::chooseBestSplitFeatures(shared_ptr<Node> &node,
                                           vector<int> &samplesVec,
                                           Data &Data) {
    vector<int> featuresVec = Data.generateFeatures(this->maxFeatureFunc);
    int bestFeatureIndex = featuresVec[0];
    int samplesTrueNum = computeTure(samplesVec, Data);
    double minValue = 1000000000, bestThreshold = 0;
    double threshold = 0, splitInfo = 0;
    int sampleIndex;
    for (auto featureIndex : featuresVec) {
        Data.sortByFeature(samplesVec, featureIndex);
        int leftSize = 0, rightSize = (int)samplesVec.size();
        int leftTrue = 0, rightTrue = samplesTrueNum;
        for (int index = 0; index < samplesVec.size();) {
            sampleIndex = samplesVec[index];
            threshold = Data.readFeature(sampleIndex, featureIndex);
            while (index < samplesVec.size() &&
                   Data.readFeature(sampleIndex, featureIndex) <= threshold) {
                leftSize++;
                rightSize--;
                if (Data.readTarget(sampleIndex) == 1) {
                    leftTrue++;
                    rightTrue--;
                }
                index++;
                sampleIndex = samplesVec[index];
            }
            if (index == samplesVec.size()) { continue; }
            double value = criterionFunc(leftTrue, leftSize, rightTrue, rightSize);
            if (value <= minValue) {
                if (value == minValue) {
                    int info = (leftSize / 100) * (rightSize / 100);
                    if (info < splitInfo) {
                        continue;
                    } else {
                        splitInfo = info;
                    }
                }
                minValue = value;
                bestThreshold = threshold;
                bestFeatureIndex = featureIndex;
            }
        }
    }
    node->featureIndex = bestFeatureIndex;
    node->threshold = bestThreshold;
//    cout << node->featureIndex << " " << node->threshold << endl;
}

shared_ptr<DecisionTree::Node>
DecisionTree::constructNode(vector<int> &samplesVec,
                            Data &trainData,
                            int depth) {
    double targetProb = computeTargetProb(samplesVec, trainData);
    shared_ptr<Node> node(new Node());
    node->depth = depth;
    node->prob = 0;
    if (targetProb == 0 || targetProb == 1 ||
        samplesVec.size() <= minSamplesSplit || depth == maxDepth) {
        node->isLeaf = true;
        node->prob = targetProb;
    } else {
        chooseBestSplitFeatures(node, samplesVec, trainData);
        vector<int> sampleLeft, sampleRight;
        splitSamplesVec(node->featureIndex, node->threshold, samplesVec,
                        sampleLeft, sampleRight, trainData);
////        cout << sampleLeft.size() << " " << sampleRight.size() << endl;
        if ((sampleLeft.size() < minSamplesLeaf) or
            (sampleRight.size() < minSamplesLeaf)) {
            node->isLeaf = true;
            node->prob = targetProb;
        } else {
            node->left = constructNode(sampleLeft, trainData,
                    depth + 1);
            node->right = constructNode(sampleRight, trainData,
                    depth + 1);
        }
    }
    return node;

}

DecisionTree::DecisionTree(const string &criterion,
                           int maxDepth,
                           int minSamplesSplit,
                           int minSamplesLeaf,
                           int sampleNum,
                           const string &maxFeatures) {
    if (criterion == "gini") {
        this->criterionFunc = computeGiniIndex;
    } else if (criterion == "entropy") {
//        this->criterionFunc = computeInformationGain;
    } else {
        this->criterionFunc = computeGiniIndex;
    }

    if (maxFeatures == "auto" || maxFeatures == "sqrt") {
        this->maxFeatureFunc = _sqrt;
    } else if (maxFeatures == "log2") {
        this->maxFeatureFunc = _log2;
    } else {
        this->maxFeatureFunc = _none;
    }
    this->sampleNum = sampleNum;
    this->maxDepth = maxDepth;
    this->minSamplesSplit = minSamplesSplit;
    this->minSamplesLeaf = minSamplesLeaf;
}

void DecisionTree::fit(Data &trainData) {
    vector<int> samplesVec = trainData.generateSample(this->sampleNum);
    root = constructNode(samplesVec, trainData, 0);
}

double DecisionTree::computeProb(int sampleIndex, Data &Data) {
    auto node = root;
    while (!node->isLeaf) {
        if (Data.readFeature(sampleIndex, node->featureIndex) >
            node->threshold) {
            node = node->right;
        } else {
            node = node->left;
        }
    }
    return node->prob;
}

void DecisionTree::predictProba(Data &Data, vector<double> &results) {
    for (int i = 0; i < results.size(); i++) {
        results[i] = computeProb(i, Data);
    }
}

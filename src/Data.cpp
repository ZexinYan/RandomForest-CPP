//
// Created by Aster on 02/07/2018.
//

#include "../include/Data.h"

vector<string> splitBySpace(string &sentence) {
    istringstream iss(sentence);
    return vector<string>{istream_iterator<string>(iss),
                          istream_iterator<string>{}};
}

void writeDataToCSV(vector<double> &results, Data &data,
                    const string &filename, bool train) {
    ofstream out(filename);
    if (out.is_open()) {
        out << "id,label,real\n";
        int i = 0;
        for (auto each : results) {
            out << i << "," << each;
            if (train) {
                out << "," << data.readTarget(i) << "\n";
            } else {
                out << "\n";
            }
            i++;
        }
        out.close();
    } else {
        cout << "Write File failed" << endl;
    }
}

Data::Data(bool isTrain, int size, int featuresSize) {
    this->featureSize = featuresSize;
    features.reserve(size);
    samplesVec.reserve(size);
    if (isTrain) { target.reserve(size); }
    this->isTrain = isTrain;
}

void Data::read(const string &filename) {
    ifstream inputFile;
    inputFile.open(filename.c_str());
    if (!inputFile.is_open()) { cout << "Failed Open" << endl; }
    string str;
    int startIndex = this->isTrain ? 1 : 0;
    while (getline(inputFile, str)) {
        auto results = splitBySpace(str);
        vector<double> sample(this->featureSize, 0);
        for (int i = startIndex; i < results.size(); i++) {
            int key = atoi(
                    results[i].substr(0, results[i].find(":")).c_str());
            double value = atof(results[i].substr(
                    results[i].find(":") + 1).c_str());
            sample[key] = value;
        }
        this->features.push_back(sample);
        if (this->isTrain) { target.push_back(atoi(results[0].c_str())); }
        samplesVec.push_back(this->samplesSize++);
    }
    inputFile.close();
    featuresVec.reserve(this->featureSize);
    for (int i = 0; i < featureSize; i++) { featuresVec.push_back(i); }
}

double Data::readFeature(int sampleIndex, int featureIndex) {
    return features[sampleIndex][featureIndex];
}

int Data::readTarget(int sampleIndex) {
    return target[sampleIndex];
}

int Data::getSampleSize() {
    return (int) features.size();
}

int Data::getFeatureSize() {
    return featureSize;
}

vector<int> Data::generateSample(int &num) {
    if (num == -1) {
        return samplesVec;
    } else {
        random_shuffle(samplesVec.begin(), samplesVec.end());
        return vector<int>(samplesVec.begin(), samplesVec.begin() + num);
    }
}

vector<int> Data::generateFeatures(function<int(int)> &func) {
    int m = func(this->getFeatureSize());
    random_shuffle(featuresVec.begin(), featuresVec.end());
    return vector<int>(featuresVec.begin(), featuresVec.begin() + m);
}

void Data::sortByFeature(vector<int> &samplesVec, int featureIndex) {
    sort(samplesVec.begin(), samplesVec.end(), [&](int a, int b) {
        return this->readFeature(a, featureIndex) <
               this->readFeature(b, featureIndex);
    });
}

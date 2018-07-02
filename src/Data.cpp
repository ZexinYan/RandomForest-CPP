//
// Created by Aster on 02/07/2018.
//

#include "../include/Data.h"

vector<string> splitBySpace(string &sentence) {
    istringstream iss(sentence);
    return vector<string>{istream_iterator<string>(iss),
                          istream_iterator<string>{}};
}

void writeDataToCSV(vector<double> &results, Data &data, const string &filename) {
    ofstream out(filename);
    if (out.is_open()) {
        out << "id,label,real\n";
        int i = 0;
        for (auto each : results) {
            out << i << "," << each << "," << data.readTarget(i) << "\n";
            i++;
        }
        out.close();
    } else {
        cout << "Write File failed" << endl;
    }
}

Data::Data(bool isTrain, int size) {
    features.reserve(size);
    if (isTrain) { target.reserve(size); }
    this->isTrain = isTrain;
}

void Data::read(const string &filename) {
    ifstream inputFile;
    inputFile.open(filename.c_str());
    if (!inputFile.is_open()) {
        cout << "Failed Open" << endl;
    }
    string str;
    int startIndex = this->isTrain ? 1 : 0;
    while (getline(inputFile, str)) {
        auto results = splitBySpace(str);
        map<int, double> sample;
        for (int i = startIndex; i < results.size(); i++) {
            int key = atoi(
                    results[i].substr(0, results[i].find(":")).c_str());
            double value = atof(results[i].substr(
                    results[i].find(":") + 1).c_str());
            sample[key] = value;
            featureSize = max(featureSize, key);
        }
        features.push_back(sample);
        if (this->isTrain) { target.push_back(atoi(results[0].c_str())); }
    }
    inputFile.close();
}

double Data::readFeature(int sampleIndex, int featureIndex) {
    if (features[sampleIndex].find(featureIndex) !=
        features[sampleIndex].end()) {
        return features[sampleIndex][featureIndex];
    } else {
        return 0;
    }
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

vector<int> Data::generateSample(int n) {
    default_random_engine generator(time(NULL));
    uniform_int_distribution<int> distribution(0, getSampleSize() - 1);
    vector<int> randomSample(n, 0);
    for (int i = 0; i < n; i++) {
        randomSample[i] = distribution(generator);
    }
    return randomSample;
}

vector<int> Data::generateFeatures(int m) {
    default_random_engine generator(time(NULL));
    uniform_int_distribution<int> distribution(0, featureSize - 1);
    vector<int> randomSample(m, 0);
    for (int i = 0; i < m; i++) {
        randomSample[i] = distribution(generator);
        cout << randomSample[i] << " ";
    }
    cout << endl;
    return randomSample;
}

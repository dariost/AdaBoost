#include "AdaBoost.hpp"
#include <cmath>

AdaBoost::AdaBoost(std::shared_ptr<DatasetView> dataset_, int32_t label_) {
    dataset = dataset_;
    label = label_;
    w.resize(dataset->get_n_samples(), 1.0 / double(dataset->get_n_samples()));
}

double AdaBoost::predict(const std::vector<double>& point) const {
    double result = 0.0;
    for(size_t i = 0; i < a.size(); i++) {
        result += a[i] * h[i].predict(point);
    }
    return std::tanh(result);
}

size_t AdaBoost::get_epoch() { return a.size(); }

void AdaBoost::next_epoch() {
    h.emplace_back(dataset, label, w);
    double epsilon = 0.0;
    std::vector<bool> predictions(dataset->get_n_samples());
    for(size_t sample = 0; sample < dataset->get_n_samples(); sample++) {
        std::vector<double> point(dataset->get_n_features());
        for(size_t i = 0; i < dataset->get_n_features(); i++) {
            point[i] = dataset->get_value(sample, i);
        }
        int32_t other_label = dataset->get_label(sample);
        int32_t this_label = (h.back().predict(point) > 0 ? label : -1);
        if(other_label != label) {
            other_label = -1;
        }
        predictions[sample] = (this_label != other_label);
        epsilon += w[sample] * double(predictions[sample]);
    }
    a.push_back(0.5 * log((1.0 - epsilon) / epsilon));
    for(size_t sample = 0; sample < dataset->get_n_samples(); sample++) {
        if(predictions[sample]) {
            w[sample] /= 2.0 * epsilon;
        } else {
            w[sample] /= 2.0 * (1.0 - epsilon);
        }
    }
}

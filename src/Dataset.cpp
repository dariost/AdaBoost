#include "Dataset.hpp"
#include <cassert>

Dataset::Dataset(size_t samples, size_t features, size_t labels) {
    next_data_index = 0;
    n_samples = samples;
    n_features = features;
    possible_labels.resize(labels);
    label.resize(samples);
    data.resize(samples * features);
}

void Dataset::set_labels(const std::vector<int32_t>& labels) {
    assert(possible_labels.size() == labels.size());
    for(size_t i = 0; i < labels.size(); i++) {
        possible_labels[i] = labels[i];
    }
}

void Dataset::add_sample(int32_t point_label, const std::vector<double>& point) {
    assert(point.size() == n_features);
    size_t k = next_data_index++;
    assert(k < n_samples);
    label[k] = point_label;
    for(size_t i = 0; i < point.size(); i++) {
        data[i * n_samples + k] = point[i];
    }
}

void Dataset::finalize() { construct_feature_order(); }

int32_t Dataset::get_label(size_t sample) const { return label[sample]; }

size_t Dataset::get_n_features() const { return n_features; }

size_t Dataset::get_n_samples() const { return n_samples; }

const std::vector<int32_t>& Dataset::get_labels() const { return possible_labels; }

double Dataset::get_value(size_t sample, size_t feature) const { return data[feature * n_samples + sample]; }

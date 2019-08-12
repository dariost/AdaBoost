#include "DatasetView.hpp"
#include <algorithm>
#include <cassert>
#include <numeric>

DatasetView::DatasetView() {}

DatasetView::DatasetView(std::shared_ptr<DatasetView> dataset_) {
    dataset = dataset_;
    range_indices.emplace_back(0, dataset->get_n_samples());
    num_samples = dataset->get_n_samples();
    construct_feature_order();
}

DatasetView::DatasetView(std::shared_ptr<DatasetView> dataset_, const std::vector<std::pair<size_t, size_t>>& indices_) {
    dataset = dataset_;
    range_indices = indices_;
    num_samples = 0;
    for(size_t i = 0; i < range_indices.size(); i++) {
        num_samples += range_indices[i].second - range_indices[i].first;
    }
    construct_feature_order();
}

int32_t DatasetView::get_label(size_t sample) const { return dataset->get_label(get_index(sample)); }

size_t DatasetView::get_n_features() const { return dataset->get_n_features(); }

size_t DatasetView::get_n_samples() const { return num_samples; }

const std::vector<int32_t>& DatasetView::get_labels() const { return dataset->get_labels(); }

double DatasetView::get_value(size_t sample, size_t feature) const { return dataset->get_value(get_index(sample), feature); }

const std::vector<size_t>& DatasetView::get_feature_order(size_t feature) const { return feature_order[feature]; }

void DatasetView::construct_feature_order() {
    feature_order.resize(get_n_features(), std::vector<size_t>(get_n_samples()));
    for(size_t i = 0; i < get_n_features(); i++) {
        std::iota(feature_order[i].begin(), feature_order[i].end(), 0);
    }
    for(size_t i = 0; i < get_n_features(); i++) {
        std::sort(feature_order[i].begin(), feature_order[i].end(),
                  [&i, this](const size_t& a, const size_t& b) { return this->get_value(a, i) < this->get_value(b, i); });
    }
}

std::vector<double> DatasetView::get_sample(size_t sample) const {
    std::vector<double> point(get_n_features());
    for(size_t i = 0; i < get_n_features(); i++) {
        point[i] = get_value(sample, i);
    }
    return point;
}

size_t DatasetView::get_index(size_t index) const {
    size_t acc = 0;
    for(size_t i = 0; i < range_indices.size(); i++) {
        if(acc + range_indices[i].second - range_indices[i].first > index) {
            return range_indices[i].first + index - acc;
        }
        acc += range_indices[i].second - range_indices[i].first;
    }
    assert(!"This shouldn't happen");
}

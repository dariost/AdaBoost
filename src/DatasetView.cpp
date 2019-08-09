#include "DatasetView.hpp"
#include <algorithm>
#include <numeric>

DatasetView::DatasetView() {}

DatasetView::DatasetView(std::shared_ptr<DatasetView> dataset_) {
    dataset = dataset_;
    indices.resize(dataset->get_n_samples());
    std::iota(indices.begin(), indices.end(), 0);
    construct_feature_order();
}

DatasetView::DatasetView(std::shared_ptr<DatasetView> dataset_, const std::vector<size_t>& indices_) {
    dataset = dataset_;
    indices = indices_;
    construct_feature_order();
}

int32_t DatasetView::get_label(size_t sample) const { return dataset->get_label(indices[sample]); }

size_t DatasetView::get_n_features() const { return dataset->get_n_features(); }

size_t DatasetView::get_n_samples() const { return indices.size(); }

const std::vector<int32_t>& DatasetView::get_labels() const { return dataset->get_labels(); }

double DatasetView::get_value(size_t sample, size_t feature) const { return dataset->get_value(indices[sample], feature); }

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

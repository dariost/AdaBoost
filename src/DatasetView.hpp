#pragma once

#include <memory>
#include <vector>

class DatasetView {
  protected:
    std::shared_ptr<DatasetView> dataset;
    std::vector<std::pair<size_t, size_t>> range_indices;
    std::vector<std::vector<size_t>> feature_order;
    size_t num_samples;
    DatasetView();
    void construct_feature_order();
    size_t get_index(size_t index) const;

  public:
    DatasetView(std::shared_ptr<DatasetView> dataset_);
    DatasetView(std::shared_ptr<DatasetView> dataset_, const std::vector<std::pair<size_t, size_t>>& indices_);
    virtual int32_t get_label(size_t sample) const;
    virtual size_t get_n_features() const;
    virtual size_t get_n_samples() const;
    virtual const std::vector<int32_t>& get_labels() const;
    virtual double get_value(size_t sample, size_t feature) const;
    virtual const std::vector<size_t>& get_feature_order(size_t feature) const;
    virtual std::vector<double> get_sample(size_t sample) const;
};

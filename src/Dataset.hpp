#pragma once

#include "DatasetView.hpp"
#include <cstdint>
#include <vector>

class Dataset : public DatasetView {
  private:
    size_t next_data_index;

  protected:
    std::vector<double> data;
    std::vector<int32_t> label;
    std::vector<int32_t> possible_labels;
    size_t n_samples;
    size_t n_features;

  public:
    Dataset(size_t samples, size_t features, size_t labels);
    void set_labels(const std::vector<int32_t>& labels);
    void add_sample(int32_t label, const std::vector<double>& point);
    int32_t get_label(size_t sample) const;
    size_t get_n_features() const;
    size_t get_n_samples() const;
    const std::vector<int32_t>& get_labels() const;
    double get_value(size_t sample, size_t feature) const;
    void finalize();
};

#pragma once

#include "DatasetView.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

template <typename T>
class OneVsAll {
  protected:
    std::shared_ptr<DatasetView> dataset;
    std::vector<std::pair<int32_t, T>> predictor;

  public:
    OneVsAll(std::shared_ptr<DatasetView> dataset_, uint32_t base_seed) {
        dataset = dataset_;
        for(const auto& label : dataset->get_labels()) {
            predictor.emplace_back(label, T(dataset, label, base_seed + label));
        }
    }

    int32_t predict(const std::vector<double>& point) const {
        int32_t best_label = -1;
        double best_value = -INFINITY;
        for(size_t i = 0; i < predictor.size(); i++) {
            double pvalue = predictor[i].second.predict(point);
            if(pvalue > best_value) {
                best_value = pvalue;
                best_label = predictor[i].first;
            }
        }
        return best_label;
    }

    void next_epoch() {
        for(size_t i = 0; i < predictor.size(); i++) {
            predictor[i].second.next_epoch();
        }
    }

    size_t get_epoch() { return predictor[0].second.get_epoch(); }
};

#pragma once

#include "DatasetView.hpp"
#include "DecisionStump.hpp"
#include <random>

class Bagging {
  protected:
    std::shared_ptr<DatasetView> dataset;
    std::vector<DecisionStump> h;
    int32_t label;
    std::mt19937 r;

  public:
    Bagging(std::shared_ptr<DatasetView> dataset_, int32_t label_, uint32_t seed);
    double predict(const std::vector<double>& point) const;
    size_t get_epoch();
    void next_epoch();
};

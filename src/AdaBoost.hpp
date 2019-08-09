#pragma once

#include "DatasetView.hpp"
#include "DecisionStump.hpp"
#include "Predictor.hpp"
#include <memory>

class AdaBoost : public Predictor {
  protected:
    std::shared_ptr<DatasetView> dataset;
    std::vector<double> w;
    std::vector<double> a;
    std::vector<DecisionStump> h;
    int32_t label;

  public:
    AdaBoost(std::shared_ptr<DatasetView> dataset_, int32_t label_);
    double predict(const std::vector<double>& point) const;
    size_t get_epoch();
    void next_epoch();
};

#pragma once

#include "DatasetView.hpp"
#include <memory>

class DecisionStump {
  protected:
    size_t feature;
    double cut_value;
    bool left_positive;

  public:
    DecisionStump(std::shared_ptr<DatasetView> dataset, int32_t label,
                  const std::vector<double>& w);
    double predict(const std::vector<double>& point) const;
};

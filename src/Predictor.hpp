#pragma once

#include <vector>

class Predictor {
  public:
    virtual double predict(const std::vector<double>& point) const = 0;
    virtual size_t get_epoch() = 0;
    virtual void next_epoch() = 0;
};

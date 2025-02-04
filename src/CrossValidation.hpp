#pragma once

#include "DatasetView.hpp"
#include <memory>

template <typename T>
class CrossValidation {
  protected:
    std::vector<std::shared_ptr<DatasetView>> test_set;
    std::vector<std::shared_ptr<DatasetView>> training_set;
    std::vector<T> predictors;

    double error(std::shared_ptr<DatasetView> dataset, T& predictor) {
        double total_loss = 0.0;
        for(size_t i = 0; i < dataset->get_n_samples(); i++) {
            int32_t prediction = predictor.predict(dataset->get_sample(i));
            total_loss += (prediction != dataset->get_label(i));
        }
        return total_loss / double(dataset->get_n_samples());
    }

  public:
    CrossValidation(std::shared_ptr<DatasetView> dataset, size_t k, uint32_t seed = 42) {
        size_t n = dataset->get_n_samples();
        for(size_t i = 0; i < k; i++) {
            std::vector<std::pair<size_t, size_t>> test_range, training_range;
            for(size_t j = 0; j < n; j++) {
                if(j >= n * i / k && j < n * (i + 1) / k) {
                    if(test_range.size() > 0 && test_range.back().second == j) {
                        test_range.back().second++;
                    } else {
                        test_range.emplace_back(j, j + 1);
                    }
                } else {
                    if(training_range.size() > 0 && training_range.back().second == j) {
                        training_range.back().second++;
                    } else {
                        training_range.emplace_back(j, j + 1);
                    }
                }
            }
            test_set.push_back(std::make_shared<DatasetView>(dataset, test_range));
            training_set.push_back(std::make_shared<DatasetView>(dataset, training_range));
            predictors.emplace_back(training_set.back(), seed);
        }
    }

    double test_error() {
        double total_error = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+ : total_error)
        for(size_t i = 0; i < test_set.size(); i++) {
            total_error += error(test_set[i], predictors[i]);
        }
        return total_error / double(test_set.size());
    }

    double training_error() {
        double total_error = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+ : total_error)
        for(size_t i = 0; i < training_set.size(); i++) {
            total_error += error(training_set[i], predictors[i]);
        }
        return total_error / double(training_set.size());
    }

    size_t get_epoch() { return predictors[0].get_epoch(); }

    void next_epoch() {
#pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < predictors.size(); i++) {
            predictors[i].next_epoch();
        }
    }
};

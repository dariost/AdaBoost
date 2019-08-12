#include "Bagging.hpp"

Bagging::Bagging(std::shared_ptr<DatasetView> dataset_, int32_t label_, uint32_t seed) : r(seed) {
    dataset = dataset_;
    label = label_;
}

double Bagging::predict(const std::vector<double>& point) const {
    double prediction = 0.0;
    for(size_t i = 0; i < h.size(); i++) {
        prediction += h[i].predict(point);
    }
    return prediction / double(h.size());
}

size_t Bagging::get_epoch() { return h.size(); }

void Bagging::next_epoch() {
    std::uniform_int_distribution<size_t> g(0, dataset->get_n_samples() - 1);
    std::vector<double> w(dataset->get_n_samples());
    for(size_t i = 0; i < dataset->get_n_samples(); i++) {
        w[g(r)] += 1.0;
    }
    h.emplace_back(dataset, label, w);
}

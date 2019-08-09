#include "DecisionStump.hpp"
#include <cmath>

DecisionStump::DecisionStump(std::shared_ptr<DatasetView> dataset, int32_t label, const std::vector<double>& w) {
    auto y = [&label](const int32_t l) { return (l == label) ? 1.0 : -1.0; };
    size_t best_feature = -1;
    double best_cut_value = -INFINITY;
    double best_edge = 0.0;
    double constant_edge = 0.0;
    for(size_t i = 0; i < dataset->get_n_samples(); i++) {
        constant_edge += w[i] * y(dataset->get_label(i));
    }
    for(size_t feat = 0; feat < dataset->get_n_features(); feat++) {
        double edge = constant_edge;
        for(size_t sample_indirect = 1; sample_indirect < dataset->get_n_samples(); sample_indirect++) {
            size_t prev_sample = dataset->get_feature_order(feat)[sample_indirect - 1];
            size_t sample = dataset->get_feature_order(feat)[sample_indirect];
            edge -= 2.0 * w[prev_sample] * y(dataset->get_label(prev_sample));
            if(dataset->get_value(prev_sample, feat) != dataset->get_value(sample, feat)) {
                if(fabs(edge) > fabs(best_edge)) {
                    best_edge = edge;
                    best_feature = feat;
                    best_cut_value = (dataset->get_value(sample, feat) + dataset->get_value(prev_sample, feat)) / 2.0;
                }
            }
        }
    }
    if(fabs(constant_edge) >= fabs(best_edge)) {
        feature = 0;
        cut_value = -INFINITY;
        left_positive = (constant_edge < 0.0);
    } else {
        feature = best_feature;
        cut_value = best_cut_value;
        left_positive = (best_edge < 0.0);
    }
}

double DecisionStump::predict(const std::vector<double>& point) const {
    if((point[feature] < cut_value) == left_positive) {
        return 1.0;
    } else {
        return -1.0;
    }
}

#include "AdaBoost.hpp"
#include "Bagging.hpp"
#include "CrossValidation.hpp"
#include "Dataset.hpp"
#include "DatasetView.hpp"
#include "OneVsAll.hpp"
#include <iomanip>
#include <iostream>
#include <memory>

using namespace std;

int main() {
    size_t n_samples, n_features, n_labels;
    cin >> n_samples >> n_features >> n_labels;
    shared_ptr<Dataset> dataset = make_shared<Dataset>(n_samples, n_features, n_labels);
    vector<int32_t> possible_labels(n_labels);
    for(size_t i = 0; i < n_labels; i++) {
        cin >> possible_labels[i];
    }
    dataset->set_labels(possible_labels);
    for(size_t i = 0; i < n_samples; i++) {
        int32_t label;
        vector<double> point(n_features);
        cin >> label;
        for(size_t j = 0; j < n_features; j++) {
            cin >> point[j];
        }
        dataset->add_sample(label, point);
    }
    dataset->finalize();
    const size_t k_fold = 20;
    CrossValidation<OneVsAll<AdaBoost>> ab(dataset, k_fold);
    CrossValidation<OneVsAll<Bagging>> bg(dataset, k_fold);
    for(size_t i = 0;; i++) {
        ab.next_epoch();
        bg.next_epoch();
        cout << "[AdaBoost] T = " << ab.get_epoch() << " -> tr. e. = " << setprecision(7) << ab.training_error();
        cout << " - te. e. = " << setprecision(7) << ab.test_error() << endl;
        cout << "[Bagging]  T = " << bg.get_epoch() << " -> tr. e. = " << setprecision(7) << bg.training_error();
        cout << " - te. e. = " << setprecision(7) << bg.test_error() << endl;
        cout << endl;
    }
    return 0;
}

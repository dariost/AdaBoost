#include "AdaBoost.hpp"
#include "Bagging.hpp"
#include "Dataset.hpp"
#include "DatasetView.hpp"
#include "OneVsAll.hpp"
#include <iomanip>
#include <iostream>
#include <memory>

using namespace std;

template <typename T>
double error(shared_ptr<DatasetView> dataset, OneVsAll<T>& ova) {
    double total_loss = 0.0;
    for(size_t i = 0; i < dataset->get_n_samples(); i++) {
        int32_t prediction = ova.predict(dataset->get_sample(i));
        total_loss += (prediction != dataset->get_label(i));
    }
    return total_loss / double(dataset->get_n_samples());
}

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
    OneVsAll<AdaBoost> ova_ab(dataset);
    OneVsAll<Bagging> ova_bg(dataset);
    for(size_t i = 0;; i++) {
        ova_ab.next_epoch();
        ova_bg.next_epoch();
        cout << "T = " << ova_ab.get_epoch() << " -> tr. e. AdaBoost = " << setprecision(15) << error(dataset, ova_ab) << endl;
        cout << "T = " << ova_bg.get_epoch() << " -> tr. e. Bagging  = " << setprecision(15) << error(dataset, ova_bg) << endl;
        cout << endl;
    }
    return 0;
}

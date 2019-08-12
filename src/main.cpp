#include "AdaBoost.hpp"
#include "Bagging.hpp"
#include "CrossValidation.hpp"
#include "Dataset.hpp"
#include "DatasetView.hpp"
#include "OneVsAll.hpp"
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace std;

bool is_quitting = false;

void sigint(int sig) {
    (void)sig;
    is_quitting = true;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, sigint);
    size_t k_fold = 20;
    if(argc > 1) {
        k_fold = atol(argv[1]);
    }
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
    const size_t prec = 15;
    CrossValidation<OneVsAll<AdaBoost>> ab(dataset, k_fold);
    CrossValidation<OneVsAll<Bagging>> bg(dataset, k_fold);
    cout << "[" << endl;
    bool to_quit = false;
    for(size_t i = 0; !to_quit; i++) {
        cout << "  {" << endl;
        ab.next_epoch();
        cout << "    \"AdaBoost\": {" << endl;
        cout << "      \"TrainingError\": " << setprecision(prec) << ab.training_error() << "," << endl;
        cout << "      \"TestError\": " << setprecision(prec) << ab.test_error() << endl;
        cout << "    }," << endl;
        bg.next_epoch();
        cout << "    \"Bagging\": {" << endl;
        cout << "      \"TrainingError\": " << setprecision(prec) << bg.training_error() << "," << endl;
        cout << "      \"TestError\": " << setprecision(prec) << bg.test_error() << endl;
        cout << "    }" << endl;
        cout << "  }";
        if(is_quitting) {
            cout << endl;
            to_quit = true;
        } else {
            cout << "," << endl;
        }
        cerr << "T = " << (i + 1) << endl;
    }
    cout << "]" << endl;
    return 0;
}

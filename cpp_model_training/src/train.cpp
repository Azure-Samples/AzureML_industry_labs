#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

// ── Argument parsing ────────────────────────────────────────────

struct Args {
    std::string data_dir;
    std::string output_dir;
    double learning_rate = 0.01;
    int epochs = 1000;
    int print_every = 100;
};

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string key = argv[i];
        std::string val = argv[i + 1];
        if (key == "--data_dir")        args.data_dir = val;
        else if (key == "--output_dir") args.output_dir = val;
        else if (key == "--learning_rate") args.learning_rate = std::stod(val);
        else if (key == "--epochs")     args.epochs = std::stoi(val);
        else if (key == "--print_every") args.print_every = std::stoi(val);
    }
    return args;
}

// ── Dataset ─────────────────────────────────────────────────────

struct Dataset {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    int n_samples  = 0;
    int n_features = 0;
};

Dataset load_csv(const std::string& path) {
    Dataset ds;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        std::exit(1);
    }

    std::string line;
    std::getline(file, line);  // skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }

        ds.y.push_back(row.back());
        row.pop_back();
        ds.X.push_back(row);
    }

    ds.n_samples  = static_cast<int>(ds.X.size());
    ds.n_features = ds.X.empty() ? 0 : static_cast<int>(ds.X[0].size());
    return ds;
}

// ── Linear model helpers ────────────────────────────────────────

double predict(const std::vector<double>& x,
               const std::vector<double>& weights,
               double bias) {
    double pred = bias;
    for (size_t j = 0; j < x.size(); j++) {
        pred += weights[j] * x[j];
    }
    return pred;
}

double compute_mse(const Dataset& ds,
                   const std::vector<double>& weights,
                   double bias) {
    double mse = 0.0;
    for (int i = 0; i < ds.n_samples; i++) {
        double err = predict(ds.X[i], weights, bias) - ds.y[i];
        mse += err * err;
    }
    return mse / ds.n_samples;
}

double compute_mae(const Dataset& ds,
                   const std::vector<double>& weights,
                   double bias) {
    double mae = 0.0;
    for (int i = 0; i < ds.n_samples; i++) {
        double err = predict(ds.X[i], weights, bias) - ds.y[i];
        mae += std::abs(err);
    }
    return mae / ds.n_samples;
}

// ── Main ────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    std::cout << "\n== C++ Linear Regression Training ==============" << std::endl;
    std::cout << "  Data dir:       " << args.data_dir << std::endl;
    std::cout << "  Output dir:     " << args.output_dir << std::endl;
    std::cout << "  Learning rate:  " << args.learning_rate << std::endl;
    std::cout << "  Epochs:         " << args.epochs << std::endl;

    // ── Load data ───────────────────────────────────────────────
    Dataset train = load_csv(args.data_dir + "/train.csv");
    Dataset test  = load_csv(args.data_dir + "/test.csv");

    std::cout << "\n  Train samples:  " << train.n_samples << std::endl;
    std::cout << "  Test samples:   " << test.n_samples << std::endl;
    std::cout << "  Features:       " << train.n_features << std::endl;

    // ── Initialise weights to zero ──────────────────────────────
    std::vector<double> weights(train.n_features, 0.0);
    double bias = 0.0;

    std::vector<double> best_weights = weights;
    double best_bias = bias;
    double best_mse  = 1e18;

    // ── Gradient descent ────────────────────────────────────────
    std::cout << "\n== Training ====================================" << std::endl;

    for (int epoch = 1; epoch <= args.epochs; epoch++) {
        std::vector<double> grad_w(train.n_features, 0.0);
        double grad_b = 0.0;

        for (int i = 0; i < train.n_samples; i++) {
            double err = predict(train.X[i], weights, bias) - train.y[i];
            for (int j = 0; j < train.n_features; j++) {
                grad_w[j] += (2.0 / train.n_samples) * err * train.X[i][j];
            }
            grad_b += (2.0 / train.n_samples) * err;
        }

        for (int j = 0; j < train.n_features; j++) {
            weights[j] -= args.learning_rate * grad_w[j];
        }
        bias -= args.learning_rate * grad_b;

        double train_mse = compute_mse(train, weights, bias);
        if (train_mse < best_mse) {
            best_mse     = train_mse;
            best_weights = weights;
            best_bias    = bias;
        }

        if (epoch % args.print_every == 0 || epoch == 1) {
            std::cout << "  Epoch " << epoch << "  MSE=" << train_mse << std::endl;
        }
    }

    // ── Test evaluation ─────────────────────────────────────────
    double test_mse = compute_mse(test, best_weights, best_bias);
    double test_mae = compute_mae(test, best_weights, best_bias);

    std::cout << "\n== Results =====================================" << std::endl;
    std::cout << "  Best train MSE: " << best_mse << std::endl;
    std::cout << "  Test MSE:       " << test_mse << std::endl;
    std::cout << "  Test MAE:       " << test_mae << std::endl;
    std::cout << "  Learned weights:";
    for (int j = 0; j < train.n_features; j++) {
        std::cout << " w" << j << "=" << best_weights[j];
    }
    std::cout << " bias=" << best_bias << std::endl;
    std::cout << "  True weights:    w0=3.0 w1=1.5 w2=-2.0 bias=7.0" << std::endl;

    // ── Save model weights as JSON ──────────────────────────────
    std::ofstream model_file(args.output_dir + "/model_weights.json");
    model_file << "{\n";
    model_file << "  \"weights\": [";
    for (int j = 0; j < train.n_features; j++) {
        if (j > 0) model_file << ", ";
        model_file << best_weights[j];
    }
    model_file << "],\n";
    model_file << "  \"bias\": " << best_bias << ",\n";
    model_file << "  \"n_features\": " << train.n_features << "\n";
    model_file << "}\n";
    model_file.close();

    // ── Save metrics for downstream pipeline steps ──────────────
    std::ofstream mae_file(args.output_dir + "/test_mae.txt");
    mae_file << test_mae;
    mae_file.close();

    std::ofstream mse_file(args.output_dir + "/metrics.txt");
    mse_file << best_mse;
    mse_file.close();

    std::cout << "\nDone — model saved to " << args.output_dir << std::endl;

    return 0;
}

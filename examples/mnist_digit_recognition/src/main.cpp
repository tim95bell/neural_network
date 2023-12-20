
#define TIM_NEURAL_NETWORK_HPP_IMPLEMENTATION 1
#define TIM_LINEAR_ALGEBRA_HPP_IMPLEMENTATION 1
#include <tnn/tnn.hpp>
#include <cstdio>
#include <vector>

static const tnn::U32 input_size = 784;
static const tnn::U32 output_size = 10;
static const tnn::U32 data_count = 42000;
static const tnn::U32 train_data_count = data_count * 0.7f;
static const tnn::U32 test_data_count = data_count - train_data_count;
static const float rate = 1e-1 * 3;
static const tnn::U32 batch_count = 420;
static const tnn::U32 epoch_count = 100;

tnn::TrainingDatum<input_size, output_size>* parse_data() {
    std::ifstream in_file("resources/train_random.csv");
    assert(in_file.is_open());
    std::string line;
    std::getline(in_file, line);
    tnn::TrainingDatum<input_size, output_size>* result = static_cast<tnn::TrainingDatum<input_size, output_size>*>(malloc(sizeof(tnn::TrainingDatum<input_size, output_size>) * data_count));
    tnn::U32 l;
    for (l = 0; l < data_count && in_file.good(); ++l) {
        auto input = tnn::get_input(result[l]);
        auto output = tnn::get_output(result[l]);
        std::getline(in_file, line);
        std::size_t line_index = 0;
        std::size_t new_line_index = 0;
        new_line_index = line.find_first_of(',', line_index);
        if (new_line_index == std::string::npos) {
            assert(false);
            return nullptr;
        }
        std::string x = line.substr(line_index, new_line_index - line_index);
        int x_int = atoi(x.c_str());
        if (x_int < 0 || x_int > 9) {
            assert(false);
            return nullptr;
        }
        tla::fill(output, 0.0f);
        output(x_int) = 1.0f;
        line_index = new_line_index + 1;

        for (tnn::U32 i = 0; i < 783; ++i) {
            new_line_index = line.find_first_of(',', line_index);
            if (new_line_index == std::string::npos) {
                return nullptr;
            }
            x = line.substr(line_index, new_line_index - line_index);
            x_int = atoi(x.c_str());
            assert(x_int >= 0 && x_int <= 255);
            input(i) = (float)x_int / 255.0f;
            line_index = new_line_index + 1;
        }

        if (line.find_first_of(',', line_index) != std::string::npos) {
            assert(false);
            return nullptr;
        }

        x = line.substr(line_index);
        x_int = atoi(x.c_str());
        assert(x_int >= 0 && x_int < 255);
        input(783) = (float)x_int / 255.0f;
    }

    assert(l == data_count);
    return result;
}

template <tnn::U32... NetworkShape>
void test(tnn::Network<NetworkShape...>& n, tnn::TrainingDatum<input_size, output_size>* data) {
    using N = tnn::Network<NetworkShape...>;
    static_assert(input_size == N::input_layer_count());
    static_assert(output_size == N::output_layer_count());
    tnn::U32 correct = 0;
    std::vector<tnn::U32> incorrect_indices;
    float average_cost = 0.0f;
    auto a = tnn::get_a<N::count()>(n);
    for (tnn::U32 i = train_data_count; i < data_count; ++i) {
        auto input = tnn::get_input(data[i]);
        auto output = tnn::get_output(data[i]);
        tnn::forward(n, input);
        tnn::U32 largest_index = 0;
        bool largest_index_is_unique = true;
        for (tnn::U32 j = 1; j < N::output_layer_count(); ++j) {
            if (a(j) > a(largest_index)) {
                largest_index = j;
                largest_index_is_unique = true;
            } else if (a(j) == a(largest_index)) {
                largest_index_is_unique = false;
            }
        }

        if (largest_index_is_unique && output(largest_index) == 1.0f){
            ++correct;
        } else {
            incorrect_indices.push_back(i);
        }

        for (tnn::U32 j = 0; j < N::output_layer_count(); ++j) {
            const float d = output(largest_index) - a(j);
            average_cost += d * d;
        }
    }
    average_cost /= static_cast<float>(test_data_count * N::output_layer_count());

    printf("correct: %lu, incorrect: %lu, correct percentage: %f\n", correct, incorrect_indices.size(), static_cast<float>(correct) / static_cast<float>(test_data_count));
    printf("average cost: %f\n", average_cost);
    if (incorrect_indices.size() > 0) {
        printf("incorrect indices:\n");
        for (tnn::U32 i = 0; i < incorrect_indices.size(); ++i) {
            printf("\t%lu\n", incorrect_indices[i]);
        }
    }
}

template <tnn::U32... NetworkShape>
void train(tnn::Network<NetworkShape...>& n, tnn::TrainingDatum<input_size, output_size>* data) {
    using N = tnn::Network<NetworkShape...>;
    static_assert(input_size == N::input_layer_count());
    static_assert(output_size == N::output_layer_count());
    printf("cost: %f\n", tnn::average_cost<train_data_count>(n, data));
    for (tnn::U32 epoch = 0; epoch < epoch_count; ++epoch) {
        tnn::shuffle_array(data, train_data_count);
        for (tnn::U32 i = 0; i < train_data_count; i += batch_count) {
            tnn::train_gradient_descent<batch_count>(n, data + i, rate);
        }
        printf("cost: %f\n", tnn::average_cost<train_data_count>(n, data));
    }
    printf("training finished\n");
}

void write_training_data(tnn::TrainingDatum<input_size, output_size>* training_data) {
    std::ofstream out_file("training_data_out.csv");
    out_file << std::endl;
    for (tnn::U32 i = 0; i < data_count; ++i) {
        tnn::U32 expected = 0;
        auto input = tnn::get_input(training_data[i]);
        auto output = tnn::get_output(training_data[i]);
        while (output(expected) == 0.0f) {
            ++expected;
        }
        out_file << expected << ",";
        for (tnn::U32 j = 0; j < input_size; ++j) {
            out_file << (input(j) * 255);
            if (j != input_size - 1) {
                out_file << ",";
            }
        }
        out_file << std::endl;
    }
}

int main() {
    assert(train_data_count % batch_count == 0);

    srand(10);
    tnn::Network<input_size, 16, 16, output_size> n;
    n.fill_random(-0.5f, 0.5f);
    tnn::TrainingDatum<input_size, output_size>* data = parse_data();
    tnn::shuffle_array(data, data_count);

    //tnn::read_network_from_file(n, "./examples/mnist_digit_recognition/network.csv");

    train(n, data);

    tnn::write_network_to_file(n, "./examples/mnist_digit_recognition/network.csv");

    test(n, data);

    return 0;
}

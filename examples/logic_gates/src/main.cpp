
#define TIM_NEURAL_NETWORK_HPP_IMPLEMENTATION 1
#define TIM_LINEAR_ALGEBRA_HPP_IMPLEMENTATION 1
#include <tnn/tnn.hpp>

static constexpr tnn::U32 input_count = 2;
static constexpr tnn::U32 output_count = 1;

static const float rate = 1e-1;
static const float train_count = 1000 * 100;

static tnn::TrainingDatum<input_count, output_count> training_data[4] {
#if 1
    // xor
    {{0.0f, 0.0f}, {0.0f}},
    {{0.0f, 1.0f}, {1.0f}},
    {{1.0f, 0.0f}, {1.0f}},
    {{1.0f, 1.0f}, {0.0f}}
#endif
#if 0
    // or
    {{0.0f, 0.0f}, {0.0f}},
    {{0.0f, 1.0f}, {1.0f}},
    {{1.0f, 0.0f}, {1.0f}},
    {{1.0f, 1.0f}, {1.0f}}
#endif
#if 0
    // and
    {{0.0f, 0.0f}, {0.0f}},
    {{0.0f, 1.0f}, {0.0f}},
    {{1.0f, 0.0f}, {0.0f}},
    {{1.0f, 1.0f}, {1.0f}}
#endif
};

static const tnn::U32 training_data_count = (sizeof(training_data) / sizeof(training_data[0]));

int main() {
    srand(10);
    tnn::Network<2, 2, 1> n;
    n.fill_random(-0.5f, 0.5f);
    using N = decltype(n);

    printf("cost: %f\n", tnn::average_cost<training_data_count>(n, training_data));

    for (tnn::U32 i = 0; i < train_count; ++i) {
        tnn::train_gradient_descent<training_data_count>(n, training_data, rate);
        printf("cost: %f\n", tnn::average_cost<training_data_count>(n, training_data));
    }

    for (int i = 0; i < training_data_count; ++i) {
        auto input = tnn::get_input(training_data[i]);
        tnn::forward(n, input);
        printf("%f , %f => %f\n", input(0), input(1), tnn::get_a<N::count()>(n)(0));
    }

    tnn::write_network_to_file(n, "./examples/logic_gates/network.csv");

    return 0;
}

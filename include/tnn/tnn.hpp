
#ifndef INCLUDE_TIM_NEURAL_NETWORK_HPP
#define INCLUDE_TIM_NEURAL_NETWORK_HPP

#include <tla/tla.hpp>
#include <fstream>

namespace tnn {
    using U8 = tla::U8;
    using U32 = tla::U32;
    using U64 = tla::U64;

    template <U32 InputLayerCount, U32... RestLayerCount>
    struct Network;

    template <tla::U32 InputCount, tla::U32 OutputCount>
    struct TrainingDatum {
        float input[InputCount];
        float output[OutputCount];
    };

    template <U32 InputSize, U32 OutputSize>
    void shuffle_array(TrainingDatum<InputSize, OutputSize>* data, U32 data_count);

    template <U32 A>
    consteval U32 largest();

    template <U32 A, U32 B, U32... Args> 
    consteval U32 largest();

    template <U32 I, U32 A, U32... Args> 
    consteval U32 nth_value();

    template <U32 A, U32... Args> 
    consteval U32 nth_value(U32 i);

    template <U32 Arg1>
    consteval U32 sum_of_product_of_pairs();

    template <U32 Arg1, U32 Arg2, U32... Args>
    consteval U32 sum_of_product_of_pairs();

    template <U32 N, U32 Arg1>
    consteval U32 sum_of_product_of_pairs_until();

    template <U32 N, U32 Arg1, U32 Arg2, U32... Args>
    consteval U32 sum_of_product_of_pairs_until();

    template <U32 Arg1>
    consteval U32 sum();

    template <U32 Arg1, U32 Arg2, U32... Args>
    consteval U32 sum();

    template <U32 N, U32 Arg1>
    consteval U32 sum_until();

    template <U32 N, U32 Arg1, U32 Arg2, U32... Args>
    consteval U32 sum_until();

    template <U32 Arg1, U32 Arg2, U32... Args>
    consteval U32 sum_until(U32 n);

    template <U32 L, U32...NetworkShape>
    inline float* w_buffer(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline auto get_w(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline float* dw_buffer(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline auto get_dw(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline float* b_buffer(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline auto get_b(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline float* db_buffer(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline auto get_db(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline float* a_buffer(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline auto get_a(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline float* z_buffer(Network<NetworkShape...>& n);

    template <U32 L, U32...NetworkShape>
    inline auto get_z(Network<NetworkShape...>& n);

    template <U32... NetworkShape>
    void write_network_to_file(Network<NetworkShape...>& n, const char* name);

    template <U32 N, U32... NetworkShape>
    void write_network_to_file(Network<NetworkShape...>& n, std::ofstream& out_file);

    template <U32... NetworkShape>
    void read_network_from_file(Network<NetworkShape...>& n, const char* name);

    template <U32 L, U32... NetworkShape>
    void read_network_from_file(Network<NetworkShape...>& n, std::ifstream& in_file);

    template <U32... NetworkShape>
    void forward(Network<NetworkShape...>& n, tla::Matrix<float, 1, Network<NetworkShape...>::input_layer_count()>& input);

    template <U32 L, U32... NetworkShape>
    void forward(Network<NetworkShape...>& n);

    template <tla::U32 InputCount, tla::U32 OutputCount>
    tla::Matrix<float, 1, InputCount> get_input(TrainingDatum<InputCount, OutputCount>& x);

    template <tla::U32 InputCount, tla::U32 OutputCount>
    tla::Matrix<float, 1, OutputCount> get_output(TrainingDatum<InputCount, OutputCount>& x);

    template <tla::U32 DataCount, U32... NetworkShape>
    float average_cost(Network<NetworkShape...>& n, TrainingDatum<Network<NetworkShape...>::input_layer_count(), Network<NetworkShape...>::output_layer_count()>* data);

    template <U32... NetworkShape>
    void clear_delta(Network<NetworkShape...>& n);

    template <U32 DataCount, U32... NetworkShape>
    void apply_deltas(Network<NetworkShape...>& n, float rate);

    template <U32 L, U32 DataCount, U32... NetworkShape>
    void apply_deltas(Network<NetworkShape...>& n, float rate);

    template <U32 DataCount, U32... NetworkShape>
    void train_gradient_descent(Network<NetworkShape...>& n, TrainingDatum<Network<NetworkShape...>::input_layer_count(), Network<NetworkShape...>::output_layer_count()>* data, float rate);

    template <U32 L, U32... NetworkShape>
    void train_gradient_descent_internal(Network<NetworkShape...>& n, tla::Matrix<float, 1, Network<NetworkShape...>::largest_layer_count()>& dc_da_l, tla::Matrix<float, 1, Network<NetworkShape...>::largest_layer_count()>& dc_da_m);

    // #region template implementation
    template <U32 InputLayerCount, U32... RestLayerCount>
    struct Network {
        static inline consteval U32 input_layer_count() {
            return layer_count<0>();
        }

        static inline consteval U32 output_layer_count() {
            return layer_count<count()>();
        }

        static inline consteval U32 largest_layer_count() {
            return largest<RestLayerCount...>();
        }

        static inline consteval U32 count() {
            return sizeof...(RestLayerCount);
        }

        static inline consteval U32 w_count() {
            return sum_of_product_of_pairs<InputLayerCount, RestLayerCount...>();
        }

        static inline consteval U32 b_count() {
            return sum<RestLayerCount...>();
        }

        static inline consteval U32 a_count() {
            return sum<InputLayerCount, RestLayerCount...>();
        }

        static inline consteval U32 z_count() {
            return sum<RestLayerCount...>();
        }

        void fill_random(float from, float to) {
            fill_random<1>(from, to);
        }

        template <U32 L>
        static inline consteval U32 layer_count() {
            static_assert(L >= 0 && L <= count());
            return nth_value<L, InputLayerCount, RestLayerCount...>();
        }

        template <U32 L>
        static inline consteval U32 w_rows() {
            static_assert(L > 0 && L <= count());
            return layer_count<L - 1>();
        }

        template <U32 L>
        static inline consteval U32 w_cols() {
            static_assert(L > 0 && L <= count());
            return layer_count<L>();
        }

        template <U32 L>
        inline float* w_buffer() {
            static_assert(L > 0 && L <= count());
            const U32 i = sum_of_product_of_pairs_until<L - 1, InputLayerCount, RestLayerCount...>();
            return &m_w[sum_of_product_of_pairs_until<L - 1, InputLayerCount, RestLayerCount...>()];
        }

        template <U32 L>
        inline auto w() {
            return tla::Matrix<float, w_rows<L>(), w_cols<L>()>(w_buffer<L>());
        }

        template <U32 L>
        inline float* dw_buffer() {
            static_assert(L > 0 && L <= count());
            return &m_dw[sum_of_product_of_pairs_until<L - 1, InputLayerCount, RestLayerCount...>()];
        }

        template <U32 L>
        inline auto dw() {
            return tla::Matrix<float, w_rows<L>(), w_cols<L>()>(dw_buffer<L>());
        }

        template <U32 L>
        inline float* b_buffer() {
            static_assert(L > 0 && L <= count());
            return &m_b[sum_until<L - 1, RestLayerCount...>()];
        }

        template <U32 L>
        inline auto b() {
            return tla::Matrix<float, 1, layer_count<L>()>(b_buffer<L>());
        }

        template <U32 L>
        inline float* a_buffer() {
            static_assert(L >= 0 && L <= count());
            return &m_a[sum_until<L, InputLayerCount, RestLayerCount...>()];
        }

        template <U32 L>
        inline auto a() {
            return tla::Matrix<float, 1, layer_count<L>()>(a_buffer<L>());
        }

        template <U32 L>
        inline float* z_buffer() {
            static_assert(L > 0 && L <= count());
            return &m_z[sum_until<L - 1, RestLayerCount...>()];
        }

        template <U32 L>
        inline auto z() {
            return tla::Matrix<float, 1, layer_count<L>()>(z_buffer<L>());
        }

        template <U32 L>
        inline float* db_buffer() {
            static_assert(L > 0 && L <= count());
            return &m_db[sum_until<L - 1, RestLayerCount...>()];
        }

        template <U32 L>
        inline auto db() {
            return tla::Matrix<float, 1, layer_count<L>()>(db_buffer<L>());
        }

        float m_w[w_count()];
        float m_b[b_count()];
        float m_a[a_count()];
        float m_z[z_count()];
        float m_dw[w_count()];
        float m_db[b_count()];

    private:
        template <U32 L>
        void fill_random(float from, float to) {
            static_assert(L > 0 && L <= count());
            auto w_m = w<L>();
            auto b_m = b<L>();
            tla::fill_random(w_m, from, to);
            tla::fill_random(b_m, from, to);
            if constexpr (L < count()) {
                fill_random<L + 1>(from, to);
            }
        }
    };

    template <U32 InputSize, U32 OutputSize>
    void shuffle_array(TrainingDatum<InputSize, OutputSize>* data, U32 data_count) {
        for (U32 i = 0; i < data_count; ++i) {
            U32 j = (rand() / static_cast<float>(RAND_MAX)) * (data_count - i) + i;
            if (i != j) {
                TrainingDatum<InputSize, OutputSize> tmp = data[i];
                data[i] = data[j];
                data[j] = tmp;
            }
        }
    }

    template <U32 A>
    inline consteval U32 largest()
    {
        return A;
    }

    template <U32 A, U32 B, U32... Args> 
    inline consteval U32 largest()
    {
        return largest<(A > B ? A : B), Args...>();
    }

    template <U32 I, U32 A, U32... Args> 
    inline consteval U32 nth_value() {
        if constexpr (I == 0) {
            return A;
        } else {
            return nth_value<I - 1, Args...>();
        }
    }

    template <U32 A, U32... Args> 
    inline consteval U32 nth_value(U32 i) {
        if (i == 0) {
            return A;
        } else {
            return nth_value<Args...>(i - 1);
        }
    }

    template <U32 Arg1>
    inline consteval U32 sum_of_product_of_pairs() {
        return 0;
    }

    template <U32 Arg1, U32 Arg2, U32... Args>
    inline consteval U32 sum_of_product_of_pairs() {
        return Arg1 * Arg2 + sum_of_product_of_pairs<Arg2, Args...>();
    }

    template <U32 N, U32 Arg1>
    inline consteval U32 sum_of_product_of_pairs_until() {
        return 0;
    }

    template <U32 N, U32 Arg1, U32 Arg2, U32... Args>
    inline consteval U32 sum_of_product_of_pairs_until() {
        if constexpr (N == 0) {
            return 0;
        }

        return Arg1 * Arg2 + sum_of_product_of_pairs_until<N - 1, Arg2, Args...>();
    }

    template <U32 Arg1>
    inline consteval U32 sum() {
        return Arg1;
    }

    template <U32 Arg1, U32 Arg2, U32... Args>
    inline consteval U32 sum() {
        return Arg1 + sum<Arg2, Args...>();
    }

    template <U32 N, U32 Arg1>
    inline consteval U32 sum_until() {
        if constexpr (N == 0) {
            return 0;
        }
        return Arg1;
    }

    template <U32 N, U32 Arg1, U32 Arg2, U32... Args>
    inline consteval U32 sum_until() {
        if constexpr (N == 0) {
            return 0;
        }

        return Arg1 + sum_until<N - 1, Arg2, Args...>();
    }

    template <U32 Arg1, U32 Arg2, U32... Args>
    inline consteval U32 sum_until(U32 n) {
        if (n == 0) {
            return 0;
        }

        return Arg1 + sum_until<Arg2, Args...>(n - 1);
    }

    template <U32 L, U32...NetworkShape>
    inline float* w_buffer(Network<NetworkShape...>& n) {
        return n.template w_buffer<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline auto get_w(Network<NetworkShape...>& n) {
        return n.template w<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline float* dw_buffer(Network<NetworkShape...>& n) {
        return n.template dw_buffer<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline auto get_dw(Network<NetworkShape...>& n) {
        return n.template dw<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline float* b_buffer(Network<NetworkShape...>& n) {
        return n.template b_buffer<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline auto get_b(Network<NetworkShape...>& n) {
        return n.template b<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline float* db_buffer(Network<NetworkShape...>& n) {
        return n.template db_buffer<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline auto get_db(Network<NetworkShape...>& n) {
        return n.template db<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline float* a_buffer(Network<NetworkShape...>& n) {
        return n.template a_buffer<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline auto get_a(Network<NetworkShape...>& n) {
        return n.template a<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline float* z_buffer(Network<NetworkShape...>& n) {
        return n.template z_buffer<L>();
    }

    template <U32 L, U32...NetworkShape>
    inline auto get_z(Network<NetworkShape...>& n) {
        return n.template z<L>();
    }

    template <U32... NetworkShape>
    void write_network_to_file(Network<NetworkShape...>& n, const char* name) {
        using N = Network<NetworkShape...>;
        std::ofstream out_file(name);
        write_network_to_file<1>(n, out_file);
        out_file.close();
    }

    template <U32 L, U32... NetworkShape>
    void write_network_to_file(Network<NetworkShape...>& n, std::ofstream& out_file) {
        using N = Network<NetworkShape...>;
        static_assert(L > 0 && L <= N::count());
        auto w = get_w<L>(n);
        auto b = get_b<L>(n);
        for (U32 i = 0; i < N::template layer_count<L - 1>(); ++i) {
            for (U32 j = 0; j < N::template layer_count<L>(); ++j) {
                out_file << w(i, j);
                if (!(i == N::template layer_count<L - 1>() - 1 && j == N::template layer_count<L>() - 1)) {
                    out_file << ",";
                }
            }
        }
        out_file << std::endl;
        for (U32 i = 0; i < N::template layer_count<L>() - 1; ++i) {
            out_file << b(i) << ",";
        }
        out_file << b(N::template layer_count<L>() - 1) << std::endl;

        if constexpr (L < N::count()) {
            write_network_to_file<L + 1>(n, out_file);
        }
    }

    template <U32... NetworkShape>
    void read_network_from_file(Network<NetworkShape...>& n, const char* name) {
        using N = Network<NetworkShape...>;
        std::ifstream in_file(name);
        assert(in_file.is_open());
        read_network_from_file<1>(n, in_file);
        in_file.close();
    }

    template <U32 L, U32... NetworkShape>
    void read_network_from_file(Network<NetworkShape...>& n, std::ifstream& in_file) {
        using N = Network<NetworkShape...>;
        static_assert(L > 0 && L <= N::count());
        std::string line;
        auto w = get_w<L>(n);
        auto b = get_b<L>(n);
        std::getline(in_file, line);
        std::size_t line_index = 0;
        std::size_t new_line_index = 0;
        for (U32 i = 0; i < N::template layer_count<L - 1>(); ++i) {
            for (U32 j = 0; j < N::template layer_count<L>(); ++j) {
                new_line_index = line.find_first_of(',', line_index);
                if (new_line_index == std::string::npos) {
                    assert(i == N::template layer_count<L - 1>() - 1 && j == N::template layer_count<L>() - 1);
                } else {
                    assert(!(i == N::template layer_count<L - 1>() - 1 && j == N::template layer_count<L>() - 1));
                }
                std::string x = line.substr(line_index, new_line_index - line_index);
                float x_float = atof(x.c_str());
                w(i, j) = x_float;
                line_index = new_line_index + 1;
            }
        }

        std::getline(in_file, line);
        line_index = 0;
        for (U32 i = 0; i < N::template layer_count<L>(); ++i) {
            new_line_index = line.find_first_of(',', line_index);
            if (new_line_index == std::string::npos) {
                assert(i == N::template layer_count<L>() - 1);
            } else {
                assert(i != N::template layer_count<L>() - 1);
            }
            std::string x = line.substr(line_index, new_line_index - line_index);
            float x_float = atof(x.c_str());
            b(i) = x_float;
            line_index = new_line_index + 1;
        }
        
        if constexpr (L < N::count()) {
            read_network_from_file<L + 1>(n, in_file);
        }
    }

    template <U32... NetworkShape>
    void forward(Network<NetworkShape...>& n, tla::Matrix<float, 1, Network<NetworkShape...>::input_layer_count()>& input) {
        using N = Network<NetworkShape...>;
        auto a = get_a<0>(n);
        a = input;
        forward<1>(n);
    }

    template <U32 L, U32... NetworkShape>
    void forward(Network<NetworkShape...>& n) {
        using N = Network<NetworkShape...>;
        static_assert(L > 0 && L <= N::count());
        auto z_l = get_z<L>(n);
        auto a_k = get_a<L - 1>(n);
        auto w_l = get_w<L>(n);
        auto b_l = get_b<L>(n);
        auto a_l = get_a<L>(n);
        tla::multiply_add(z_l, a_k, w_l, b_l);

        tla::assign_apply(a_l, z_l, tla::sigmoid);

        if constexpr (L < N::count()) {
            forward<L + 1>(n);
        }
    }

    template <tla::U32 InputCount, tla::U32 OutputCount>
    tla::Matrix<float, 1, InputCount> get_input(TrainingDatum<InputCount, OutputCount>& x) {
        return tla::Matrix<float, 1, InputCount>(x.input);
    }

    template <tla::U32 InputCount, tla::U32 OutputCount>
    tla::Matrix<float, 1, OutputCount> get_output(TrainingDatum<InputCount, OutputCount>& x) {
        return tla::Matrix<float, 1, OutputCount>(x.output);
    }

    template <tla::U32 DataCount, U32... NetworkShape>
    float average_cost(Network<NetworkShape...>& n, TrainingDatum<Network<NetworkShape...>::input_layer_count(), Network<NetworkShape...>::output_layer_count()>* data) {
        using N = Network<NetworkShape...>;
        float result = 0.0f;
        for (U32 i = 0; i < DataCount; ++i) {
            auto input = get_input(data[i]);
            forward(n, input);
            for (U32 j = 0; j < N::output_layer_count(); ++j) {
                const float d = get_a<N::count()>(n)(j) - data[i].output[j];
                result += d * d;
            }
        }
        return result / static_cast<float>(DataCount * N::output_layer_count());
    }

    template <U32... NetworkShape>
    void clear_delta(Network<NetworkShape...>& n) {
        using N = Network<NetworkShape...>;
        memset(dw_buffer<1>(n), 0.0f, (N::w_count() + N::b_count()) * sizeof(float));
    }

    template <U32 DataCount, U32... NetworkShape>
    void apply_deltas(Network<NetworkShape...>& n, float rate) {
        apply_deltas<1, DataCount>(n, rate);
    }

    template <U32 L, U32 DataCount, U32... NetworkShape>
    void apply_deltas(Network<NetworkShape...>& n, float rate) {
        using N = Network<NetworkShape...>;
        static_assert(L > 0 && L <= N::count());
        auto dw = get_dw<L>(n);
        auto db = get_db<L>(n);
        auto w = get_w<L>(n);
        auto b = get_b<L>(n);

        dw *= rate / static_cast<float>(DataCount);
        db *= rate / static_cast<float>(DataCount);

        w -= dw;
        b -= db;

        if constexpr (L < N::count()) {
            apply_deltas<L + 1, DataCount>(n, rate);
        }
    }

    template <U32 DataCount, U32... NetworkShape>
    void train_gradient_descent(Network<NetworkShape...>& n, TrainingDatum<Network<NetworkShape...>::input_layer_count(), Network<NetworkShape...>::output_layer_count()>* data, float rate) {
        using N = Network<NetworkShape...>;
        clear_delta(n);

        for (U32 t = 0; t < DataCount; ++t) {
            auto input = get_input(data[t]);
            auto output = get_output(data[t]);

            forward(n, input);
            auto a_l = get_a<N::count()>(n);
            auto z_l = get_z<N::count()>(n);
            auto db_l = get_db<N::count()>(n);
            auto dw_l = get_dw<N::count()>(n);
            auto a_k = get_a<N::count() - 1>(n);
            auto a_k_t = tla::swap_dimensions(a_k);

            float dc_da_l_buffer[N::largest_layer_count()];
            float dc_da_m_buffer[N::largest_layer_count()];
            tla::Matrix<float, 1, N::output_layer_count()> dc_da_l(&dc_da_l_buffer[0]);

            for (U32 j = 0; j < N::output_layer_count(); ++j) {
                dc_da_l(j) = 2 * (a_l(j) - output(j)) * tla::sigmoid_derivative(z_l(j));
            }

            db_l += dc_da_l;
            tla::multiply_accumulate(dw_l, a_k_t, dc_da_l);

            train_gradient_descent_internal<N::count() - 1>(n, dc_da_m_buffer, dc_da_l_buffer);
        }

        apply_deltas<DataCount>(n, rate);
    }

    template <U32 L, U32... NetworkShape>
    void train_gradient_descent_internal(Network<NetworkShape...>& n, float dc_da_l_buffer[Network<NetworkShape...>::largest_layer_count()], float dc_da_m_buffer[Network<NetworkShape...>::largest_layer_count()]) {
        using N = Network<NetworkShape...>;
        static_assert(L > 0 && L < N::count());
        tla::Matrix<float, 1, N::template layer_count<L>()> dc_da_l(&dc_da_l_buffer[0]);
        tla::Matrix<float, 1, N::template layer_count<L + 1>()> dc_da_m(&dc_da_m_buffer[0]);
        tla::Matrix<float, N::template layer_count<L + 1>(), 1> dc_da_m_t = tla::swap_dimensions(dc_da_m);
        tla::Matrix<float, N::template layer_count<L>(), 1> dc_da_l_t = tla::swap_dimensions(dc_da_l);
        auto w_m = get_w<L + 1>(n);
        auto z_l = get_z<L>(n);
        auto db_l = get_db<L>(n);
        auto dw_l = get_dw<L>(n);
        auto a_k = get_a<L - 1>(n);
        auto a_k_t = tla::swap_dimensions(a_k);

        tla::multiply(dc_da_l_t, w_m, dc_da_m_t);
        tla::elementwise_multiply_by_sigmoid_derivative_of(dc_da_l, z_l);

        db_l += dc_da_l;
        tla::multiply_accumulate(dw_l, a_k_t, dc_da_l);
        
        if constexpr (L > 1) {
            train_gradient_descent_internal<L - 1>(n, dc_da_m_buffer, dc_da_l_buffer);
        }
    }
    // #endregion
}
#endif // INCLUDE_TIM_NEURAL_NETWORK_HPP

#ifdef TIM_NEURAL_NETWORK_HPP_IMPLEMENTATION
namespace tnn {
}
#endif // TIM_NEURAL_NETWORK_HPP_IMPLEMENTATION

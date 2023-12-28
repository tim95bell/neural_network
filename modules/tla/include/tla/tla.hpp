
#ifndef INCLUDE_TIM_LINEAR_ALGEBRA_HPP
#define INCLUDE_TIM_LINEAR_ALGEBRA_HPP

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <utility>

#if USE_NEON
#include <arm_neon.h>
#endif

namespace tla {
    using U8 = unsigned char;
    using U32 = unsigned long;
    using U64 = unsigned long long;

    template <typename Type, U32 Rows, U32 Cols>
    struct Matrix;

    extern float lerp(float a, float b, float d);
    extern float sigmoid(float x);
    extern void do_sigmoid(float& x);
    extern float sigmoid_derivative(float x);
    extern float relu(float x);
    extern void do_relu(float& x);
    extern float relu_derivative(float x);
    extern float rand_float();
    extern float rand_float(float from, float to);

    template <typename Type>
    inline constexpr Type zero();

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Cols, Rows> swap_dimensions(Matrix<Type, Rows, Cols> x);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols>
    void transpose(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols>
    void transpose_col(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x, U32 c);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline void assert_equal_dimensions(const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline void copy_data(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 Rows, U32 Cols>
    inline void fill(Matrix<Type, Rows, Cols> x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    inline void clear(Matrix<Type, Rows, Cols> x);

    template <typename Type, U32 Rows, U32 Cols>
    void fill_random(Matrix<Type, Rows, Cols> x);

    template <typename Type, U32 Rows, U32 Cols>
    void fill_random(Matrix<Type, Rows, Cols> x, float from, float to);

    template <typename Type, U32 Rows, U32 Cols, typename Function>
    void apply(Matrix<Type, Rows, Cols> x, Function f);

    template <typename Type, U32 XRows, U32 XCols, U32 YRows, U32 YCols, typename Function>
    void assign_apply(Matrix<Type, XRows, XCols> x, const Matrix<Type, YRows, YCols> y, Function f);

#if USE_NEON
    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void add_neon(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);
#endif

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void add_basic(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void add(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

#if USE_NEON
    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void subtract_neon(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);
#endif

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void subtract_basic(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void subtract(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

#if USE_NEON
    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    float multiply_neon_1_for_row_and_col(const Matrix<Type, ARows, ACols> a_r, const Matrix<Type, BRows, BCols> b_c_t);
#endif

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_neon_1(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);
#endif

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_neon_2(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);
#endif

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_neon(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);
#endif

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_basic(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate_neon_1(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);
#endif

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate_neon_2(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);
#endif

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate_neon(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);
#endif

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate_basic(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols, U32 YRows, U32 YCols, U32 ZRows, U32 ZCols>
    void multiply_add_neon(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x, const Matrix<Type, YRows, YCols> y, const Matrix<Type, ZRows, ZCols> z);
#endif

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols, U32 YRows, U32 YCols, U32 ZRows, U32 ZCols>
    void multiply_add_basic(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x, const Matrix<Type, YRows, YCols> y, const Matrix<Type, ZRows, ZCols> z);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols, U32 YRows, U32 YCols, U32 ZRows, U32 ZCols>
    void multiply_add(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x, const Matrix<Type, YRows, YCols> y, const Matrix<Type, ZRows, ZCols> z);

#if USE_NEON
    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void elementwise_multiply_by_sigmoid_derivative_of_neon(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);
#endif

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void elementwise_multiply_by_sigmoid_derivative_of_basic(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void elementwise_multiply_by_sigmoid_derivative_of(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void divide(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

#if USE_NEON
    template <typename Type, U32 Rows, U32 Cols>
    void add_neon(const Matrix<Type, Rows, Cols> x, Type v);
#endif

    template <typename Type, U32 Rows, U32 Cols>
    void add_basic(const Matrix<Type, Rows, Cols> x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    void add(const Matrix<Type, Rows, Cols> x, Type v);

#if USE_NEON
    template <typename Type, U32 Rows, U32 Cols>
    void subtract_neon(const Matrix<Type, Rows, Cols> x, Type v);
#endif

    template <typename Type, U32 Rows, U32 Cols>
    void subtract_basic(const Matrix<Type, Rows, Cols> x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    void subtract(const Matrix<Type, Rows, Cols> x, Type v);

#if USE_NEON
    template <typename Type, U32 Rows, U32 Cols>
    void multiply_neon(const Matrix<Type, Rows, Cols> x, Type v);
#endif

    template <typename Type, U32 Rows, U32 Cols>
    void multiply_basic(const Matrix<Type, Rows, Cols> x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    void multiply(const Matrix<Type, Rows, Cols> x, Type v);

#if USE_NEON
    template <typename Type, U32 Rows, U32 Cols>
    void divide_neon(const Matrix<Type, Rows, Cols> x, Type v);
#endif

    template <typename Type, U32 Rows, U32 Cols>
    void divide_basic(const Matrix<Type, Rows, Cols> x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    void divide(const Matrix<Type, Rows, Cols> x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    void print(const Matrix<Type, Rows, Cols> x);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    bool operator==(const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline bool operator!=(const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator+=(Matrix<Type, Rows, Cols> x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator-=(Matrix<Type, Rows, Cols> x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator*=(Matrix<Type, Rows, Cols> x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator/=(Matrix<Type, Rows, Cols> x, Type v);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    Matrix<Type, ARows, ACols> operator+=(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    Matrix<Type, ARows, ACols> operator-=(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 Rows, U32 Cols>
    struct Matrix {
        using ValueType = Type;
        using ThisType = Matrix<Type, Rows, Cols>;

        Type* m_data;

        explicit Matrix(Type* in_data)
            : m_data(in_data)
        {}

        Matrix(const ThisType& other)
            : m_data(other.m_data)
        {}

        inline ThisType& operator=(const ThisType& other) {
            copy_data(*this, other);
            return *this;
        }
        
        inline const Type& operator()(U32 r, U32 c) const {
            assert(r < Rows && c < Cols);
            return m_data[r * Cols + c];
        }

        inline Type& operator()(U32 r, U32 c) {
            return const_cast<Type&>(static_cast<const ThisType*>(this)->operator()(r, c));
        }

        inline const Type& operator()(U32 i) const {
            static_assert(Rows == 1 || Cols == 1);
            if constexpr (Rows == 1) {
                assert(i < Cols);
            } else {
                assert(i < Rows);
            }
            return m_data[i];
        }

        inline Type& operator()(U32 i) {
            return const_cast<Type&>(static_cast<const ThisType*>(this)->operator()(i));
        }
    };

    template <>
    inline constexpr float zero<float>() {
        return 0.0f;
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Cols, Rows> swap_dimensions(Matrix<Type, Rows, Cols> x) {
        return Matrix<Type, Cols, Rows>(x.m_data);
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols>
    void transpose(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x) {
        static_assert(ResultRows == XCols);
        static_assert(ResultCols == XRows);

        for (U32 r = 0; r < XRows; ++r) {
            for (U32 c = 0; c < XCols; ++c) {
                result(c, r) = x(r, c);
            }
        }
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols>
    void transpose_col(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x, U32 c) {
        static_assert(ResultRows == XRows);
        static_assert(ResultCols == 1);

        for (U32 r = 0; r < XRows; ++r) {
            result(r) = x(r, c);
        }
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline void assert_equal_dimensions(const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        static_assert(ARows == BRows);
        static_assert(ACols == BCols);
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline void copy_data(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        static_assert(ARows * ACols == BRows * BCols);
        memcpy(a.m_data, b.m_data, ARows * ACols * sizeof(Type));
    }

    template <typename Type, U32 Rows, U32 Cols>
    inline void fill(Matrix<Type, Rows, Cols> x, Type v) {
        memset(x.m_data, v, Rows * Cols * sizeof(Type));
    }

    template <typename Type, U32 Rows, U32 Cols>
    inline void clear(Matrix<Type, Rows, Cols> x) {
        fill(x, zero<Type>());
    }

    template <typename Type, U32 Rows, U32 Cols>
    void fill_random(Matrix<Type, Rows, Cols> x) {
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                x(r, c) = rand_float();
            }
        }
    }

    template <typename Type, U32 Rows, U32 Cols>
    void fill_random(Matrix<Type, Rows, Cols> x, float from, float to) {
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                x(r, c) = rand_float(from, to);
            }
        }
    }

    template <typename Type, U32 Rows, U32 Cols, typename Function>
    void apply(Matrix<Type, Rows, Cols> x, Function f) {
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                f(x(r, c));
            }
        }
    }

    template <typename Type, U32 XRows, U32 XCols, U32 YRows, U32 YCols, typename Function>
    void assign_apply(Matrix<Type, XRows, XCols> x, const Matrix<Type, YRows, YCols> y, Function f) {
        assert_equal_dimensions(x, y);

        for (U32 r = 0; r < XRows; ++r) {
            for (U32 c = 0; c < XCols; ++c) {
                x(r, c) = f(y(r, c));
            }
        }
    }

#if USE_NEON
    template <U32 Count>
    void assign_apply_sigmoid_neon(Matrix<float, 1, Count> a_l, const Matrix<float, 1, Count> z_l) {
        static constexpr U32 count = Count / 4;
        static constexpr U32 leftover_start = count * 4;
        static constexpr U32 r = 0;
        float buffer[4];
        float32x4_t ones = vmovq_n_f32(1.0f);
        for (U32 c = 0; c < count; ++c) {
            for (U32 i = 0; i < 4; ++i) {
                buffer[i] = expf(-z_l(r, c * 4 + i));
            }

            float32x4_t x = vld1q_f32(&buffer[0]);
            x = vaddq_f32(ones, x);
#if 1
            float32x4_t x_inverse = vrecpeq_f32(x);
#else
            float32x4_t x_reciprocal = vrecpeq_f32(x);
            float32x4_t x_inverse = vmulq_f32(vrecpsq_f32(x, x_reciprocal), x_reciprocal);
#endif
            vst1q_f32(&a_l(r, c * 4), x_inverse);
        }
        for (U32 c = leftover_start; c < Count; ++c) {
            a_l(r, c) = sigmoid(z_l(r, c));
        }
    }
#endif

    template <U32 Count>
    void assign_apply_sigmoid_basic(Matrix<float, 1, Count> a_l, const Matrix<float, 1, Count> z_l) {
        assign_apply(a_l, z_l, sigmoid);
    }

    template <U32 Count>
    void assign_apply_sigmoid(Matrix<float, 1, Count> a_l, const Matrix<float, 1, Count> z_l) {
#if USE_NEON
        assign_apply_sigmoid_neon(a_l, z_l);
#else
        assign_apply_sigmoid_basic(a_l, z_l);
#endif
    }

#if USE_NEON
    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void add_neon(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        // TODO(TB): try doing this with a bigger batch size. maybe 4 at a time
        static constexpr U32 count = ACols / 4;
        static constexpr U32 leftover_start = count * 4;
        for (U32 r = 0; r < ARows; ++r) {
            for (U32 c = 0; c < count; ++c) {
                float32x4_t a_row = vld1q_f32(&a(r, c * 4));
                float32x4_t b_row = vld1q_f32(&b(r, c * 4));
                a_row = vaddq_f32(a_row, b_row);
                vst1q_f32(&a(r, c * 4), a_row);
            }
            for (U32 c = leftover_start; c < ACols; ++c) {
                a(r, c) += b(r, c);
            }
        }
    }
#endif

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void add_basic(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        for (U32 r = 0; r < ARows; ++r) {
            for (U32 c = 0; c < ACols; ++c) {
                a(r, c) += b(r, c);
            }
        }
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void add(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        assert_equal_dimensions(a, b);

#if USE_NEON
        add_neon(a, b);
#else
        add_basic(a, b);
#endif
    }

#if USE_NEON
    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void subtract_neon(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        // TODO(TB): try doing this with a bigger batch size. maybe 4 at a time
        static constexpr U32 count = ACols / 4;
        static constexpr U32 leftover_start = count * 4;
        for (U32 r = 0; r < ARows; ++r) {
            for (U32 c = 0; c < count; ++c) {
                float32x4_t a_row = vld1q_f32(&a(r, c * 4));
                float32x4_t b_row = vld1q_f32(&b(r, c * 4));
                a_row = vsubq_f32(a_row, b_row);
                vst1q_f32(&a(r, c * 4), a_row);
            }
            for (U32 c = leftover_start; c < ACols; ++c) {
                a(r, c) -= b(r, c);
            }
        }
    }
#endif

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void subtract_basic(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        for (U32 r = 0; r < ARows; ++r) {
            for (U32 c = 0; c < ACols; ++c) {
                a(r, c) -= b(r, c);
            }
        }
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void subtract(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        assert_equal_dimensions(a, b);

#if USE_NEON
        subtract_neon(a, b);
#else
        subtract_basic(a, b);
#endif
    }

#if USE_NEON
    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    float multiply_neon_1_for_row_and_col(const Matrix<Type, ARows, ACols> a_r, const Matrix<Type, BRows, BCols> b_c_t) {
        static constexpr U32 Inner = BRows;
        static constexpr U32 OuterCols = BCols;
        static constexpr U32 first_count = (Inner / 8);
        static constexpr bool first_leftover_vector = (Inner - (first_count * 8)) / 4 == 1;
        static constexpr U32 first_leftover_single = Inner - (first_count * 8)  - (first_leftover_vector ? 4 : 0);
        float32x4_t res[first_count];
        float result;
        if constexpr (first_count != 0) {
            for (U32 i = 0; i < first_count; ++i) {
                float32x4_t a_row_1 = vld1q_f32(&a_r(i * 4));
                float32x4_t b_row_1 = vld1q_f32(&b_c_t(i * 4));
                res[i] = vmulq_f32(a_row_1, b_row_1);
                float32x4_t a_row_2 = vld1q_f32(&a_r((i + first_count) * 4));
                float32x4_t b_row_2 = vld1q_f32(&b_c_t((i + first_count) * 4));
                float32x4_t t = vmulq_f32(a_row_2, b_row_2);
                res[i] = vaddq_f32(res[i], t);
            }
            if constexpr (first_leftover_vector) {
                float32x4_t a_row_2 = vld1q_f32(&a_r(first_count * 8));
                float32x4_t b_row_2 = vld1q_f32(&b_c_t(first_count * 8));
                float32x4_t t = vmulq_f32(a_row_2, b_row_2);
                res[0] = vaddq_f32(res[0], t);
            } else {
                assert((Inner - (first_count * 8)) / 4 == 0);
            }
            bool leftover = first_count % 2 == 1;
            U32 count = first_count / 2;
            while (count > 0) {
                for (U32 i = 0; i < count; ++i) {
                    res[i] = vaddq_f32(res[i], res[count + i]);
                }
                if (leftover) {
                    res[0] = vaddq_f32(res[0], res[count * 2]);
                }
                leftover = count % 2 == 1;
                count /= 2;
            }
            result = vdups_laneq_f32(res[0], 0) + vdups_laneq_f32(res[0], 1) + vdups_laneq_f32(res[0], 2) + vdups_laneq_f32(res[0], 3);
        } else {
            if constexpr (first_leftover_vector) {
                float32x4_t a_row_2 = vld1q_f32(&a_r(0));
                float32x4_t b_row_2 = vld1q_f32(&b_c_t(0));
                res[0] = vmulq_f32(a_row_2, b_row_2);
                result = vdups_laneq_f32(res[0], 0) + vdups_laneq_f32(res[0], 1) + vdups_laneq_f32(res[0], 2) + vdups_laneq_f32(res[0], 3);
            } else {
                result = 0;
            }
        }
        if constexpr (first_leftover_single) {
            for (U32 i = Inner - first_leftover_single; i < Inner; ++i) {
                result += a_r(i) * b_c_t(i);
            }
        }
        return result;
    }
#endif

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_neon_1(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        static constexpr U8 OuterRows = ARows;
        static constexpr U8 Inner = ACols;
        static constexpr U8 OuterCols = BCols;
        static constexpr U32 first_count = (Inner / 8);
        static constexpr bool first_leftover_vector = (Inner - (first_count * 8)) / 4 == 1;
        static constexpr U32 first_leftover_single = Inner - (first_count * 8)  - (first_leftover_vector ? 4 : 0);
        float32x4_t res[first_count];

        float* b_t_buffer = static_cast<float*>(malloc(BRows * sizeof(Type)));
        for (U32 c = 0; c < OuterCols; ++c) {
            if constexpr (OuterCols == 1) {
                for (U32 r = 0; r < OuterRows; ++r) {
                    // TODO(TB): clean this up, make a function to get a row or col
                    Matrix<Type, 1, ACols> a_r(a.m_data + r * ACols);
                    result(r, c) = multiply_neon_1_for_row_and_col(a_r, b);
                }
            } else {
                Matrix<Type, 0, 0> b_t(b_t_buffer, 1, BCols);
                transpose_col(b_t, b, c);

                for (U32 r = 0; r < OuterRows; ++r) {
                    Matrix<Type, 0, 0> a_r(a.m_data + r * ACols, 1, ACols);
                    result(r, c) = multiply_neon_1_for_row_and_col(a_r, b_t);
                }
            }
        }

        free(b_t_buffer);
    }
#endif

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_neon_2(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        static constexpr U32 batch_size = 4;
        static constexpr U32 OuterRows = ARows;
        static constexpr U32 Inner = ACols;
        static constexpr U32 OuterCols = BCols;
        static constexpr U32 vector_size = 4;
        // number of batches
        static constexpr U32 batch_count = (OuterCols / vector_size) / batch_size;
        // number of vectors leftover (for an incomplete batch)
        static constexpr U32 leftover_vectors = (OuterCols / vector_size) - (batch_count * batch_size);
        static constexpr U32 leftover_singles = OuterCols - (batch_count * batch_size * vector_size) - (leftover_vectors * vector_size);

        float32x4x4_t res;
        for (U32 a_r = 0; a_r < OuterRows; ++a_r) {
            for (U32 batch_i = 0; batch_i < batch_count; ++batch_i) {
                // perform first multiply (for 0th column of a) first, as no addition is required
                // a[a_r][0] * b[0]
                for (U32 b_c = 0; b_c < batch_size; ++b_c) {
                    res.val[b_c] = vld1q_f32(&b(0, ((batch_i * batch_size) + b_c) * vector_size));
                    res.val[b_c] = vmulq_n_f32(res.val[b_c], a(a_r, 0));
                }
                for (U32 b_r = 1; b_r < Inner; ++b_r) {
                    for (U32 b_c = 0; b_c < batch_size; ++b_c) {
                        float32x4_t x = vld1q_f32(&b(b_r, ((batch_i * batch_size) + b_c) * vector_size));
                        x = vmulq_n_f32(x, a(a_r, b_r));
                        res.val[b_c] = vaddq_f32(res.val[b_c], x);
                    }
                }
                vst1q_f32_x4(&result(a_r, batch_i * batch_size * vector_size), res);
            }
            if constexpr (leftover_vectors > 0) {
                const U32 start = batch_count * batch_size * vector_size;
                for (U32 b_c = 0; b_c < leftover_vectors; ++b_c) {
                    res.val[b_c] = vld1q_f32(&b(0, start + (b_c * vector_size)));
                    res.val[b_c] = vmulq_n_f32(res.val[b_c], a(a_r, 0));
                }
                for (U32 b_r = 1; b_r < Inner; ++b_r) {
                    for (U32 b_c = 0; b_c < leftover_vectors; ++b_c) {
                        float32x4_t x = vld1q_f32(&b(b_r, start + (b_c * vector_size)));
                        x = vmulq_n_f32(x, a(a_r, b_r));
                        res.val[b_c] = vaddq_f32(res.val[b_c], x);
                    }
                }
                // TODO(TB): how to avoid this
                if constexpr (leftover_vectors == 1) {
                    vst1q_f32(&result(a_r, start), res.val[0]);
                } else if constexpr (leftover_vectors == 2) {
                    vst1q_f32(&result(a_r, start), res.val[0]);
                    vst1q_f32(&result(a_r, start + 4), res.val[1]);
                } else if constexpr (leftover_vectors == 3) {
                    vst1q_f32(&result(a_r, start), res.val[0]);
                    vst1q_f32(&result(a_r, start + 4), res.val[1]);
                    vst1q_f32(&result(a_r, start + 8), res.val[2]);
                }
            }
            if constexpr (leftover_singles > 0) {
                const U32 start = batch_count * batch_size * vector_size + leftover_vectors * vector_size;
                for (U32 b_c = start; b_c < OuterCols; ++b_c) {
#if 0
                    float b_t[Inner];
                    transpose_col<Inner, OuterCols>(b_t, b, b_c);
                    result[a_r][b_c] = multiply_3_for_row_and_col<Inner, OuterCols>(a[a_r], b_t);
#else
                    result(a_r, b_c) = 0.0f;
                    for (U32 i = 0; i < ACols; ++i) {
                        result(a_r, b_c) += a(a_r, i) * b(i, b_c);
                    }
#endif
                }
            }
        }
    }
#endif

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_neon(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        if constexpr (BCols == 1) {
            multiply_neon_1(result, a, b);
        } else {
            multiply_neon_2(result, a, b);
        }
    }
#endif

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_basic(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        for (U32 r = 0; r < ResultRows; ++r) {
            for (U32 c = 0; c < ResultCols; ++c) {
                result(r, c) = zero<Type>();
                for (U32 i = 0; i < ACols; ++i) {
                    result(r, c) += a(r, i) * b(i, c);
                }
            }
        }
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        // TODO(TB): assert result is not pointing to same data as a or b?
        static_assert(ACols == BRows);
        static_assert(ResultRows == ARows);
        static_assert(ResultCols == BCols);

#if USE_NEON
        multiply_neon(result, a, b);
#else
        multiply_basic(result, a, b);
#endif
    }

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate_neon_1(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        static constexpr U8 OuterRows = ARows;
        static constexpr U8 Inner = ACols;
        static constexpr U8 OuterCols = BCols;
        static constexpr U32 first_count = (Inner / 8);
        static constexpr bool first_leftover_vector = (Inner - (first_count * 8)) / 4 == 1;
        static constexpr U32 first_leftover_single = Inner - (first_count * 8)  - (first_leftover_vector ? 4 : 0);
        float32x4_t res[first_count];

        float* b_t_buffer = static_cast<float*>(malloc(BRows * sizeof(Type)));
        for (U32 c = 0; c < OuterCols; ++c) {
            if constexpr (OuterCols == 1) {
                for (U32 r = 0; r < OuterRows; ++r) {
                    // TODO(TB): clean this up, make a function to get a row or col
                    Matrix<Type, 1, ACols> a_r(a.m_data + r * ACols);
                    result(r, c) += multiply_neon_1_for_row_and_col(a_r, b);
                }
            } else {
                Matrix<Type, 0, 0> b_t(b_t_buffer, 1, BCols);
                transpose_col(b_t, b, c);

                for (U32 r = 0; r < OuterRows; ++r) {
                    Matrix<Type, 0, 0> a_r(a.m_data + r * ACols, 1, ACols);
                    result(r, c) += multiply_neon_1_for_row_and_col(a_r, b_t);
                }
            }
        }

        free(b_t_buffer);
    }
#endif

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate_neon_2(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        static constexpr U32 batch_size = 4;
        static constexpr U32 OuterRows = ARows;
        static constexpr U32 Inner = ACols;
        static constexpr U32 OuterCols = BCols;
        static constexpr U32 vector_size = 4;
        // number of batches
        static constexpr U32 batch_count = (OuterCols / vector_size) / batch_size;
        // number of vectors leftover (for an incomplete batch)
        static constexpr U32 leftover_vectors = (OuterCols / vector_size) - (batch_count * batch_size);
        static constexpr U32 leftover_singles = OuterCols - (batch_count * batch_size * vector_size) - (leftover_vectors * vector_size);

        float32x4x4_t res;
        for (U32 a_r = 0; a_r < OuterRows; ++a_r) {
            for (U32 batch_i = 0; batch_i < batch_count; ++batch_i) {
                for (U32 b_c = 0; b_c < batch_size; ++b_c) {
                    res.val[b_c] = vld1q_f32(&result(a_r, ((batch_i * batch_size) + b_c) * vector_size));
                }
                for (U32 b_r = 0; b_r < Inner; ++b_r) {
                    for (U32 b_c = 0; b_c < batch_size; ++b_c) {
                        float32x4_t x = vld1q_f32(&b(b_r, ((batch_i * batch_size) + b_c) * vector_size));
                        x = vmulq_n_f32(x, a(a_r, b_r));
                        res.val[b_c] = vaddq_f32(res.val[b_c], x);
                    }
                }
                vst1q_f32_x4(&result(a_r, batch_i * batch_size * vector_size), res);
            }
            if constexpr (leftover_vectors > 0) {
                const U32 start = batch_count * batch_size * vector_size;
                for (U32 b_c = 0; b_c < leftover_vectors; ++b_c) {
                    res.val[b_c] = vld1q_f32(&result(a_r, start + (b_c * vector_size)));
                }
                for (U32 b_r = 0; b_r < Inner; ++b_r) {
                    for (U32 b_c = 0; b_c < leftover_vectors; ++b_c) {
                        float32x4_t x = vld1q_f32(&b(b_r, start + (b_c * vector_size)));
                        x = vmulq_n_f32(x, a(a_r, b_r));
                        res.val[b_c] = vaddq_f32(res.val[b_c], x);
                    }
                }
                if constexpr (leftover_vectors == 1) {
                    vst1q_f32(&result(a_r, start), res.val[0]);
                } else if constexpr (leftover_vectors == 2) {
                    vst1q_f32(&result(a_r, start), res.val[0]);
                    vst1q_f32(&result(a_r, start + 4), res.val[1]);
                } else if constexpr (leftover_vectors == 3) {
                    vst1q_f32(&result(a_r, start), res.val[0]);
                    vst1q_f32(&result(a_r, start + 4), res.val[1]);
                    vst1q_f32(&result(a_r, start + 8), res.val[2]);
                }
            }
            if constexpr (leftover_singles > 0) {
                const U32 start = batch_count * batch_size * vector_size + leftover_vectors * vector_size;
                for (U32 b_c = start; b_c < OuterCols; ++b_c) {
#if 0
                    float b_t[Inner];
                    transpose_col<Inner, OuterCols>(b_t, b, b_c);
                    result(a_r, b_c) += multiply_3_for_row_and_col<Inner, OuterCols>(a(a_r), b_t);
#else
                    for (U32 i = 0; i < ACols; ++i) {
                        result(a_r, b_c) += a(a_r, i) * b(i, b_c);
                    }
#endif
                }
            }
        }
    }
#endif

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate_neon(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        if constexpr (BCols == 1) {
            multiply_accumulate_neon_1(result, a, b);
        } else {
            multiply_accumulate_neon_2(result, a, b);
        }
    }
#endif

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate_basic(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        for (U32 r = 0; r < ResultRows; ++r) {
            for (U32 c = 0; c < ResultCols; ++c) {
                for (U32 i = 0; i < ACols; ++i) {
                    result(r, c) += a(r, i) * b(i, c);
                }
            }
        }
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        // TODO(TB): assert result is not pointing to same data as a or b?
        static_assert(ACols == BRows);
        static_assert(ResultRows == ARows);
        static_assert(ResultCols == BCols);

#if USE_NEON
        multiply_accumulate_neon(result, a, b);
#else
        multiply_accumulate_basic(result, a, b);
#endif
    }

#if USE_NEON
    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols, U32 YRows, U32 YCols, U32 ZRows, U32 ZCols>
    void multiply_add_neon(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x, const Matrix<Type, YRows, YCols> y, const Matrix<Type, ZRows, ZCols> z) {
        if constexpr (YCols == 1) {
            multiply_neon_1(result, x, y);
        } else {
            multiply_neon_2(result, x, y);
        }
        add_neon(result, z);
    }
#endif

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols, U32 YRows, U32 YCols, U32 ZRows, U32 ZCols>
    void multiply_add_basic(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x, const Matrix<Type, YRows, YCols> y, const Matrix<Type, ZRows, ZCols> z) {
        for (U32 r = 0; r < ResultRows; ++r) {
            for (U32 c = 0; c < ResultCols; ++c) {
                result(r, c) = z(r, c);
                for (U32 i = 0; i < XCols; ++i) {
                    result(r, c) += x(r, i) * y(i, c);
                }
            }
        }
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols, U32 YRows, U32 YCols, U32 ZRows, U32 ZCols>
    void multiply_add(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x, const Matrix<Type, YRows, YCols> y, const Matrix<Type, ZRows, ZCols> z) {
        // TODO(TB): assert result is not pointing to same data as a or b?
        static_assert(XCols == YRows);
        static_assert(ResultRows == XRows);
        static_assert(ResultCols == YCols);
        static_assert(ResultRows == ZRows);
        static_assert(ResultCols == ZCols);

#if USE_NEON
        multiply_add_neon(result, x, y, z);
#else
        multiply_add_basic(result, x, y, z);
#endif
    }

#if USE_NEON
    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void elementwise_multiply_by_sigmoid_derivative_of_neon(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        static constexpr U32 count = ACols / 4;
        static constexpr U32 leftover_start = count * 4;
        float buffer[4];
        float32x4_t ones = vmovq_n_f32(1.0f);
        for (U32 r = 0; r < ARows; ++r) {
            for (U32 c = 0; c < count; ++c) {
                for (U32 i = 0; i < 4; ++i) {
                    buffer[i] = expf(-b(r, c * 4 + i));
                }

                float32x4_t x = vld1q_f32(&buffer[0]);
                x = vaddq_f32(ones, x);
#if 1
                float32x4_t x_inverse = vrecpeq_f32(x);
#else
                float32x4_t x_reciprocal = vrecpeq_f32(x);
                float32x4_t x_inverse = vmulq_f32(vrecpsq_f32(x, x_reciprocal), x_reciprocal);
#endif
                float32x4_t y = vsubq_f32(ones, x_inverse);
                y = vmulq_f32(x_inverse, y);
                float32x4_t z = vld1q_f32(&a(r, c * 4));
                z = vmulq_f32(z, y);
                vst1q_f32(&a(r, c * 4), z);
            }
            for (U32 c = leftover_start; c < ACols; ++c) {
                a(r, c) *= sigmoid_derivative(b(r, c));
            }
        }
        // sigmoid = 1.0f / (1.0f + expf(-x));
        // sigmoid_defivative = sigmoid(x) * (1.0f - sigmoid(x));
    }
#endif

#if USE_NEON
    template <U32 Count>
    void first_dc_da_neon(tla::Matrix<float, 1, Count> dc_da_l, tla::Matrix<float, 1, Count> a_l, tla::Matrix<float, 1, Count> output, tla::Matrix<float, 1, Count> z_l) {
        static constexpr U32 count = Count / 4;
        static constexpr U32 leftover_start = count * 4;
        static constexpr U32 r = 0;
        float buffer[4];
        float32x4_t ones = vmovq_n_f32(1.0f);
        for (U32 c = 0; c < count; ++c) {
            for (U32 i = 0; i < 4; ++i) {
                buffer[i] = expf(-z_l(r, c * 4 + i));
            }

            float32x4_t x = vld1q_f32(&buffer[0]);
            x = vaddq_f32(ones, x);
#if 1
            float32x4_t x_inverse = vrecpeq_f32(x);
#else
            float32x4_t x_reciprocal = vrecpeq_f32(x);
            float32x4_t x_inverse = vmulq_f32(vrecpsq_f32(x, x_reciprocal), x_reciprocal);
#endif
            float32x4_t y = vsubq_f32(ones, x_inverse);
            // sigmoid_derivative(z_l)
            y = vmulq_f32(x_inverse, y);
            // a_l
            float32x4_t a = vld1q_f32(&a_l(r, c * 4));
            // output
            float32x4_t o = vld1q_f32(&output(r, c * 4));
            // a_l - output
            a = vsubq_f32(a, o);
            // (a_l - output) * sigmoid_derivative(z_l)
            a = vmulq_f32(a, y);
            // (a_l - output) * sigmoid_derivative(z_l) * 2
            a = vmulq_n_f32(a, 2.0f);
            vst1q_f32(&dc_da_l(r, c * 4), a);
        }
        for (U32 c = leftover_start; c < Count; ++c) {
            dc_da_l(r, c) = 2 * (a_l(r, c) - output(r, c)) * sigmoid_derivative(z_l(r, c));
        }
    }
#endif

    template <U32 Count>
    void first_dc_da_basic(tla::Matrix<float, 1, Count> dc_da_l, tla::Matrix<float, 1, Count> a_l, tla::Matrix<float, 1, Count> output, tla::Matrix<float, 1, Count> z_l) {
        for (U32 j = 0; j < Count; ++j) {
            dc_da_l(j) = 2 * (a_l(j) - output(j)) * sigmoid_derivative(z_l(j));
        }
    }

    template <U32 Count>
    void first_dc_da(tla::Matrix<float, 1, Count> dc_da_l, tla::Matrix<float, 1, Count> a_l, tla::Matrix<float, 1, Count> output, tla::Matrix<float, 1, Count> z_l) {
#if USE_NEON
        first_dc_da_neon(dc_da_l, a_l, output, z_l);
#else
        first_dc_da_basic(dc_da_l, a_l, output, z_l);
#endif
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void elementwise_multiply_by_sigmoid_derivative_of_basic(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        for (U32 r = 0; r < ARows; ++r) {
            for (U32 c = 0; c < ACols; ++c) {
                a(r, c) *= sigmoid_derivative(b(r, c));
            }
        }
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void elementwise_multiply_by_sigmoid_derivative_of(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        assert_equal_dimensions(a, b);

#if USE_NEON
        elementwise_multiply_by_sigmoid_derivative_of_neon(a, b);
#else
        elementwise_multiply_by_sigmoid_derivative_of_basic(a, b);
#endif

    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void divide(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        // TODO(TB): assert result is not pointing to same data as a or b?
        static_assert(ACols == BRows);
        static_assert(ResultRows == ARows);
        static_assert(ResultCols == BCols);

        clear(result);
        for (U32 r = 0; r < ResultRows; ++r) {
            for (U32 c = 0; c < ResultCols; ++c) {
                for (U32 i = 0; i < ACols; ++i) {
                    assert(b(i, c) != zero<Type>());
                    result(r, c) += a(r, i) / b(i, c);
                }
            }
        }
    }

#if USE_NEON
    template <typename Type, U32 Rows, U32 Cols>
    void add_neon(Matrix<Type, Rows, Cols> x, Type v) {
        static constexpr U32 count = Cols / 4;
        static constexpr U32 leftover_start = count * 4;
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < count; ++c) {
                float32x4_t x_row = vld1q_f32(&x(r, c * 4));
                x_row = vaddq_f32(x_row, v);
                vst1q_f32(&x(r, c * 4), x_row);
            }
            for (U32 c = leftover_start; c < Cols; ++c) {
                x(r, c) += v;
            }
        }
    }
#endif

    template <typename Type, U32 Rows, U32 Cols>
    void add_basic(Matrix<Type, Rows, Cols> x, Type v) {
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                x(r, c) += v;
            }
        }
    }

    template <typename Type, U32 Rows, U32 Cols>
    void add(Matrix<Type, Rows, Cols> x, Type v) {
        assert(v != zero<Type>());
#if USE_NEON
        add_neon(x, v);
#else
        add_basic(x, v);
#endif
    }

#if USE_NEON
    template <typename Type, U32 Rows, U32 Cols>
    void subtract_neon(Matrix<Type, Rows, Cols> x, Type v) {
        static constexpr U32 count = Cols / 4;
        static constexpr U32 leftover_start = count * 4;
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < count; ++c) {
                float32x4_t x_row = vld1q_f32(&x(r, c * 4));
                x_row = vsubq_f32(x_row, v);
                vst1q_f32(&x(r, c * 4), x_row);
            }
            for (U32 c = leftover_start; c < Cols; ++c) {
                x(r, c) -= v;
            }
        }
    }
#endif

    template <typename Type, U32 Rows, U32 Cols>
    void subtract_basic(Matrix<Type, Rows, Cols> x, Type v) {
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                x(r, c) -= v;
            }
        }
    }

    template <typename Type, U32 Rows, U32 Cols>
    void subtract(Matrix<Type, Rows, Cols> x, Type v) {
        assert(v != zero<Type>());
#if USE_NEON
        subtract_neon(x, v);
#else
        subtract_basic(x, v);
#endif
    }

#if USE_NEON
    template <typename Type, U32 Rows, U32 Cols>
    void multiply_neon(Matrix<Type, Rows, Cols> x, Type v) {
        static constexpr U32 count = Cols / 4;
        static constexpr U32 leftover_start = count * 4;
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < count; ++c) {
                float32x4_t x_row = vld1q_f32(&x(r, c * 4));
                x_row = vmulq_n_f32(x_row, v);
                vst1q_f32(&x(r, c * 4), x_row);
            }
            for (U32 c = leftover_start; c < Cols; ++c) {
                x(r, c) *= v;
            }
        }
    }
#endif

    template <typename Type, U32 Rows, U32 Cols>
    void multiply_basic(Matrix<Type, Rows, Cols> x, Type v) {
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                x(r, c) *= v;
            }
        }
    }

    template <typename Type, U32 Rows, U32 Cols>
    void multiply(Matrix<Type, Rows, Cols> x, Type v) {
        assert(v != zero<Type>());
#if USE_NEON
        multiply_neon(x, v);
#else
        multiply_basic(x, v);
#endif
    }

#if USE_NEON
    template <typename Type, U32 Rows, U32 Cols>
    void divide_neon(Matrix<Type, Rows, Cols> x, Type v) {
        static constexpr U32 count = Cols / 4;
        static constexpr U32 leftover_start = count * 4;
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < count; ++c) {
                float32x4_t x_row = vld1q_f32(&x(r, c * 4));
                x_row = vmulq_n_f32(x_row, 1.0f / v);
                vst1q_f32(&x(r, c * 4), x_row);
            }
            for (U32 c = leftover_start; c < Cols; ++c) {
                x(r, c) /= v;
            }
        }
    }
#endif

    template <typename Type, U32 Rows, U32 Cols>
    void divide_basic(Matrix<Type, Rows, Cols> x, Type v) {
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                x(r, c) /= v;
            }
        }
    }

    template <typename Type, U32 Rows, U32 Cols>
    void divide(Matrix<Type, Rows, Cols> x, Type v) {
        assert(v != zero<Type>());
#if USE_NEON
        divide_neon(x, v);
#else
        divide_basic(x, v);
#endif
    }

    template <typename Type, U32 Rows, U32 Cols>
    void print(const Matrix<Type, Rows, Cols> x) {
        for (U32 r = 0; r < Rows; ++r) {
            for(U32 c = 0; c < Cols; ++c) {
                printf("%f, ", x(r, c));
            }
            printf("%f\n", x(r, Cols - 1));
        }
        printf("\n");
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    bool operator==(const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        if constexpr (ARows != BRows) {
            return false;
        }

        if constexpr (ACols != BCols) {
            return false;
        }

        for (U32 r = 0; r < ARows; ++r) {
            for (U32 c = 0; c < ACols; ++c) {
                if (a(r, c) != b(r, c)) {
                    return false;
                }
            }
        }

        return true;
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline bool operator!=(const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        return !(a == b);
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator+=(Matrix<Type, Rows, Cols> x, Type v) {
        add(x, v);
        return x;
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator-=(Matrix<Type, Rows, Cols> x, Type v) {
        subtract(x, v);
        return x;
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator*=(Matrix<Type, Rows, Cols> x, Type v) {
        multiply(x, v);
        return x;
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator/=(Matrix<Type, Rows, Cols> x, Type v) {
        divide(x, v);
        return x;
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    Matrix<Type, ARows, ACols> operator+=(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        add(a, b);
        return a;
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    Matrix<Type, ARows, ACols> operator-=(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        subtract(a, b);
        return a;
    }
}
#endif // INCLUDE_TIM_LINEAR_ALGEBRA_HPP

#ifdef TIM_LINEAR_ALGEBRA_HPP_IMPLEMENTATION
namespace tla {
    float lerp(float a, float b, float d) {
        assert(d >= 0.0f && d <= 1.0f);
        return a + ((b - a) * d);
    }

    float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    void do_sigmoid(float& x) {
        x = sigmoid(x);
    }

    float sigmoid_derivative(float x) {
        return sigmoid(x) * (1.0f - sigmoid(x));
    }

    float relu(float x) {
        return x < 0 ? 0 : x;
    }

    void do_relu(float& x) {
        x = relu(x);
    }

    float relu_derivative(float x) {
        return x < 0 ? 0 : 1;
    }

    float rand_float() {
        return (float)rand() / (float)RAND_MAX;
    }

    float rand_float(float from, float to) {
        return rand_float() * (to - from) + from;
    }
}
#endif // TIM_LINEAR_ALGEBRA_HPP_IMPLEMENTATION

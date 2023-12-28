
#ifndef INCLUDE_TIM_LINEAR_ALGEBRA_HPP
#define INCLUDE_TIM_LINEAR_ALGEBRA_HPP

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <utility>

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

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols, U32 YRows, U32 YCols, U32 ZRows, U32 ZCols>
    void multiply_add(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, XRows, XCols> x, const Matrix<Type, YRows, YCols> y, const Matrix<Type, ZRows, ZCols> z);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void elementwise_multiply_by_sigmoid_derivative_of(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void divide(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b);

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

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply(Matrix<Type, ResultRows, ResultCols> result, const Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        // TODO(TB): assert result is not pointing to same data as a or b?
        static_assert(ACols == BRows);
        static_assert(ResultRows == ARows);
        static_assert(ResultCols == BCols);

        clear(result);
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

        for (U32 r = 0; r < ResultRows; ++r) {
            for (U32 c = 0; c < ResultCols; ++c) {
                for (U32 i = 0; i < ACols; ++i) {
                    result(r, c) += a(r, i) * b(i, c);
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

        for (U32 r = 0; r < ResultRows; ++r) {
            for (U32 c = 0; c < ResultCols; ++c) {
                result(r, c) = z(r, c);
                for (U32 i = 0; i < XCols; ++i) {
                    result(r, c) += x(r, i) * y(i, c);
                }
            }
        }
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void elementwise_multiply_by_sigmoid_derivative_of(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        assert_equal_dimensions(a, b);

        for (U32 r = 0; r < ARows; ++r) {
            for (U32 c = 0; c < ACols; ++c) {
                a(r, c) *= sigmoid_derivative(b(r, c));
            }
        }
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
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                x(r, c) += v;
            }
        }
        return x;
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator-=(Matrix<Type, Rows, Cols> x, Type v) {
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                x(r, c) -= v;
            }
        }
        return x;
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator*=(Matrix<Type, Rows, Cols> x, Type v) {
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                x(r, c) *= v;
            }
        }
        return x;
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols> operator/=(Matrix<Type, Rows, Cols> x, Type v) {
        assert(v != zero<Type>());
        for (U32 r = 0; r < Rows; ++r) {
            for (U32 c = 0; c < Cols; ++c) {
                x(r, c) /= v;
            }
        }
        return x;
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    Matrix<Type, ARows, ACols> operator+=(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        assert_equal_dimensions(a, b);

        for (U32 r = 0; r < ARows; ++r) {
            for (U32 c = 0; c < ACols; ++c) {
                a(r, c) += b(r, c);
            }
        }
        return a;
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    Matrix<Type, ARows, ACols> operator-=(Matrix<Type, ARows, ACols> a, const Matrix<Type, BRows, BCols> b) {
        assert_equal_dimensions(a, b);

        for (U32 r = 0; r < ARows; ++r) {
            for (U32 c = 0; c < ACols; ++c) {
                a(r, c) -= b(r, c);
            }
        }
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

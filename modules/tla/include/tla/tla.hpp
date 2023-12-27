
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

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline void assert_valid_to_assign(const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline void assert_equal_dimensions(const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline void copy_data(Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b);

    template <typename Type, U32 Rows, U32 Cols>
    inline constexpr U32 rows(const Matrix<Type, Rows, Cols>& x);

    template <typename Type, U32 Rows, U32 Cols>
    inline constexpr U32 cols(const Matrix<Type, Rows, Cols>& x);

    template <typename Type, U32 Rows, U32 Cols>
    inline constexpr U32 count(const Matrix<Type, Rows, Cols>& x);

    template <typename Type, U32 Rows, U32 Cols>
    inline constexpr U32 is_dynamic(const Matrix<Type, Rows, Cols>& x);

    template <typename Type, U32 Rows, U32 Cols>
    inline void fill(Matrix<Type, Rows, Cols>& x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    inline void clear(Matrix<Type, Rows, Cols>& x);

    template <typename Type, U32 Rows, U32 Cols>
    void fill_random(Matrix<Type, Rows, Cols>& x);

    template <typename Type, U32 Rows, U32 Cols>
    void fill_random(Matrix<Type, Rows, Cols>& x, float from, float to);

    template <typename Type, U32 Rows, U32 Cols, typename Function>
    void apply(Matrix<Type, Rows, Cols>& x, Function f);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply(Matrix<Type, ResultRows, ResultCols>& result, const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b);

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void divide(Matrix<Type, ResultRows, ResultCols>& result, const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b);

    template <typename Type, U32 Rows, U32 Cols>
    void print(const Matrix<Type, Rows, Cols>& x);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    bool operator==(const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline bool operator!=(const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b);

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols>& operator+=(Matrix<Type, Rows, Cols>& x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols>& operator-=(Matrix<Type, Rows, Cols>& x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols>& operator*=(Matrix<Type, Rows, Cols>& x, Type v);

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols>& operator/=(Matrix<Type, Rows, Cols>& x, Type v);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    Matrix<Type, ARows, ACols>& operator+=(Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b);

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    Matrix<Type, ARows, ACols>& operator-=(Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b);

    template <typename Type, U32 Rows, U32 Cols>
    struct Matrix {
        using ValueType = Type;
        // Rows > 0
        // Cols > 0
        using ThisType = Matrix<Type, Rows, Cols>;

        Type* m_data;

        explicit Matrix(Type* in_data)
            : m_data(in_data)
        {}

        template <U32 OtherRows, U32 OtherCols>
        Matrix(Matrix<Type, OtherRows, OtherCols>& other)
            : m_data(other.m_data)
        {
            assert_valid_to_assign(*this, other);
        }
        
        Matrix(ThisType& other)
            : m_data(other.m_data)
        {
            assert_valid_to_assign(*this, other);
        }

        template <U32 OtherRows, U32 OtherCols>
        Matrix(Matrix<Type, OtherRows, OtherCols>&& other)
            : m_data(other.m_data)
        {
            assert_valid_to_assign(*this, other);
        }

        Matrix(ThisType&& other)
            : m_data(other.m_data)
        {
            assert_valid_to_assign(*this, other);
        }

        template <U32 OtherRows, U32 OtherCols>
        inline ThisType& operator=(const Matrix<Type, OtherRows, OtherCols>& other) {
            assert_valid_to_assign(*this, other);
            copy_data(*this, other);
            return *this;
        }

        inline ThisType& operator=(const ThisType& other) {
            assert_valid_to_assign(*this, other);
            copy_data(*this, other);
            return *this;
        }
        
        template <U32 OtherRows, U32 OtherCols>
        inline ThisType& operator=(Matrix<Type, OtherRows, OtherCols>&& other) {
            assert_valid_to_assign(*this, other);
            m_data = other.m_data;
            return *this;
        }

        inline ThisType& operator=(ThisType&& other) {
            assert_valid_to_assign(*this, other);
            m_data = other.m_data;
            return *this;
        }

        template <U32 OtherRows, U32 OtherCols>
        inline void view(Matrix<Type, OtherRows, OtherCols>& other) {
            assert_valid_to_assign(*this, other);
            m_data = other.m_data;
        }

        inline void view(Type* new_data) {
            m_data = new_data;
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

        Matrix<Type, Cols, Rows> free_transpose() {
            static_assert(Cols == 1 || Rows == 1);
            return Matrix<Type, Cols, Rows>(m_data);
        }
    };

    template <typename Type>
    struct Matrix<Type, 0, 0> {
        using ValueType = Type;
        // Rows = 0
        // Cols = 0
        using ThisType = Matrix<Type, 0, 0>;

        Type* m_data;
        U32 m_rows;
        U32 m_cols;

        Matrix(Type* in_data, U32 in_rows, U32 in_cols)
            : m_data(in_data)
            , m_rows(in_rows)
            , m_cols(in_cols)
        {
            assert(m_rows > 0);
            assert(m_cols > 0);
        }

        template <U32 OtherRows, U32 OtherCols>
        Matrix(Matrix<Type, OtherRows, OtherCols>& other)
            : m_data(other.m_data)
            , m_rows(rows(other))
            , m_cols(cols(other))
        {}

        Matrix(ThisType& other)
            : m_data(other.m_data)
            , m_rows(rows(other))
            , m_cols(cols(other))
        {}

        template <U32 OtherRows, U32 OtherCols>
        Matrix(Matrix<Type, OtherRows, OtherCols>&& other)
            : m_data(other.m_data)
            , m_rows(rows(other))
            , m_cols(cols(other))
        {}

        Matrix(ThisType&& other)
            : m_data(other.m_data)
            , m_rows(rows(other))
            , m_cols(cols(other))
        {}

        template <U32 OtherRows, U32 OtherCols>
        inline ThisType& operator=(const Matrix<Type, OtherRows, OtherCols>& other) {
            assert_valid_to_assign(*this, other);
            copy_data(*this, other);
            return *this;
        }

        inline ThisType& operator=(const ThisType& other) {
            assert_valid_to_assign(*this, other);
            copy_data(*this, other);
            return *this;
        }

        template <U32 OtherRows, U32 OtherCols>
        inline ThisType& operator=(Matrix<Type, OtherRows, OtherCols>&& other) {
            assert_valid_to_assign(*this, other);
            m_data = other.m_data;
            m_rows = rows(other);
            m_cols = cols(other);
            return *this;
        }

        inline ThisType& operator=(ThisType&& other) {
            assert_valid_to_assign(*this, other);
            m_data = other.m_data;
            m_rows = rows(other);
            m_cols = cols(other);
            return *this;
        }

        template <U32 OtherRows, U32 OtherCols>
        inline void view(Matrix<Type, OtherRows, OtherCols>& other) {
            m_data = other.m_data;
            m_rows = rows(other);
            m_cols = cols(other);
        }

        inline void view(Type* new_data, U32 new_rows, U32 new_cols) {
            m_data = new_data;
            m_rows = new_rows;
            m_cols = new_cols;
        }

        inline void resize(U32 new_rows, U32 new_cols) {
            m_rows = new_rows;
            m_cols = new_cols;
        }

        Matrix<Type, 0, 0> transpose() {
            if (m_rows == 1 || m_cols == 1) {
                return Matrix<Type, 0, 0>(m_data, m_cols, m_rows);
            }

            Matrix<Type, 0, 0> result(m_data, m_cols, m_rows);
            transpose<Type, 0, 0, 0, 0>(result, *this);
            return result;
        }

        inline const Type& operator()(U32 r, U32 c) const {
            assert(r < m_rows && c < m_cols);
            return m_data[r * m_cols + c];
        }

        inline Type& operator()(U32 r, U32 c) {
            return const_cast<Type&>(static_cast<const ThisType*>(this)->operator()(r, c));
        }

        inline const Type& operator()(U32 i) const {
            assert((m_rows == 1 && m_cols == 1) ? (i == 0) : (m_rows == 1) ? (i < m_cols) : (m_cols == 1) ? (i < m_rows) : false);
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

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline void assert_valid_to_assign(const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        if constexpr (ARows == 0 || ACols == 0) {
            // a is dynamic
            if constexpr (BRows == 0 || BCols == 0) {
                // b is dymamic
                // if either is a row or col vector, allow sizes to be equal or opposite, otherwise they must be equal
                assert((rows(b) == rows(a) && cols(b) == cols(a)) || ((rows(a) == 1 || cols(a) == 1) && (rows(b) == cols(a) && cols(b) == rows(a))));
            } else {
                // b is static
                if constexpr (BRows == 1 || BCols == 1) {
                    // either a row vector or col vector (or 1x1 matrix)
                    // allow col vector to be assigned to row vector and visa versa
                    assert((rows(b) == rows(a) && cols(b) == cols(a)) || (rows(b) == cols(a) && cols(b) == rows(a)));
                } else {
                    assert(rows(b) == rows(a) && cols(b) == cols(a));
                }
            }
        } else {
            // a is static
            if constexpr (BRows == 0 || BCols == 0) {
                // b is dymamic
                if constexpr (ARows == 1 || ACols == 1) {
                    // either a row vector or col vector (or 1x1 matrix)
                    // allow col vector to be assigned to row vector and visa versa
                    assert((rows(b) == rows(a) && cols(b) == cols(a)) || (rows(b) == cols(a) && cols(b) == rows(a)));
                } else {
                    assert(rows(b) == rows(a) && cols(b) == cols(a));
                }
            } else {
                // b is static
                if constexpr (ARows == 1 || ACols == 1) {
                    // either a row vector or col vector (or 1x1 matrix)
                    // allow col vector to be assigned to row vector and visa versa
                    static_assert((BRows == ARows && BCols == ACols) || (BRows == ACols && BCols == ARows));
                } else {
                    static_assert(BRows == ARows && BCols == ACols);
                }
            }
        }
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline void assert_equal_dimensions(const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        if constexpr (ARows == 0 || BRows == 0) {
            assert(rows(a) == rows(b));
        } else {
            static_assert(ARows == BRows);
        }

        if constexpr (ACols == 0 || BCols == 0) {
            assert(cols(a) == cols(b));
        } else {
            static_assert(ACols == BCols);
        }
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline void copy_data(Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        if constexpr (ARows != 0 && ACols != 0 && BRows != 0 && BCols != 0) {
            static_assert(ARows * ACols == BRows * BCols);
        } else {
            assert(rows(a) * cols(a) == rows(b) * cols(b));
        }
        memcpy(a.m_data, b.m_data, ARows * ACols * sizeof(Type));
    }

    template <typename Type, U32 Rows, U32 Cols>
    inline constexpr U32 rows(const Matrix<Type, Rows, Cols>& x) {
        if constexpr (Rows == 0 || Cols == 0) {
            return x.m_rows;
        } else {
            return Rows;
        }
    }

    template <typename Type, U32 Rows, U32 Cols>
    inline constexpr U32 cols(const Matrix<Type, Rows, Cols>& x) {
        if constexpr (Rows == 0 || Cols == 0) {
            return x.m_cols;
        } else {
            return Cols;
        }
    }

    template <typename Type, U32 Rows, U32 Cols>
    inline constexpr U32 count(const Matrix<Type, Rows, Cols>& x) {
        if constexpr (Rows == 0 || Cols == 0) {
            return x.m_rows * x.m_cols;
        } else {
            return Rows * Cols;
        }
    }

    template <typename Type, U32 Rows, U32 Cols>
    inline constexpr U32 is_dynamic(const Matrix<Type, Rows, Cols>& x) {
        return Rows == 0 || Cols == 0;
    }

    template <typename Type, U32 Rows, U32 Cols>
    inline void fill(Matrix<Type, Rows, Cols>& x, Type v) {
        memset(x.m_data, v, rows(x) * cols(x) * sizeof(Type));
    }

    template <typename Type, U32 Rows, U32 Cols>
    inline void clear(Matrix<Type, Rows, Cols>& x) {
        fill(x, zero<Type>());
    }

    template <typename Type, U32 Rows, U32 Cols>
    void fill_random(Matrix<Type, Rows, Cols>& x) {
        for (U32 r = 0; r < rows(x); ++r) {
            for (U32 c = 0; c < cols(x); ++c) {
                x(r, c) = rand_float();
            }
        }
    }

    template <typename Type, U32 Rows, U32 Cols>
    void fill_random(Matrix<Type, Rows, Cols>& x, float from, float to) {
        for (U32 r = 0; r < rows(x); ++r) {
            for (U32 c = 0; c < cols(x); ++c) {
                x(r, c) = rand_float(from, to);
            }
        }
    }

    template <typename Type, U32 Rows, U32 Cols, typename Function>
    void apply(Matrix<Type, Rows, Cols>& x, Function f) {
        for (U32 r = 0; r < rows(x); ++r) {
            for (U32 c = 0; c < cols(x); ++c) {
                f(x(r, c));
            }
        }
    }

    template <typename Type, U32 XRows, U32 XCols, U32 YRows, U32 YCols, typename Function>
    void assign_apply(Matrix<Type, XRows, XCols>& x, const Matrix<Type, YRows, YCols>& y, Function f) {
        static_assert(XRows == YRows || XRows == 0 || YRows == 0);
        assert(rows(x) == rows(y));

        static_assert(XCols == YCols || XCols == 0 || YCols == 0);
        assert(cols(x) == cols(y));

        for (U32 r = 0; r < rows(x); ++r) {
            for (U32 c = 0; c < cols(x); ++c) {
                x(r, c) = f(y(r, c));
            }
        }
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply(Matrix<Type, ResultRows, ResultCols>& result, const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        // TODO(TB): assert result is not pointing to same data as a or b?
        static_assert(ACols == BRows || ACols == 0 || BRows == 0);
        assert(cols(a) == rows(b));

        static_assert(ResultRows == ARows || ResultRows == 0 || ARows == 0);
        assert(rows(result) == rows(a));

        static_assert(ResultCols == BCols || ResultCols == 0 || BCols == 0);
        assert(cols(result) == cols(b));

        clear(result);
        for (U32 r = 0; r < rows(result); ++r) {
            for (U32 c = 0; c < cols(result); ++c) {
                for (U32 i = 0; i < cols(a); ++i) {
                    result(r, c) += a(r, i) * b(i, c);
                }
            }
        }
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void multiply_accumulate(Matrix<Type, ResultRows, ResultCols>& result, const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        // TODO(TB): assert result is not pointing to same data as a or b?
        static_assert(ACols == BRows || ACols == 0 || BRows == 0);
        assert(cols(a) == rows(b));

        static_assert(ResultRows == ARows || ResultRows == 0 || ARows == 0);
        assert(rows(result) == rows(a));

        static_assert(ResultCols == BCols || ResultCols == 0 || BCols == 0);
        assert(cols(result) == cols(b));

        for (U32 r = 0; r < rows(result); ++r) {
            for (U32 c = 0; c < cols(result); ++c) {
                for (U32 i = 0; i < cols(a); ++i) {
                    result(r, c) += a(r, i) * b(i, c);
                }
            }
        }
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols, U32 YRows, U32 YCols, U32 ZRows, U32 ZCols>
    void multiply_add(Matrix<Type, ResultRows, ResultCols>& result, const Matrix<Type, XRows, XCols>& x, const Matrix<Type, YRows, YCols>& y, const Matrix<Type, ZRows, ZCols>& z) {
        // TODO(TB): assert result is not pointing to same data as a or b?
        static_assert(XCols == YRows || XCols == 0 || YRows == 0);
        assert(cols(x) == rows(y));

        static_assert(ResultRows == XRows || ResultRows == 0 || XRows == 0);
        assert(rows(result) == rows(x));

        static_assert(ResultCols == YCols || ResultCols == 0 || YCols == 0);
        assert(cols(result) == cols(y));

        static_assert(ResultRows == ZRows || ResultRows == 0 || ZRows == 0);
        assert(rows(result) == rows(z));

        static_assert(ResultCols == ZCols || ResultCols == 0 || ZCols == 0);
        assert(cols(result) == cols(z));

        for (U32 r = 0; r < rows(result); ++r) {
            for (U32 c = 0; c < cols(result); ++c) {
                result(r, c) = z(r, c);
                for (U32 i = 0; i < cols(x); ++i) {
                    result(r, c) += x(r, i) * y(i, c);
                }
            }
        }
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols>
    void transpose(Matrix<Type, ResultRows, ResultCols>& result, const Matrix<Type, XRows, XCols>& x) {
        static_assert(ResultRows == XCols || ResultRows == 0 || XCols == 0);
        assert(rows(result) == cols(x));

        static_assert(ResultCols == XRows || ResultCols == 0 || XRows == 0);
        assert(cols(result) == rows(x));

        for (U32 r = 0; r < rows(x); ++r) {
            for (U32 c = 0; c < cols(x); ++c) {
                result(c, r) = x(r, c);
            }
        }
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 XRows, U32 XCols>
    void transpose_col(Matrix<Type, ResultRows, ResultCols>& result, const Matrix<Type, XRows, XCols>& x, U32 c) {
        static_assert(ResultRows == XRows || ResultRows == 0 || XRows == 0);
        assert(rows(result) == rows(x));

        static_assert(ResultCols == 1 || ResultCols == 0);
        assert(cols(result) == 1);

        for (U32 r = 0; r < rows(x); ++r) {
            result(r) = x(r, c);
        }
    }

    template <typename Type, U32 ResultRows, U32 ResultCols, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void divide(Matrix<Type, ResultRows, ResultCols>& result, const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        // TODO(TB): assert result is not pointing to same data as a or b?
        static_assert(ACols == BRows || ACols == 0 || BRows == 0);
        assert(cols(a) == rows(b));

        static_assert(ResultRows == ARows || ResultRows == 0 || ARows == 0);
        assert(rows(result) == rows(a));

        static_assert(ResultCols == BCols || ResultCols == 0 || BCols == 0);
        assert(cols(result) == cols(b));

        clear(result);
        for (U32 r = 0; r < rows(result); ++r) {
            for (U32 c = 0; c < cols(result); ++c) {
                for (U32 i = 0; i < cols(a); ++i) {
                    assert(b(i, c) != zero<Type>());
                    result(r, c) += a(r, i) / b(i, c);
                }
            }
        }
    }

    template <typename Type, U32 Rows, U32 Cols>
    void print(const Matrix<Type, Rows, Cols>& x) {
        for (U32 r = 0; r < rows(x); ++r) {
            for(U32 c = 0; c < cols(x) - 1; ++c) {
                printf("%f, ", x(r, c));
            }
            printf("%f\n", x(r, cols(x) - 1));
        }
        printf("\n");
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    bool operator==(const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        if constexpr (ARows == 0 || BRows == 0) {
            if (rows(a) != rows(b)) {
                return false;
            }
        } else if constexpr (ARows != BRows) {
            return false;
        }

        if constexpr (ACols == 0 || BCols == 0) {
            if (cols(a) != cols(b)) {
                return false;
            }
        } else if constexpr (ACols != BCols) {
            return false;
        }

        for (U32 r = 0; r < rows(a); ++r) {
            for (U32 c = 0; c < cols(a); ++c) {
                if (a(r, c) != b(r, c)) {
                    return false;
                }
            }
        }

        return true;
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    inline bool operator!=(const Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        return !(a == b);
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols>& operator+=(Matrix<Type, Rows, Cols>& x, Type v) {
        for (U32 r = 0; r < rows(x); ++r) {
            for (U32 c = 0; c < cols(x); ++c) {
                x(r, c) += v;
            }
        }
        return x;
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols>& operator-=(Matrix<Type, Rows, Cols>& x, Type v) {
        for (U32 r = 0; r < rows(x); ++r) {
            for (U32 c = 0; c < cols(x); ++c) {
                x(r, c) -= v;
            }
        }
        return x;
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols>& operator*=(Matrix<Type, Rows, Cols>& x, Type v) {
        for (U32 r = 0; r < rows(x); ++r) {
            for (U32 c = 0; c < cols(x); ++c) {
                x(r, c) *= v;
            }
        }
        return x;
    }

    template <typename Type, U32 Rows, U32 Cols>
    Matrix<Type, Rows, Cols>& operator/=(Matrix<Type, Rows, Cols>& x, Type v) {
        assert(v != zero<Type>());
        for (U32 r = 0; r < rows(x); ++r) {
            for (U32 c = 0; c < cols(x); ++c) {
                x(r, c) /= v;
            }
        }
        return x;
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    Matrix<Type, ARows, ACols>& operator+=(Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        assert_equal_dimensions(a, b);

        for (U32 r = 0; r < rows(a); ++r) {
            for (U32 c = 0; c < cols(a); ++c) {
                a(r, c) += b(r, c);
            }
        }
        return a;
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    Matrix<Type, ARows, ACols>& operator-=(Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        assert_equal_dimensions(a, b);

        for (U32 r = 0; r < rows(a); ++r) {
            for (U32 c = 0; c < cols(a); ++c) {
                a(r, c) -= b(r, c);
            }
        }
        return a;
    }

    template <typename Type, U32 ARows, U32 ACols, U32 BRows, U32 BCols>
    void elementwise_multiply_by_sigmoid_derivative_of(Matrix<Type, ARows, ACols>& a, const Matrix<Type, BRows, BCols>& b) {
        assert_equal_dimensions(a, b);

        for (U32 r = 0; r < rows(a); ++r) {
            for (U32 c = 0; c < cols(a); ++c) {
                a(r, c) *= sigmoid_derivative(b(r, c));
            }
        }
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

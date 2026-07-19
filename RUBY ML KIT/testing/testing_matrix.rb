require_relative 'matrix'
require_relative 'MODELS/logistic_regression'


def assert_equal(expected, actual, test_name)
  if expected == actual
    puts "PASS: #{test_name}"
  else
    puts "FAIL: #{test_name}"
    puts "  expected: #{expected.inspect}"
    puts "  actual:   #{actual.inspect}"
  end
end

def assert_matrices_close(expected_matrix, actual_matrix, test_name, tolerance = 1e-9)
  same_shape = expected_matrix.same_shape?(actual_matrix)
  if !same_shape
    puts "FAIL: #{test_name} (shape mismatch)"
    return
  end

  all_close = true
  for row_idx in 0...expected_matrix.rows
    for col_idx in 0...expected_matrix.cols
      diff = (expected_matrix.get(row_idx, col_idx) - actual_matrix.get(row_idx, col_idx)).abs
      all_close = false if diff > tolerance
    end
  end

  if all_close
    puts "PASS: #{test_name}"
  else
    puts "FAIL: #{test_name}"
    puts "  expected: #{expected_matrix.data.inspect}"
    puts "  actual:   #{actual_matrix.data.inspect}"
  end
end


# ============ KNOWN-VALUE TEST ============

# Hand-calculated: for [[4, 7], [2, 6]], det = 4*6 - 7*2 = 10
# inverse = (1/10) * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
a = Matrix.new([[4, 7], [2, 6]])
expected_inverse = Matrix.new([[0.6, -0.7], [-0.2, 0.4]])
result = a.inverse
assert_matrices_close(expected_inverse, result, "inverse: known 2x2 hand-calculated result")


# ============ ROUND-TRIP TEST (most important) ============

# A x A_inverse should equal the identity matrix
identity_check = a.matrix_multiply(a.inverse)
assert_matrices_close(Matrix.identity(2), identity_check, "inverse: A x A_inverse = I")

# A_inverse x A should ALSO equal identity (true for square invertible matrices)
identity_check_reversed = a.inverse.matrix_multiply(a)
assert_matrices_close(Matrix.identity(2), identity_check_reversed, "inverse: A_inverse x A = I")


# ============ LARGER MATRIX ROUND-TRIP ============

b = Matrix.new([
  [2, 1, 1],
  [1, 3, 2],
  [1, 0, 0]
])
identity_check_3x3 = b.matrix_multiply(b.inverse)
assert_matrices_close(Matrix.identity(3), identity_check_3x3, "inverse: 3x3 A x A_inverse = I")


# ============ IDENTITY MATRIX IS ITS OWN INVERSE ============

id3 = Matrix.identity(3)
assert_matrices_close(id3, id3.inverse, "inverse: identity matrix is its own inverse")


# ============ NON-SQUARE MATRIX SHOULD RAISE ============

non_square = Matrix.new([[1, 2, 3], [4, 5, 6]])
begin
  non_square.inverse
  puts "FAIL: non-square matrix should have raised on inverse"
rescue ArgumentError
  puts "PASS: non-square matrix raises ArgumentError on inverse"
end


# ============ SINGULAR MATRIX SHOULD RAISE ============

# Row 2 is just 2x row 1 -- determinant is 0, not invertible
singular = Matrix.new([[1, 2], [2, 4]])
begin
  singular.inverse
  puts "FAIL: singular matrix should have raised on inverse"
rescue ArgumentError
  puts "PASS: singular matrix raises ArgumentError on inverse"
end

# ============ SLICE_MATRIX (keyword arguments) ============

m = Matrix.new([
  [1,  2,  3,  4],
  [5,  6,  7,  8],
  [9,  10, 11, 12],
  [13, 14, 15, 16]
])

# Test: top-left 2x2 corner
result = m.slice(row_start: 0, row_end: 1, col_start: 0, col_end: 1)
assert_equal([[1, 2], [5, 6]], result.data, "slice_matrix: top-left 2x2 with keyword args")

# Test: right half of the matrix (this is the exact shape inverse() needs)
result = m.slice(row_start: 0, row_end: 3, col_start: 2, col_end: 3)
assert_equal([[3, 4], [7, 8], [11, 12], [15, 16]], result.data, "slice_matrix: right half via keyword args")

# Test: defaults still work (no args = whole matrix)
result = m.slice
assert_equal(m.data, result.data, "slice_matrix: defaults return full matrix")

# Test: mixing only some keyword args, relying on defaults for the rest
result = m.slice(row_end: 1)
assert_equal([[1, 2, 3, 4], [5, 6, 7, 8]], result.data, "slice_matrix: partial keyword args use defaults for the rest")


# ============ ROW OPERATION PRIMITIVES ============

# Test: swap_rows! mutates in place correctly
a = Matrix.new([[1, 2], [3, 4]])
a.swap_rows!(0, 1)
assert_equal([[3, 4], [1, 2]], a.data, "swap_rows!: swaps two rows in place")

# Test: scale_row! mutates in place correctly
b = Matrix.new([[1, 2], [3, 4]])
b.scale_row!(0, 10)
assert_equal([[10, 20], [3, 4]], b.data, "scale_row!: scales a single row in place")

# Test: scale_row! by 0 zeroes out the row (sanity check on the scalar actually being used)
c = Matrix.new([[1, 2], [3, 4]])
c.scale_row!(1, 0)
assert_equal([[1, 2], [0, 0]], c.data, "scale_row!: scaling by 0 zeroes the row")

# Test: add_scaled_row! with scalar 0 leaves target row unchanged
d = Matrix.new([[1, 2], [3, 4]])
d.add_scaled_row!(0, 1, 0)
assert_equal([[1, 2], [3, 4]], d.data, "add_scaled_row!: scalar of 0 has no effect")

# Test: add_scaled_row! actually uses the scalar (regression test for the bug we caught earlier)
e = Matrix.new([[1, 2], [3, 4]])
e.add_scaled_row!(0, 1, 2)
# row 0 becomes row0 + 2*row1 = [1,2] + [6,8] = [7,10]
assert_equal([[7, 10], [3, 4]], e.data, "add_scaled_row!: scalar is correctly applied")


# ============ REDUCED_ROW_ECHELON_FORM ============

# Test: known 2x2 case
f = Matrix.new([[2, 4], [1, 3]])
result = f.reduced_row_echelon_form
assert_matrices_close(Matrix.identity(2), result, "reduced_row_echelon_form: invertible 2x2 reduces to identity")

=begin
# Test: singular matrix does NOT reduce to identity
g = Matrix.new([[1, 2], [2, 4]])  # row 2 = 2x row 1
result = g.reduced_row_echelon_form
#identity_check = result.approximately_equals?(Matrix.identity(2))
identity_check = assert_matrices_close(Matrix.identity(2), result, "singular matrix from RREF does not reduce to ID")
assert_equal(false, identity_check, "reduced_row_echelon_form: singular matrix does not reduce to identity")
=end

# Test: identity matrix reduces to itself
id3 = Matrix.identity(3)
result = id3.reduced_row_echelon_form
assert_matrices_close(id3, result, "reduced_row_echelon_form: identity matrix stays identity")


# ============ TRANSPOSE ============

# Test: basic non-square transpose
h = Matrix.new([[1, 2, 3], [4, 5, 6]])  # 2x3
result = h.transpose
assert_equal([[1, 4], [2, 5], [3, 6]], result.data, "transpose: 2x3 becomes 3x2")

# Test: shape is correctly swapped
assert_equal([3, 2], result.shape, "transpose: shape dimensions are swapped")

# Test: transposing twice returns the original
double_transpose = h.transpose.transpose
assert_equal(h.data, double_transpose.data, "transpose: double transpose returns original")

# Test: transposing a symmetric matrix returns an equal matrix
symmetric = Matrix.new([[1, 2], [2, 1]])
assert_equal(true, symmetric.transpose.equals?(symmetric), "transpose: symmetric matrix equals its own transpose")


# ============ HADAMARD (element-wise multiply) ============

i = Matrix.new([[1, 2], [3, 4]])
j = Matrix.new([[5, 6], [7, 8]])

# Test: basic known-value case
result = i.hadamard_multiply(j)
assert_equal([[5, 12], [21, 32]], result.data, "hadamard: basic element-wise multiplication")

# Test: hadamard with identity-like all-ones matrix returns original
ones = Matrix.new([[1, 1], [1, 1]])
result = i.hadamard_multiply(ones)
assert_equal(i.data, result.data, "hadamard: multiplying by all-ones matrix returns original")

# Test: hadamard with zero matrix gives all zeros
zero = Matrix.create_zeroes(2, 2)
result = i.hadamard_multiply(Matrix.new(zero))
assert_equal([[0, 0], [0, 0]], result.data, "hadamard: multiplying by zero matrix gives all zeros")

# Test: mismatched dimensions should raise
k = Matrix.new([[1, 2, 3], [4, 5, 6]])
begin
  i.hadamard_multiply(k)
  puts "FAIL: hadamard with mismatched dimensions should have raised"
rescue ArgumentError
  puts "PASS: hadamard with mismatched dimensions raises ArgumentError"
end

# ============ COMBINE (underlying primitive) ============

a = Matrix.new([[1, 2], [3, 4]])
b = Matrix.new([[5, 6], [7, 8]])

# Test: combine with a custom block (not add/subtract/hadamard specifically)
result = a.combine(b) { |x, y| x * 10 + y }
assert_equal([[15, 26], [37, 48]], result.data, "combine: custom block combines cell-by-cell correctly")

# Test: combine requires same shape
c = Matrix.new([[1, 2, 3], [4, 5, 6]])
begin
  a.combine(c) { |x, y| x + y }
  puts "FAIL: combine with mismatched dimensions should have raised"
rescue ArgumentError
  puts "PASS: combine with mismatched dimensions raises ArgumentError"
end

# Test: combine does not mutate either input matrix
a_before = a.data.dup
b_before = b.data.dup
a.combine(b) { |x, y| x + y }
assert_equal(a_before, a.data, "combine: does not mutate self")
assert_equal(b_before, b.data, "combine: does not mutate other")


# ============ ADD (via combine) ============

result = a.add(b)
assert_equal([[6, 8], [10, 12]], result.data, "add (combine-based): basic 2x2 addition")

zero = Matrix.new(Matrix.create_zeroes(2, 2))
result = a.add(zero)
assert_equal(a.data, result.data, "add (combine-based): adding zero matrix returns original")

begin
  a.add(c)
  puts "FAIL: add with mismatched dimensions should have raised"
rescue ArgumentError
  puts "PASS: add (combine-based) with mismatched dimensions raises ArgumentError"
end


# ============ SUBTRACT (via combine) ============

result = b.subtract(a)
assert_equal([[4, 4], [4, 4]], result.data, "subtract (combine-based): basic 2x2 subtraction")

result = a.subtract(a)
assert_equal([[0, 0], [0, 0]], result.data, "subtract (combine-based): matrix minus itself is zero")

begin
  a.subtract(c)
  puts "FAIL: subtract with mismatched dimensions should have raised"
rescue ArgumentError
  puts "PASS: subtract (combine-based) with mismatched dimensions raises ArgumentError"
end


# ============ HADAMARD (via combine) ============

result = a.hadamard_multiply(b)
assert_equal([[5, 12], [21, 32]], result.data, "hadamard (combine-based): basic element-wise multiplication")

ones = Matrix.new([[1, 1], [1, 1]])
result = a.hadamard_multiply(ones)
assert_equal(a.data, result.data, "hadamard (combine-based): multiplying by all-ones returns original")

begin
  a.hadamard_multiply(c)
  puts "FAIL: hadamard with mismatched dimensions should have raised"
rescue ArgumentError
  puts "PASS: hadamard (combine-based) with mismatched dimensions raises ArgumentError"
end


# ============ CROSS-CHECK: combine-based results match direct hand math ============

# A slightly larger case, so it's not just 2x2 coincidence
d = Matrix.new([[1, 2, 3], [4, 5, 6]])
e = Matrix.new([[10, 20, 30], [40, 50, 60]])

assert_equal([[11, 22, 33], [44, 55, 66]], d.add(e).data, "add (combine-based): 2x3 matrices")
assert_equal([[9, 18, 27], [36, 45, 54]], e.subtract(d).data, "subtract (combine-based): 2x3 matrices")
assert_equal([[10, 40, 90], [160, 250, 360]], d.hadamard_multiply(e).data, "hadamard (combine-based): 2x3 matrices")

puts "end of file"

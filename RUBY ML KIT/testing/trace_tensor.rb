require_relative "../neural_network/losses"
require_relative "../neural_network/layers"
require_relative "../core/tensor"
require_relative "../neural_network/activation_layers"

t = Tensor.new((0..11).to_a, [2, 2, 3])
puts t.strides.inspect
# => [6, 3, 1]

# --- get traces ---
puts t.get(0, 0, 0)   # => 0
puts t.get(0, 1, 2)   # => 5
puts t.get(1, 0, 0)   # => 6
puts t.get(1, 1, 2)   # => 11

# --- set trace ---
t.set(1, 0, 2, 99)
puts t.get(1, 0, 2)   # => 99

# --- validation trace ---
begin
  t.get(2, 0, 0)
rescue IndexError => e
  puts e.message
  # => "index 2 out of bounds for dimension 0 (size 2)"
end

# --- each_index trace ---
Tensor.each_index([2, 2, 3]) do |indices, flat_idx|
  puts "flat_idx=#{flat_idx} -> #{indices.inspect}"
end

t = Tensor.new((0..11).to_a, [2, 2, 3])

flat = t.reshape([4, 3])
puts flat.shape.inspect     # => [4, 3]
puts flat.strides.inspect   # => [3, 1]
puts flat.get(2, 1)         # => 7

back = flat.reshape([2, 2, 3])
puts back.get(1, 0, 1)      # => 7  (same value, original indexing scheme)

# Mismatched size should raise
begin
  t.reshape([5, 5])
rescue ArgumentError => e
  puts e.message
  # => "cannot reshape tensor of size 12 into shape [5, 5] (size 25)"
end

t = Tensor.new((0..11).to_a, [2, 2, 3])

transposed = t.transpose([0, 2, 1])
puts transposed.shape.inspect     # => [2, 3, 2]
puts transposed.strides.inspect   # => [6, 1, 3]
puts transposed.get(0, 2, 1)      # => 5, matches t.get(0, 1, 2)

# Invalid permutation should raise
begin
  t.transpose([0, 1])
rescue ArgumentError => e
  puts e.message
  # => "axes [0, 1] is not a valid permutation of 0..2"
end

t = Tensor.new((0..11).to_a, [2, 2, 3])
transposed = t.transpose([0, 2, 1])

puts transposed.contiguous?   # => false (strides are [6, 1, 3], not the fresh [6, 2, 1] for shape [2,3,2])

fixed = transposed.materialize
puts fixed.contiguous?        # => true
puts fixed.strides.inspect    # => [6, 2, 1] — freshly computed for shape [2, 3, 2]
puts fixed.get(0, 2, 1)       # => 5, same logical value as before
puts fixed.data.equal?(transposed.data)  # => false — genuinely separate array now
Tensor.broadcast_shape([8, 1, 64, 64], [3, 1, 64])

# --- broadcast_shape tests ---

# Standard case from the trace: [8,1,64,64] vs [3,1,64]
result = Tensor.broadcast_shape([8, 1, 64, 64], [3, 1, 64])
puts result.inspect
# expected => [8, 3, 64, 64]

# Same-rank, all compatible
result2 = Tensor.broadcast_shape([2, 3], [2, 1])
puts result2.inspect
# expected => [2, 3]

# Incompatible shapes should raise
begin
  Tensor.broadcast_shape([3, 4], [5, 4])
rescue ArgumentError => e
  puts "raised correctly: #{e.message}"
end

# --- combine tests ---

# Case 1: matrix + bias vector, matches the hand-trace from last message
a = Tensor.new((1..6).to_a, [2, 3])   # [[1,2,3],[4,5,6]]
b = Tensor.new([10, 20, 30], [3])     # broadcast across rows

sum = a.combine(b) { |x, y| x + y }
puts sum.shape.inspect        # expected => [2, 3]
puts sum.get(0, 0)            # expected => 1 + 10 = 11
puts sum.get(1, 2)            # expected => 6 + 30 = 36

# Case 2: same-shape tensors, no broadcasting needed
c = Tensor.new([1, 1, 1, 1], [2, 2])
d = Tensor.new([5, 6, 7, 8], [2, 2])
prod = c.combine(d) { |x, y| x * y }
puts prod.get(0, 1)   # expected => 1 * 6 = 6
puts prod.get(1, 1)   # expected => 1 * 8 = 8

# Case 3: scalar-like tensor [1,1] broadcasting against a bigger tensor
e = Tensor.new((1..12).to_a, [2, 2, 3])
scalar_like = Tensor.new([100], [1, 1, 1])
shifted = e.combine(scalar_like) { |x, y| x + y }
puts shifted.shape.inspect   # expected => [2, 2, 3]
puts shifted.get(1, 1, 2)    # expected => 12 + 100 = 112

# Case 4: incompatible combine should raise via broadcast_shape internally
begin
  f = Tensor.new([1, 2, 3, 4, 5], [5])
  g = Tensor.new([1, 2, 3, 4], [4])
  f.combine(g) { |x, y| x + y }
rescue ArgumentError => e
  puts "raised correctly: #{e.message}"
end

t = Tensor.new((0..5).to_a, [2, 3])
doubled = t.map { |v| v * 2 }

puts doubled.shape.inspect   # => [2, 3]
puts doubled.get(0, 0)       # => 0
puts doubled.get(1, 2)       # => 10

relu = ReLU.new
t = Tensor.new([-2, -1, 0, 1, 2, 3], [2, 3])
result = relu.forward(t)
puts result.data.inspect   # expect [0, 0, 0, 1, 2, 3]

puts "end of file"

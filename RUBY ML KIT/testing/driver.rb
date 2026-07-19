require_relative 'matrix'
require_relative 'vector_operations'
require_relative 'layers'
require_relative 'neural_network'
require_relative 'losses'

def numerical_gradient_check(dense, input, output_gradient, epsilon = 1e-5)
  weights = dense.instance_variable_get(:@weights)
  analytical_grad = dense.backward(output_gradient, 0)  # learning_rate 0 so weights don't actually update

  # for one specific weight, e.g. weights[0][0]:
  original = weights.get(0, 0)

  weights.set(0, 0, original + epsilon)
  output_plus = dense.forward(input)

  weights.set(0, 0, original - epsilon)
  output_minus = dense.forward(input)

  weights.set(0, 0, original)  # restore

  # approximate d(loss)/d(weight) via finite differences,
  # using output_gradient as a stand-in for d(loss)/d(output)
  numerical_grad = (output_plus.subtract(output_minus)).scalar_divide(2 * epsilon)
  # compare numerical_grad to the weight gradient your backward computed internally
end



if __FILE__ == $0
  puts "Driver initialized :)"

  dense = Dense.new(3, 2)   # 3 inputs, 2 outputs
  input = Matrix.new([[1.0, 2.0, 3.0]])   # 1 sample, 3 features

  output = dense.forward(input)
  puts output.shape   # expect [1, 2]

  fake_gradient = Matrix.new([[0.1, 0.2]])   # same shape as output
  input_gradient = dense.backward(fake_gradient, 0.01)
  puts input_gradient.shape   # expect [1, 3] — must match original input shape
  
  dense = Dense.new(2, 1)
  dense.instance_variable_set(:@weights, Matrix.new([[2.0], [3.0]]))
  dense.instance_variable_set(:@bias, Matrix.new([[1.0]]))

  input = Matrix.new([[1.0, 1.0]])
  output = dense.forward(input)
# expected: (1*2 + 1*3) + 1 = 6
  puts output.get(0, 0)  # should be 6.0

  input1 = Matrix.new([[55, 15]])
  output = dense.forward(input1)

  puts output

  mat = Matrix.init_rand(2,2)
  puts mat
  puts mat.reshape(1, 4)
  

  list1 = [1,2,3,4,5,6,7,8]
  list2 = [8,7,6,5,4,3,2,1]

  puts Matrix.cosine_similarity(list1, list2)
  puts NeuralNetwork.new([])
  puts mat.softmax
end

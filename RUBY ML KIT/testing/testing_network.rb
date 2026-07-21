# Can this network learn the XOR function?
# Dwight Mayer, July 14th, 2026

require_relative "../core/matrix"
require_relative "../neural_network/losses"
require_relative "../neural_network/layers"

require_relative "../neural_network/activations"
require_relative "../neural_network/activation_layers"
require_relative "../neural_network/neural_network"

require_relative "../data/csv_data_source"

require "csv"



=begin
csv_path = File.join(File.dirname(__FILE__), "..", "data", "mnist.csv")
rows = CSV.read(csv_path, headers: true)


#rows = CSV.read("../data/mnist.csv", headers: true)
images = []
labels = []
puts rows.is_a?(Array)


rows.each do |row|
  labels << row[0].to_i

  pixels = row.fields[1..].map(&:to_f)
  images << Matrix.new([pixels])
end

network = NeuralNetwork.new([
    Dense.new(784, 256),
    ReLU.new,

    Dense.new(256, 128),
    ReLU.new,

    Dense.new(128, 10)
]
)

#x_train = Matrix.new(images)
#y_train = Matrix.new([labels])

#puts y_train

total_loss = 0.0

images.each_with_index do |image, i|
  label = Matrix.new([[labels[i]]])   # if CrossEntropy expects a Matrix

  total_loss += network.train_step(image, label, 0.01)
end

#puts total_loss / images.length




images.each_with_index do |x, i|
    y = Matrix.one_hot(labels[i], 10)
    
    
    logits = network.train_step(x, i, 0.01)
    loss = criterion.forward(logits, y)

    gradient = criterion.backward
    network.backward(gradient)
end

=end

csv_path_train = File.join(File.dirname(__FILE__), "..", "data", "mnist_train.csv")
csv_path_test = File.join(File.dirname(__FILE__), "..", "data", "mnist_test.csv")


network_old  = NeuralNetwork.new([
Dense.new(784, 256),
ReLU.new, 
Dense.new(256, 128),
ReLU.new,
Dense.new(128, 10)],
loss:CrossEntropy.new)


network = NeuralNetwork.new([
Dense.new(784, 128),
ReLU.new,
Dense.new(128, 64),
ReLU.new,
Dense.new(64, 10)],
loss:CrossEntropy.new)

train_source = CSVDataSource.new(csv_path_train, 128)
test_source  = CSVDataSource.new(csv_path_test, 128)

network.train_alt(train_source, epochs: 5, learning_rate: 0.015, verbose: true)
def evaluate_accuracy(network, data_source)
  correct = 0
  total = 0

  data_source.each_batch do |batch_input, batch_target|
    predictions = network.predict(batch_input)   # raw logits, shape [batch, 10]

    (0...predictions.rows).each do |r|
      row_values = predictions.get_row(r)
      predicted_class = row_values.each_with_index.max.last   # argmax
      actual_class = batch_target.get(r, 0)

      correct += 1 if predicted_class == actual_class
      total += 1
    end
  end

  correct.to_f / total
end




accuracy = evaluate_accuracy(network, test_source)
puts "test accuracy: #{(accuracy * 100).round(2)}%"







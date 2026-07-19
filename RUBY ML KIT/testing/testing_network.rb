# Can this network learn the XOR function?
# Dwight Mayer, July 14th, 2026

require_relative "matrix"
require_relative "losses"
require_relative "layers"

require_relative "activations"
require_relative "activation_layers"
require_relative "neural_network"


require "csv"

rows = CSV.read("mnist.csv", headers: true)
images = []
labels = []

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



=begin
images.each_with_index do |x, i|
    y = Matrix.one_hot(labels[i], 10)
    
    
    logits = network.train_step(x, i, 0.01)
    loss = criterion.forward(logits, y)

    gradient = criterion.backward
    network.backward(gradient)
end

=end


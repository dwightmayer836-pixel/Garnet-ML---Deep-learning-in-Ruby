# legitimate neural network :)
# July 14th, 2026

# Dwight Mayer

class NeuralNetwork
  def initialize(layers, loss:CrossEntropy.new)
    @layers = layers
    @loss = loss
    puts "Neural Network Initialized"
  end  

  def predict(input)
    @layers.reduce(input) {|output, layer| layer.forward(output)}
  end

  def train_step(input, target, learning_rate)
    prediction = self.predict(input)
    #puts "PREDICTION SHAPE, TARGET SHAPE"
    #puts prediction.shape
    #puts target.shape

    loss_value = @loss.forward(prediction, target)
    gradient = @loss.backward(prediction, target)

    @layers.reverse.each {|layer| gradient=layer.backward(gradient, learning_rate)}
    return loss_value
  end

  def train(input, target, epochs, learning_rate, tolerance:1e-6, verbose:false)
    # Runs a full training loop
    epochs.times do |epoch|
      loss = train_step(input, target, learning_rate)
      puts "epoch #{epoch}: loss = #{loss}" if verbose && epoch % 100 == 0
      if loss < tolerance
        return loss
      end
    end
    return nil
  end

end

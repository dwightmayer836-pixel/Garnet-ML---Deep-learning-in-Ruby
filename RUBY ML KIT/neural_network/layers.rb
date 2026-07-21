# LAYER superclass :) 
# Dwight Mayer, July 13th 2026

class Layer
  def initialize
    @training = true
    @params = {}
    @training_state = {}
    @gradients = {}
  end

  def train!
    @training = true
  end
  def eval!
    @training = false
  end
  def training?
    return @training
  end

  def parameters
    return []
  end
  def forward(input)
    raise NotImplementedError, "Subclass must implement forward"
  end
  def backward(output_gradient, learning_rate)
    raise NotImplementedError, "Subclass must implement backward"
  end

end

class Linear < Layer
  def initialize(input_size, output_size, weights, bias)
    @weights = weights
    @bias = bias
  end
  def parameters
    return [@weights, @bias]
  end
  def forward(input)
    @input = input
    return input.dot_product(@weights).add(@bias)
  end
  def compute_gradients(output_gradient)
    weights_gradient = @input.transpose.dot_product(output_gradient)
    bias_gradient = Matrix.new([output_gradient.sum(axis:0)])
    input_gradient = output_gradient.dot_product(@weights.transpose)
    grads = {}
    grads["weights_grad"] = weights_gradient
    grads["bias_grad"] = bias_gradient
    grads["input_grad"] = input_gradient

    return grads
  end

  def apply_gradients(grads, learning_rate)
    @weights = @weights.subtract(grads["weights_grad"].scalar_multiply(learning_rate))
    @bias = @bias.subtract(grads["bias_grad"].scalar_multiply(learning_rate))
  end

  def backward(output_gradient, learning_rate)
    grads = self.compute_gradients(output_gradient)
    self.apply_gradients(grads, learning_rate)
    return grads["input_grad"]
  end
end

class Dense < Linear
  def initialize(input_size, output_size, initializer: :he)
  
    weights = case initializer
    	      when :he then Matrix.init_he(input_size, output_size)
              when :xavier then Matrix.init_xavier(input_size, output_size)
              when :random then Matrix.init_rand(input_size, output_size)
              else raise ArgumentError, 'unknown initializer'
	      end

    bias = Matrix.new(Matrix.create_zeroes(1, output_size))
    super(input_size, output_size, weights, bias)
  end
end

class Flatten < Layer
  def initialize
    super()
  end
  
  def forward(input)
    @input_shape = input.shape
    batch_size = @input.shape[0]
    feature_size = @input.shape[1..-1].reduce(1,:*)
    input.reshape([batch_size, feature_size])

  end

  def backward(output_gradient, learning_rate)
    output_gradient.reshape(@input_shape)
  end

end


class Dropout < Layer
  # many ways to do dropout; masking, scaling, etc.
  def initialize(rate)
    super()
    unless rate >= 0.0 && rate < 1.0
      raise ArgumentError, "Dropout rate must be in [0, 1)"
    end
    @keep_probability = 1.0 - rate
    @drop_probability = rate
    @mask = nil
    
  end
 
  def forward(input)
    return input unless self.training?
    # create mask, apply mask
    @mask = Matrix.new(Matrix.create_zeroes(input.rows, input.cols)).map do |_|
      rand < @keep_probability ? 1.0 : 0.0
    end
    input.hadamard_multiply(@mask).scalar_divide(@keep_probability)
  end

  def backward(output_gradient, learning_rate=nil)
    return output_gradient unless training?
    return output_gradient.hadamard_multiply(@mask).scalar_divide(@keep_probability)
  end
end



# This whole layer is kinda deprecated at the moment...
# NO TENSOR SUPPORT, JUST MATRICES...
class BatchNormalization < Layer
  def initialize(num_features, epsilon:1e-7, momentum:0.9)
    super()
    @epsilon = epsilon
    @momentum = momentum

    @params[:gamma] = Matrix.new([Array.new(num_features, 1.0)])
    @params[:beta] = Matrix.new([Array.new(num_features, 0.0)])

    @running_mean = Array.new(num_features, 0.0)
    @running_var = Array.new(num_features, 1.0)

    @training_state[:cache] = nil
  end


  def forward(input, training:true)
    if self.training?
      batch_mean = input.mean(axis:0)
      batch_var = input.variance(axis:0)

      @running_mean = @running_mean.each_with_index.map {|r, i| @momentum*r+(1-@momentum)*batch_mean[i]}
      @running_var = @running_var.each_with_index.map {|r,i| @momentum*r+(1-@momentum)*batch_var[i]}
      mean, var=batch_mean,batch_var
    else
      mean, var = @running_mean, @running_var
    end      

    st_dev = var.map {|v| Math.sqrt(v+@epsilon)}
    centered = input.combine(mean, axis:0) {|x, m| x-m}
    normalized = centered.combine(st_dev, axis:0) {|c,s| c/s}

    output = normalized.combine(@params[:gamma], axis:0) {|n,g| n*g}
    output = output.combine(@params[:beta], axis:0) {|o, b| o + b}

    if self.training?
      @training_state[:cache] = {
	normalized:normalized,
	centered:centered,
	st_dev:st_dev,
	batch_size:input.rows
}
    end

    return output

  end


  def backward(output_gradient, learning_rate)
    cache = @training_state[:cache]
    normalized = cache[:normalized]
    centered = cache[:centered]
    st_dev = cache[:st_dev]
    batch_size = cache[:batch_size]

    gamma_grad = output_gradient.hadamard_multiply(normalized).sum(axis:0)
    beta_grad = output_gradient.sum(axis:0)
    @gradients = {gamma: Matrix.new([gamma_grad]), beta: Matrix.new([beta_grad])}
    normalized_gradient = output_gradient.combine(@params[:gamma], axis:0) {|d, g| d * g}

    inv_std_cubed = st_dev.map {|s| -0.5 * (s**-3)}
    var_gradient = normalized_gradient.hadamard_multiply(centered)
	.combine(inv_std_cubed, axis:0) {|v, scale| v * scale}
        .sum(axis:0)    
    neg_inv_std = st_dev.map {|s| -1.0/s}
    direct = normalized_gradient.combine(neg_inv_std, axis:0) {|d,s| d*s}.sum(axis:0)
    centered_sums = centered.sum(axis:0)
    indirect = var_gradient.each_with_index.map {|dv, c| dv * -2.0 * centered_sums[c] / batch_size}
    mean_gradient = direct.each_with_index.map {|d, c| d + indirect[c]}
    term1 = normalized_gradient.combine(st_dev, axis:0) {|d,s| d/s}
    term2 = centered.combine(var_gradient, axis:0) {|dv, c| dv * 2.0 * c / batch_size}
    term3_per_col = mean_gradient.map {|m| m/batch_size}
    input_gradient = term1.add(term2).combine(term3_per_col, axis:0) {|v,m| v+m}

    @params[:gamma] = @params[:gamma].subtract(@gradients[:gamma].scalar_multiply(learning_rate))
    @params[:beta] = @params[:beta].subtract(@gradients[:beta].scalar_multiply(learning_rate))

    return input_gradient
   
  end

end



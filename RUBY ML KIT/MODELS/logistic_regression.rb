# Dwight Mayer, logistic regression implementation, July 10th, 2026

require_relative '../core/matrix'


class LogisticRegression
  def initialize(learning_rate, num_iterations)
    @bias = 0
    @learning_rate = learning_rate
    @num_iterations = num_iterations
    @weights = nil
  end

  def fit(x_train, y_true)
    # x_train n_cols features X n_obsrvations Matrix object
    # y_true n_observations x 1 element in each array
    @weights = Matrix.new(Matrix.create_zeroes(x_train.cols, 1))
    
    for iteration in 0...@num_iterations
      linear_predictions = (x_train.matrix_multiply(@weights)).add(@bias)
      probabilities = linear_predictions.sigmoid
      errors = probabilities.subtract(y_true)
      derivative_weights =  x_train.transpose.matrix_multiply(errors).scalar_multiply(1.0 / x_train.rows)     
      derivative_bias = errors.mean      
      @weights = @weights.subtract(derivative_weights.scalar_multiply(@learning_rate))
      @bias = @bias - (derivative_bias * @learning_rate)
=begin
      if iteration % 1000 == 0
        puts "Iteration #{iteration}"
        puts "Weights:"
        puts @weights
        puts "Bias:"
        puts @bias
        puts "Predictions:"
        puts probabilities
      end
=end     
    end
    return @weights
  end

  def predict(x_mat)
    linear_predictions = (x_mat.matrix_multiply(@weights)).add(@bias)
    probabilities = linear_predictions.sigmoid
    predictions = []
    for i in 0...x_mat.rows
      prob = probabilities.get(i, 0)
      if prob >= 0.50
        predictions.push(1)
      elsif prob < 0.50
        predictions.push(0)
      end
    end
  return predictions
  end

end
    



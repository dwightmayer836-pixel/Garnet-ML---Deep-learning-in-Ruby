# Getting statistical functions for the Matrix object
# Dwight Mayer, July 13th, 2026


module Statistics

  def covariance
    means = self.mean(axis:0)
    covariance_grid = Matrix.create_zeroes(@cols, @cols)
    for i in 0...@cols
      for j in 0...@cols
        covariance_sum = 0
        for row_idx in 0...@rows
          diff_i = self.get(row_idx, i) - means[i]
          diff_j = self.get(row_idx, j) - means[j]
          covariance_sum += (diff_i * diff_j)
        end
        covariance_grid[i][j] = covariance_sum / (@rows-1)
      end
    end
    return Matrix.new(covariance_grid)
  end
  
  def correlation
    covariance_matrix = self.covariance
    st_devs = self.standard_deviation(axis:0)
    correlation_grid = Matrix.create_zeroes(@cols, @cols)

    for i in 0...@cols
      for j in 0...@cols
        covariance_value = covariance_matrix.get(i, j)
        denominator = st_devs[i] * st_devs[j]
        correlation = 0
        unless denominator == 0
          correlation = covariance_value / denominator
        end
        correlation_grid[i][j] = correlation
      end
    end
    return Matrix.new(correlation_grid)
  end

  def each_axis(axis)
    # 0 means collapse down rows, operate on each column COL WISE
    # 1 means collapse down columns, operate on each row ROW WISE

    case axis
    when 0
      return (0...@cols).map {|i| get_col(i)}
    when 1
      return (0...@rows).map { |i| get_row(i) }
    else
      raise ArgumentError, "axis must be 0 or 1"
    end
  end

  def aggregate(axis:nil)
    if axis.nil?
      return yield(flatten)
    end
    each_axis(axis).map do |values|
      yield(values)
    end
  end

  def transform_axis!(axis:nil)
    case axis
    when 0
      for col_idx in 0...@cols
        values = self.get_col(col_idx)
        transformed = yield(values)
        self.set_col!(col_idx, transformed)
      end
    when 1
      for row_idx in 0...@rows
        values = self.get_row(row_idx)
        transformed = yield(values)
        self.set_row!(row_idx, transformed)
      end

    else
      raise ArgumentError, "invalid axis for transformation"
    end
    return self
  end

  def mean(axis:nil)
    aggregate(axis:axis) {|values| Matrix.vector_mean(values)}
  end

  def max(axis:nil)
    aggregate(axis:axis) {|values| values.max}
  end

  def min(axis:nil)
   aggregate(axis:axis) {|values| values.min}
  end

  def median(axis:nil)
    aggregate(axis:axis) {|values| Matrix.vector_median(values)}
  end

  def variance(axis:nil)
    aggregate(axis:axis) {|values| Matrix.vector_variance(values)}
  end

  def standard_deviation(axis:nil)
    aggregate(axis:axis) {|values| Matrix.vector_standard_deviation(values)}
  end

  def sum(axis:nil)
    aggregate(axis:axis) {|values| Matrix.vector_sum(values)}
  end

  def product(axis:nil)
    aggregate(axis:axis) {|values| Matrix.vector_product(values)}
  end
  def argmin(axis:nil)
    aggregate(axis:axis) {|values| Matrix.vector_argmin(values)}
  end

  def argmax(axis:nil)
    aggregate(axis:axis) {|values| Matrix.vector_argmax(values)}
  end

  def standardize(axis:0)
    self.transform_axis!(axis:axis) {|values| Matrix.vector_standardize(values)}
  end

  def quantile(q, axis:nil)
    aggregate(axis:axis) {|values| Matrix.vector_quantile(values, q)}
  end

  def normalize(axis:nil)
    # aggregation logic
    self.transform_axis!(axis:axis) {|values| Matrix.vector_normalize(values)}

  end

  def mean_squared_error(y_pred, y_true)
    errors = y_true.subtract(y_pred)
    errors = errors.hadamard_multiply(errors)
    return Matrix.vector_mean(errors.flatten)
  end  


end





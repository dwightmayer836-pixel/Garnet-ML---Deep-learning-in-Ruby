# Vector ops module, July 13th, 2026


module VectorOperations

  def vector_dot_product(vector1, vector2)
    unless vector1.length == vector2.length
      raise ArgumentError, "vector length mismatch in dot product"
    end
    sum = 0
    vector1.each_index { |i| sum += (vector1[i] * vector2[i])}
    return sum
  end
  
  def vector_median(values)
    return self.vector_quantile(values, 0.5)    
  end

  def vector_quantile(values, quantile)
    sorted = values.sort
    continuous_index = (sorted.length-1) * quantile
    lower_idx = continuous_index.floor
    upper_idx = continuous_index.ceil
    fraction = continuous_index - lower_idx
    if lower_idx == upper_idx
      return sorted[lower_idx]
    else
      lower_value = sorted[lower_idx]
      upper_value = sorted[upper_idx]
      return lower_value + fraction * (upper_value - lower_value)
    end    
  end

  def vector_mean(values)
    return values.sum.to_f / values.length
  end

  def vector_variance(values)
    mean = values.sum.to_f / values.length
    return values.sum { |value| (value - mean) ** 2 } / values.length
  end

  def vector_standard_deviation(values)
    return Math.sqrt(self.vector_variance(values)) 
  end

  def vector_zero?(values)
    return false if values.include?(0)
    return true
  end

  def vector_argmin(values)
    return values.each_with_index.min.last
  end

  def vector_argmax(values)
    return values.each_with_index.max.last
  end

  def vector_sum(values)
    return values.sum
  end  
   
  def vector_product(values)
    return values.inject(1) {|prod, value| prod * value}
  end

  def vector_standardize(values)
    # returns the standardized versions of the vector
    st_dev = self.vector_standard_deviation(values)
    mean = self.vector_mean(values)

    unless st_dev != 0
      return Array.new(values.length, 0.0)
    end
    
    # values.map{|value| ((value-mean)/st_dev)}

    new_vector = []
    for value in values
      standardized = (value - mean) / st_dev
      new_vector.push(standardized)
    end
    return new_vector
  end

  def create_column_vector(values)
    return Matrix.build(values.length, 1) {|row, col| values[row]}
  end

  def vector_magnitude(values)
    sum = 0
    for value in values
      sum += value**2
    end
    return Math.sqrt(sum)
  end

  def cosine_similarity(values, other)
    ab = self.vector_dot_product(values, other)
    a_mag = self.vector_magnitude(values)
    b_mag = self.vector_magnitude(other)
    return ab / (a_mag * b_mag)
  
  end

  def vector_normalize(values)
    # X new = (x - xmin) / (xmax-xmin)
    xmax = values.max
    xmin = values.min
    return Array.new(values.length, 0.0) if xmax == xmin
    return values.map {|x| (x-xmin) / (xmax-xmin)}

  end

end

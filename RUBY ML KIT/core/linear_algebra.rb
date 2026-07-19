# Partitioning out LinALG functionality
# July 13th, 2026 -> Dwight Mayer


module LinearAlgebra

  def reduced_row_echelon_form
    working = self.copy
    pivot_row = 0
    for pivot_col in 0...@cols
      break if pivot_row >= @rows

      # find a row at / below pivot row w/ nonzero value
      target_row = nil
      for row_idx in pivot_row...@rows
        if working.get(row_idx, pivot_col) != 0
          target_row = row_idx
          break
        end
      end
      # entire column from pivot row down is zero...
      next if target_row.nil?
      working.swap_rows!(pivot_row, target_row) if target_row != pivot_row
      pivot_value = working.get(pivot_row, pivot_col)
      working.scale_row!(pivot_row, 1.0 / pivot_value)

      for idx in 0...@rows
        next if idx == pivot_row
        factor = working.get(idx, pivot_col).to_f / working.get(
							pivot_row, pivot_col)
        working.add_scaled_row!(idx, pivot_row, -factor)
      end
      pivot_row +=1
    end
    return working
  end

  def inverse
    # gets the inverse of a matrix...
    self.validate_square

    # create augmented matrix, reduce left to identity
    identity_mat = Matrix.identity(@rows)
    augmented = self.hstack(identity_mat)
    reduced_mat = augmented.reduced_row_echelon_form

    # check if left IS identity, then slices off the right half of the matrix
    left_half = reduced_mat.slice(row_start:0, row_end:@rows-1,col_start:0, col_end:@rows-1)
    if not left_half.equals?(Matrix.identity(@rows))
      raise ArgumentError, "matrix is singular, not invertible"
    end
    inverse_mat = reduced_mat.slice(
      row_start:0,row_end:@rows-1, col_start:@rows, col_end:(2 * @rows - 1))
    return inverse_mat

  end
  
  def solve(other)
    self.validate_square
    unless @rows == other.rows
      raise ArgumentError, "bad row in matrix solve"
    end

    wide_mat = self.hstack(other)
    reduced_mat = wide_mat.reduced_row_echelon_form

    left_half = reduced_mat.slice(
      row_start:0, row_end:@rows-1, col_start:0, col_end:@cols-1)

    unless left_half.equals?(Matrix.identity(@rows))
      raise ArgumentError, "matrix is singular, system has NO unique solution"
    end

    # last column of the fully reduced matrix is x
    x = reduced_mat.slice(
      row_start:0, row_end:@rows-1, col_start:@cols, col_end:reduced_mat.cols-1)
    return x

  end

  def rank
    rank = 0
    reduced = self.reduced_row_echelon_form
    for i in 0...reduced.rows
      current_row = reduced.get_row(i)
      unless Matrix.vector_zero?(current_row)
        rank +=1
      end
    end
    return rank
  end

  def determinant
    self.validate_square
    # creates copy and stores sign
    working = self.copy
    sign = 1

    for pivot in 0...@rows
      # find pivot
      pivot_row = pivot
      while pivot_row < @rows and working.get(pivot_row, pivot) == 0
        pivot_row += 1
      end
      if pivot_row == @rows
        return 0
      end
      if pivot_row != pivot
        working.swap_rows!(pivot, pivot_row)
        sign *= -1
      end
      for row_idx in (pivot+1)...@rows
        factor = working.get(row_idx, pivot).to_f  / working.get(pivot, pivot)
        for col in pivot...@cols
          scaled = working.get(pivot, col) * factor
          new_val = working.get(row_idx, col)
          working.set(row_idx, col, (new_val - scaled))
        end
      end
    end
  det = sign
  for i in 0...@rows
    det *= working.get(i, i)
  end
  return det
  end

  def trace
    self.validate_square
    sum = 0
    for idx in 0...@rows
      sum += self.get(idx, idx)
    end
    return sum
  end
end


import numpy as np

def rolling_window(a, window):
   """
   Make an ndarray with a rolling window of the last dimension

   Parameters
   ----------
   a : array_like
       Array to add rolling window to
   window : int
       Size of rolling window

   Returns
   -------
   Array that is a view of the original array with a added dimension
   of size w.

   Examples
   --------
   >>> x=np.arange(10).reshape((2,5))
   >>> rolling_window(x, 3)
   array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
          [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

   Calculate rolling mean of last dimension:
   >>> np.mean(rolling_window(x, 3), -1)
   array([[ 1.,  2.,  3.],
          [ 6.,  7.,  8.]])

   """
   if window < 1:
       raise ValueError("`window` must be at least 1.")
   if window > a.shape[-1]:
       raise ValueError("`window` is too long.")
   shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
   strides = a.strides + (a.strides[-1],)
   return np.ascontiguousarray(np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides))

x=np.arange(10).reshape((2,5))
temp = rolling_window(x, 3)
print(temp)
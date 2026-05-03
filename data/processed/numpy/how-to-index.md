---
title: "How to index ndarrays — NumPy v2.4 Manual"
source_url: https://numpy.org/doc/stable/user/how-to-index.html
---

# How to index `ndarrays`

See also

Indexing on ndarrays

This page tackles common examples. For an in-depth look into indexing, refer
to Indexing on ndarrays.

## Access specific/arbitrary rows and columns

Use Basic indexing features like Slicing and striding, and
Dimensional indexing tools.

```
>>> a = np.arange(30).reshape(2, 3, 5)
>>> a
array([[[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14]],

        [[15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29]]])
>>> a[0, 2, :]
array([10, 11, 12, 13, 14])
>>> a[0, :, 3]
array([ 3,  8, 13])
```

Note that the output from indexing operations can have different shape from the
original object. To preserve the original dimensions after indexing, you can
use `newaxis`. To use other such tools, refer to
Dimensional indexing tools.

```
>>> a[0, :, 3].shape
(3,)
>>> a[0, :, 3, np.newaxis].shape
(3, 1)
>>> a[0, :, 3, np.newaxis, np.newaxis].shape
(3, 1, 1)
```

Variables can also be used to index:

```
>>> y = 0
>>> a[y, :, y+3]
array([ 3,  8, 13])
```

Refer to Dealing with variable numbers of indices within programs to see how to use
slice and `Ellipsis` in your index variables.

### Index columns

To index columns, you have to index the last axis. Use
Dimensional indexing tools to get the desired number of dimensions:

```
>>> a = np.arange(24).reshape(2, 3, 4)
>>> a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>> a[..., 3]
array([[ 3,  7, 11],
       [15, 19, 23]])
```

To index specific elements in each column, make use of Advanced indexing
as below:

```
>>> arr = np.arange(3*4).reshape(3, 4)
>>> arr
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> column_indices = [[1, 3], [0, 2], [2, 2]]
>>> np.arange(arr.shape[0])
array([0, 1, 2])
>>> row_indices = np.arange(arr.shape[0])[:, np.newaxis]
>>> row_indices
array([[0],
       [1],
       [2]])
```

Use the `row_indices` and `column_indices` for advanced
indexing:

```
>>> arr[row_indices, column_indices]
array([[ 1,  3],
       [ 4,  6],
       [10, 10]])
```

### Index along a specific axis

Use `take`. See also `take_along_axis` and
`put_along_axis`.

```
>>> a = np.arange(30).reshape(2, 3, 5)
>>> a
array([[[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14]],

        [[15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29]]])
>>> np.take(a, [2, 3], axis=2)
array([[[ 2,  3],
        [ 7,  8],
        [12, 13]],

        [[17, 18],
        [22, 23],
        [27, 28]]])
>>> np.take(a, [2], axis=1)
array([[[10, 11, 12, 13, 14]],

        [[25, 26, 27, 28, 29]]])
```

## Create subsets of larger matrices

Use Slicing and striding to access chunks of a large array:

```
>>> a = np.arange(100).reshape(10, 10)
>>> a
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
        [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
        [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
>>> a[2:5, 2:5]
array([[22, 23, 24],
       [32, 33, 34],
       [42, 43, 44]])
>>> a[2:5, 1:3]
array([[21, 22],
       [31, 32],
       [41, 42]])
>>> a[:5, :5]
array([[ 0,  1,  2,  3,  4],
       [10, 11, 12, 13, 14],
       [20, 21, 22, 23, 24],
       [30, 31, 32, 33, 34],
       [40, 41, 42, 43, 44]])
```

The same thing can be done with advanced indexing in a slightly more complex
way. Remember that
advanced indexing creates a copy:

```
>>> a[np.arange(5)[:, None], np.arange(5)[None, :]]
array([[ 0,  1,  2,  3,  4],
       [10, 11, 12, 13, 14],
       [20, 21, 22, 23, 24],
       [30, 31, 32, 33, 34],
       [40, 41, 42, 43, 44]])
```

You can also use `mgrid` to generate indices:

```
>>> indices = np.mgrid[0:6:2]
>>> indices
array([0, 2, 4])
>>> a[:, indices]
array([[ 0,  2,  4],
       [10, 12, 14],
       [20, 22, 24],
       [30, 32, 34],
       [40, 42, 44],
       [50, 52, 54],
       [60, 62, 64],
       [70, 72, 74],
       [80, 82, 84],
       [90, 92, 94]])
```

## Filter values

### Non-zero elements

Use `nonzero` to get a tuple of array indices of non-zero elements
corresponding to every dimension:

```
>>> z = np.array([[1, 2, 3, 0], [0, 0, 5, 3], [4, 6, 0, 0]])
>>> z
array([[1, 2, 3, 0],
       [0, 0, 5, 3],
       [4, 6, 0, 0]])
>>> np.nonzero(z)
(array([0, 0, 0, 1, 1, 2, 2]), array([0, 1, 2, 2, 3, 0, 1]))
```

Use `flatnonzero` to fetch indices of elements that are non-zero in
the flattened version of the ndarray:

```
>>> np.flatnonzero(z)
array([0, 1, 2, 6, 7, 8, 9])
```

### Arbitrary conditions

Use `where` to generate indices based on conditions and then
use Advanced indexing.

```
>>> a = np.arange(30).reshape(2, 3, 5)
>>> indices = np.where(a % 2 == 0)
>>> indices
(array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
array([0, 0, 0, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2]),
array([0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3]))
>>> a[indices]
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
```

Or, use Boolean array indexing:

```
>>> a > 14
array([[[False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]],

       [[ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True]]])
>>> a[a > 14]
array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
```

### Replace values after filtering

Use assignment with filtering to replace desired values:

```
>>> p = np.arange(-10, 10).reshape(2, 2, 5)
>>> p
array([[[-10,  -9,  -8,  -7,  -6],
        [ -5,  -4,  -3,  -2,  -1]],

       [[  0,   1,   2,   3,   4],
        [  5,   6,   7,   8,   9]]])
>>> q = p < 0
>>> q
array([[[ True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True]],

       [[False, False, False, False, False],
        [False, False, False, False, False]]])
>>> p[q] = 0
>>> p
array([[[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]],

       [[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]]])
```

## Fetch indices of max/min values

Use `argmax` and `argmin`:

```
>>> a = np.arange(30).reshape(2, 3, 5)
>>> np.argmax(a)
29
>>> np.argmin(a)
0
```

Use the `axis` keyword to get the indices of maximum and minimum
values along a specific axis:

```
>>> np.argmax(a, axis=0)
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
>>> np.argmax(a, axis=1)
array([[2, 2, 2, 2, 2],
       [2, 2, 2, 2, 2]])
>>> np.argmax(a, axis=2)
array([[4, 4, 4],
       [4, 4, 4]])

>>> np.argmin(a, axis=1)
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])
>>> np.argmin(a, axis=2)
array([[0, 0, 0],
       [0, 0, 0]])
```

Set `keepdims` to `True` to keep the axes which are reduced in the
result as dimensions with size one:

```
>>> np.argmin(a, axis=2, keepdims=True)
array([[[0],
        [0],
        [0]],

       [[0],
        [0],
        [0]]])
>>> np.argmax(a, axis=1, keepdims=True)
array([[[2, 2, 2, 2, 2]],

       [[2, 2, 2, 2, 2]]])
```

To get the indices of each maximum or minimum value for each
(N-1)-dimensional array in an N-dimensional array, use `reshape`
to reshape the array to a 2D array, apply `argmax` or `argmin`
along `axis=1` and use `unravel_index` to recover the index of the
values per slice:

```
>>> x = np.arange(2*2*3).reshape(2, 2, 3) % 7  # 3D example array
>>> x
array([[[0, 1, 2],
        [3, 4, 5]],

       [[6, 0, 1],
        [2, 3, 4]]])
>>> x_2d = np.reshape(x, (x.shape[0], -1))
>>> indices_2d = np.argmax(x_2d, axis=1)
>>> indices_2d
array([5, 0])
>>> np.unravel_index(indices_2d, x.shape[1:])
(array([1, 0]), array([2, 0]))
```

The first array returned contains the indices along axis 1 in the original
array, the second array contains the indices along axis 2. The highest
value in `x[0]` is therefore `x[0, 1, 2]`.

## Index the same ndarray multiple times efficiently

It must be kept in mind that basic indexing produces views
and advanced indexing produces copies, which are
computationally less efficient. Hence, you should take care to use basic
indexing wherever possible instead of advanced indexing.

## Further reading

Nicolas Rougier’s 100 NumPy exercises
provide a good insight into how indexing is combined with other operations.
Exercises 6, 8, 10, 15, 16, 19, 20, 45, 59,
64, 65, 70, 71, 72, 76, 80, 81, 84, 87, 90,
93, 94 are specially focused on indexing.
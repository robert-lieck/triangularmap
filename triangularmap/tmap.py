#  Copyright (c) 2022 Robert Lieck.

import re
from copy import deepcopy
from collections import defaultdict

import torch
import numpy as np


class TMap:
    """
    A wrapper around a 1D array that provides access in triangular (or ternary) coordinates. A 1D array of length
    :math:`N = n (n + 1) / 2` is mapped to a triangular layout as follows (here with :math:`N=21` and :math:`n=6`):

    ::

                               /\\         depth  level
                              /0 \\            0      6
                             /\\  /\\
                            /1 \\/ 2\\          1      5
                           /\\  /\\  /\\
                          /3 \\/4 \\/5 \\        2      4
                         /\\  /\\  /\\  /\\
                        /6 \\/7 \\/8 \\/9 \\      3      3
                       /\\  /\\  /\\  /\\  /\\
                      /10\\/11\\/12\\/13\\/14\\    4      2
                     /\\  /\\  /\\  /\\  /\\  /\\
                    /15\\/16\\/17\\/18\\/19\\/20\\  5      1
                   |   |   |   |   |   |   |
        start/end: 0   1   2   3   4   5   6

    Values can be accessed by start and end index (:math:`0 \leq start < end \leq n`) as follows:

    ::

                       (0, 6)
                    (0, 5) (1, 6)
                 (0, 4) (1, 5) (2, 6)
              (0, 3) (1, 4) (2, 5) (3, 6)
           (0, 2) (1, 3) (2, 4) (3, 5) (4, 6)
        (0, 1) (1, 2) (2, 3) (3, 4) (4, 5) (5, 6)

    That is (start, end) is mapped to the linear index :math:`depth (depth + 1) / 2 + end - level`,
    where :math:`depth = n - (end - start)` and :math:`level = n - depth`. Advanced integer index arrays are
    processed in the same way and
    are applied to the underlying array following standard numpy rules (e.g. direct assignment works but otherwise a
    copy of the values is returned instead of a view). Additionally, slices by depth or level return views of the
    underlying array segment. Slices by start or end index are also supported but internally use advanced indexing, so
    they return a copy, not a view.
    """

    _flatten_regex = re.compile("^(?P<outer_sign>[+-]?)(?P<outer>[sel])(?P<inner_sign>[+-]?)(?P<inner>[sel])$")

    class UnDef:
        """Class to indicate undefined indices"""
        pass

    class GetSetWrapper:
        """
        Wrapper class that delegates __getitem__ and __setitem__ to custom functions
        """
        def __init__(self, getter, setter):
            self.getter = getter
            self.setter = setter

        def __getitem__(self, item):
            return self.getter(item)

        def __setitem__(self, key, value):
            self.setter(key, value)

    @classmethod
    def _to_int(cls, i):
        """
        Convert i to integer, no matter whether it is a single number or a numpy array.

        :param i: number or array of numbers
        :return: integer or array of integers
        """
        if isinstance(i, np.ndarray):
            return i.astype(int)
        else:
            return int(i)

    @classmethod
    def _unpack_item(cls, item):
        if isinstance(item, tuple):
            return item[0], item[1:]
        else:
            return item, cls.UnDef

    @classmethod
    def size_from_n(cls, n):
        """
        Calculate the size `N` of the underlying 1D array for a given width ``n`` of the triangular map:
        :math:`N = n (n + 1)) / 2`. This function also works with arrays.

        :param n: Width (number of entries at the bottom of the map)
        :return: Length of underlying 1D array (total number of entries in the map)
        """
        return cls._to_int((n * (n + 1)) / 2)

    @classmethod
    def n_from_size(cls, n):
        """
        Calculate width ``n`` of the map given the size `N` of the underlying 1D array:
        :math:`n = (\\sqrt{8 * N + 1} - 1) / 2`.
        Checks for valid size (i.e. if the resulting n is actually an integer) and raises a ValueError otherwise.
        This function also works with arrays.

        :param n: size of the underlying 1D array
        :return: width of the map
        """
        n_ = (np.sqrt(8 * n + 1) - 1) / 2
        if cls.size_from_n(np.floor(n_)) != n:
            raise ValueError(f"{n} is not a valid size for a triangular map (n={n_})")
        return cls._to_int(np.floor(n_))

    @classmethod
    def get_reindex_from_start_end_to_top_down(cls, n):
        """
        For a map of width ``n``, get an index array of length :math:`n (n + 1)) / 2` to reindex from start-end order to
        top-down order. This is used in :func:`~triangularmap.TMap.reindex_from_start_end_to_top_down`
        to perform the reindexing.

        :param n: width of map
        :return: index array of length :math:`n (n + 1)) / 2`
        """
        n_elem = np.arange(n + 1)
        idx_shift = n * (n + 1) // 2 - np.flip(n_elem * (n_elem + 1) // 2)
        index_list = []
        for idx in range(1, n + 1):
            index_list.append(idx_shift[:idx] + (n - idx))
        return np.concatenate(index_list)

    @classmethod
    def reindex_from_start_end_to_top_down(cls, arr):
        """
        Reindex ``arr`` from start-end order to top-down order.

        :param arr: linear array in start-end order
        :return: linear array in top-down order

        Given a linear array `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`, assuming start-end order translates to the triangular
        map:

        ::

                    ╱╲
                   ╱ 3╲
                  ╱╲  ╱╲
                 ╱ 2╲╱ 6╲
                ╱╲  ╱╲  ╱╲
               ╱ 1╲╱ 5╲╱ 8╲
              ╱╲  ╱╲  ╱╲  ╱╲
             ╱ 0╲╱ 4╲╱ 7╲╱ 9╲

        Assuming top-down order translates into:

        ::

                    ╱╲
                   ╱ 0╲
                  ╱╲  ╱╲
                 ╱ 1╲╱ 2╲
                ╱╲  ╱╲  ╱╲
               ╱ 3╲╱ 4╲╱ 5╲
              ╱╲  ╱╲  ╱╲  ╱╲
             ╱ 6╲╱ 7╲╱ 8╲╱ 9╲

        Applying :func:`~triangularmap.TMap.reindex_from_start_end_to_top_down` to the array
        `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` reorders it to `[3, 2, 6, 1, 5, 8, 0, 4, 7, 9]`, so that the upper
        triangular map is produced using the lower ordering principle. This is the inverse function of
        :func:`~triangularmap.TMap.reindex_from_top_down_to_start_end`.
        """
        n = TMap.n_from_size(arr.shape[0])
        return arr[cls.get_reindex_from_start_end_to_top_down(n)]

    @classmethod
    def get_reindex_from_top_down_to_start_end(cls, n):
        """
        For a map of width ``n``, get an index array of length :math:`n (n + 1)) / 2` to reindex from top-down order to
        start-end order. This is used in :func:`~triangularmap.TMap.reindex_from_top_down_to_start_end`
        to perform the reindexing.

        :param n: width of map
        :return: index array of length :math:`n (n + 1)) / 2`
        """
        n_elem = np.arange(n + 1)
        sum_elem = n_elem * (n_elem + 1) // 2
        index_list = []
        for idx in range(n):
            index_list.append(np.flip(sum_elem[idx:n] + idx))
        return np.concatenate(index_list)

    @classmethod
    def reindex_from_top_down_to_start_end(cls, arr):
        """
        Reindex ``arr`` from top-down order to start-end order.

        :param arr: linear array in top-down order
        :return: linear array in start-end order

        Given a linear array `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`, assuming top-down order translates to the triangular
        map:

        ::

                    ╱╲
                   ╱ 0╲
                  ╱╲  ╱╲
                 ╱ 1╲╱ 2╲
                ╱╲  ╱╲  ╱╲
               ╱ 3╲╱ 4╲╱ 5╲
              ╱╲  ╱╲  ╱╲  ╱╲
             ╱ 6╲╱ 7╲╱ 8╲╱ 9╲

        Assuming start-end order translates into:

        ::

                    ╱╲
                   ╱ 3╲
                  ╱╲  ╱╲
                 ╱ 2╲╱ 6╲
                ╱╲  ╱╲  ╱╲
               ╱ 1╲╱ 5╲╱ 8╲
              ╱╲  ╱╲  ╱╲  ╱╲
             ╱ 0╲╱ 4╲╱ 7╲╱ 9╲

        Applying :func:`~triangularmap.TMap.reindex_from_top_down_to_start_end` to the array
        `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` reorders it to `[6, 3, 1, 0, 7, 4, 2, 8, 5, 9]`, so that the upper
        triangular map is produced using the lower ordering principle. This is the inverse function of
        :func:`~triangularmap.TMap.reindex_from_start_end_to_top_down`
        """
        n = TMap.n_from_size(arr.shape[0])
        return arr[cls.get_reindex_from_top_down_to_start_end(n)]

    def __init__(self, arr, linearise_blocks=False, _n=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _n is None:
            self._n = self.n_from_size(len(arr))
        else:
            self._n = _n
        self._arr = arr
        try:
            self._value_shape = self._arr.shape[1:]
        except AttributeError:
            self._value_shape = ()
        self._linearise_blocks = linearise_blocks
        self._lslice = TMap.GetSetWrapper(getter=self.get_lslice, setter=self.set_lslice)
        self._dslice = TMap.GetSetWrapper(getter=self.get_dslice, setter=self.set_dslice)
        self._sslice = TMap.GetSetWrapper(getter=self.get_sslice, setter=self.set_sslice)
        self._eslice = TMap.GetSetWrapper(getter=self.get_eslice, setter=self.set_eslice)
        self._sblock = TMap.GetSetWrapper(getter=self.get_sblock, setter=self.set_sblock)
        self._eblock = TMap.GetSetWrapper(getter=self.get_eblock, setter=self.set_eblock)

    @property
    def n(self):
        """
        Size of the triangular map, i.e., the width at the triangle base. This is not the total size of the
        underlying storage (use :meth:`~size_from_n` to convert to that).
        """
        return self._n

    @property
    def arr(self):
        """
        Underlying storage container.
        """
        return self._arr

    @arr.setter
    def arr(self, value):
        raise AttributeError("Can't set attribute 'arr', which is read-only. Use the underlying '_arr' attribute if "
                             "you know what you are doing...")

    @property
    def value_shape(self):
        """
        Shape of the values stored in the map. If the underlying array has more than one dimension, the first one is
        used for the map and :meth:`~value_shape` corresponds to the remaining dimensions.
        """
        return self._value_shape

    @property
    def linearise_blocks(self):
        return self._linearise_blocks

    @property
    def lslice(self):
        """
        A convenience property that uses :meth:`get_lslice` and :meth:`set_lslice` as getter and setter, respectively.
        """
        return self._lslice

    @property
    def dslice(self):
        """
        A convenience property that uses :meth:`get_dslice` and :meth:`set_dslice` as getter and setter, respectively.
        """
        return self._dslice

    @property
    def sslice(self):
        """
        A convenience property that uses :meth:`get_sslice` and :meth:`set_sslice` as getter and setter, respectively.
        """
        return self._sslice

    @property
    def eslice(self):
        """
        A convenience property that uses :meth:`get_eslice` and :meth:`set_eslice` as getter and setter, respectively.
        """
        return self._eslice

    @property
    def sblock(self):
        """
        A convenience property that uses :meth:`get_sblock` and :meth:`set_sblock` as getter and setter, respectively.
        """
        return self._sblock

    @property
    def eblock(self):
        """
        A convenience property that uses :meth:`get_eblock` and :meth:`set_eblock` as getter and setter, respectively.
        """
        return self._eblock

    def _is_pytorch(self):
        return isinstance(self.arr, torch.Tensor)

    def _check(self, start, end):
        """
        Check whether 0 <= start < end < n. This function also works with arrays, in which case all start/end values
        have to pass the check. If the check is not passed, an IndexError is raised

        :param start: start index or array of indices
        :param end: end index or array of indices
        """
        if isinstance(start, np.ndarray) or isinstance(end, np.ndarray):
            c = start < end
            c = np.logical_and(c, 0 <= start)
            c = np.logical_and(c, end <= self.n)
            do_raise = not np.all(c)
        else:
            do_raise = not (0 <= start < end <= self.n)
        if do_raise:
            raise IndexError(f"Invalid indices for TMap with size n={self.n}\nstart: {start}\nend: {end}")

    def depth(self, start, end):
        """
        Compute the depth d corresponding to (start, end): d = n - (end - start). This function also works with arrays.

        :param start: start index or array of indices
        :param end: end index or array of indices
        :return: depth or array of depth values
        """
        return self._to_int(self.n - (end - start))

    def level(self, *args):
        """
        Compute the level from depth or (start, end). If (start, end) is given, the depth function is used to first
        compute the depth. This function also works with arrays.

        :param args: either one argument (depth) or two (start, end), which can also be arrays
        :return: depth or array of depth values
        """
        if len(args) == 1:
            depth = args[0]
        elif len(args) == 2:
            depth = self.depth(*args)
        else:
            raise TypeError(f"Expected one positional argument (depth) or two (start, end) to compute level but got "
                            f"{len(args)}")
        return self.n - depth

    def linear_from_start_end(self, start, end):
        """
        Compute the linear index (in the underlying 1D array) corresponding to a (start, end) pair. This function also
        works with arrays.

        :param start: start index or array of indices
        :param end: end index or array of indices
        :return: linear index or array of linear indices
        """
        depth = self.depth(start, end)
        level = self.level(depth)
        return self._to_int(depth * (depth + 1) / 2 + end - level)

    def __getitem__(self, item):
        """
        Get the item corresponding to (start, end) or the sub-map corresponding to slice start:end. For items, this
        function also works with arrays, resulting in advanced indexing of the underlying 1D array.

        :param item: (start, end) or slice
        :return: element or sub-map
        """
        if isinstance(item, slice):
            # return sub-map
            # function to concatenate slices
            if isinstance(self.arr, np.ndarray):
                cat = np.concatenate
            elif self._is_pytorch():
                cat = torch.cat
            else:
                cat = lambda ls: sum(ls, [])
            # star/end and depth to start with
            start = item.start
            end = item.stop
            step = item.step
            start_depth = self.depth(start, end)
            # get new array of sub-map by concatenating slices
            arr_tuple = tuple(
                self.dslice[d][slice(start, start + (d - start_depth) + 1, step)] for d in range(start_depth, self.n)
            )
            arr = cat(arr_tuple)
            # return new TMap
            return TMap(arr, linearise_blocks=self.linearise_blocks)
        else:
            # return element
            start, end = item
            self._check(start, end)
            linear_idx = self.linear_from_start_end(start, end)
            return self.arr[linear_idx]

    def __setitem__(self, key, value):
        """
        Set the item corresponding to (start, end). This function also works with arrays, resulting in advanced indexing
        of the underlying 1D array.

        :param key: (start, end)
        :value: value to set
        """
        start, end = key
        self._check(start, end)
        linear_idx = self.linear_from_start_end(start, end)
        self.arr[linear_idx] = value

    def copy(self):
        """
        Copy the map. If the underlying data is a lists, tuple, numpy array or pytorch tensor, the appropriate functions
        are called, otherwise a deepcopy of the data is made.

        :return: Copied map.
        """
        if isinstance(self.arr, np.ndarray):
            copy = self.arr.copy()
        elif self._is_pytorch():
            copy = self.arr.detach().clone()
        elif isinstance(self.arr, list):
            copy = list(self.arr)
        elif isinstance(self.arr, tuple):
            copy = tuple(self.arr)
        else:
            copy = deepcopy(self.arr)
        return TMap(arr=copy, linearise_blocks=self.linearise_blocks)

    def top(self, depth=None):
        """
        Return the sub-map corresponding to the top-most levels. A view of the underlying data is used, so the returned
        TMap shares the same buffer and modification affect both objects.

        :param depth: How many levels from the top to include
        :return: sub-map
        """
        if depth is None:
            return self
        else:
            start, end = self.linear_start_end_from_level(self.level(depth - 1))
            return TMap(self.arr[:end + 1], linearise_blocks=self.linearise_blocks)

    def linear_start_end_from_level(self, level):
        """
        Compute the linear 1D start and end index corresponding to all values in the respective level of the map.
        Slicing the underlying array as arr[start:end + 1] will return a view of the values on the level.

        :param level: level for which to compute the indices
        :return: linear 1D start and end index
        """
        linear_start = self.linear_from_start_end(0, level)
        linear_end = self.linear_from_start_end(self.n - level, self.n)
        return linear_start, linear_end

    def get_lslice(self, level):
        """
        Slice the map at the given level, returning a view of the values.

        :param level: level to use for slicing
        :return: view of the values
        """
        linear_start, linear_end = self.linear_start_end_from_level(level)
        return self.arr[linear_start:linear_end + 1]

    def set_lslice(self, level, value):
        linear_start, linear_end = self.linear_start_end_from_level(level)
        self.arr[linear_start:linear_end + 1] = value

    def get_dslice(self, depth):
        """
        Slice the map at the given depth, returning a view of the values.

        :param depth: depth to use for slicing
        :return: view of the values
        """
        linear_start, linear_end = self.linear_start_end_from_level(self.level(depth))
        return self.arr[linear_start:linear_end + 1]

    def set_dslice(self, depth, value):
        linear_start, linear_end = self.linear_start_end_from_level(self.level(depth))
        self.arr[linear_start:linear_end + 1] = value

    def _end_indices_for_sslice(self, start):
        """
        Compute the end indices corresponding to a slice at the give start index.

        :param start: start index
        :return: integer array of end indices
        """
        return np.arange(start + 1, self.n + 1)

    def _start_indices_for_eslice(self, end):
        """
        Compute the start indices corresponding to a slice at the give end index.

        :param end: end index
        :return: integer array of start indices
        """
        return np.arange(0, end)

    def get_sslice(self, item):
        """
        Return a slice for the given start index. Internally, advanced indexing is used, so the returned values are
        a copy, not a view. item can be a tuple to further slice down before retrieving the values.

        :param item: start index or tuple of start index and additional indices/slices
        :return: copy of slice at start index
        """
        start, s = self._unpack_item(item)
        end_indices = self._end_indices_for_sslice(start)
        if s is not self.UnDef:
            end_indices = end_indices[s]
        return self[start, end_indices]

    def set_sslice(self, key, value):
        """
        Like get_sslice but set value instead of returning values.
        """
        start, s = self._unpack_item(key)
        end_indices = self._end_indices_for_sslice(start)
        if s is not self.UnDef:
            end_indices = end_indices[s]
        self[start, end_indices] = value

    def get_eslice(self, item):
        """
        Return a slice for the given end index. Internally, advanced indexing is used, so the returned values are
        a copy, not a view. item can be a tuple to further slice down before retrieving the values.

        :param item: end index or tuple of end index and additional indices/slices
        :return: copy of slice at end index
        """
        end, s = self._unpack_item(item)
        start_indices = self._start_indices_for_eslice(end)
        if s is not self.UnDef:
            start_indices = start_indices[s]
        return self[start_indices, end]

    def set_eslice(self, key, value):
        """
        Like get_eslice but set value instead of returning values.
        """
        end, s = self._unpack_item(key)
        start_indices = self._start_indices_for_eslice(end)
        if s is not self.UnDef:
            start_indices = start_indices[s]
        self[start_indices, end] = value

    def _get_sblock_index(self, item):
        level, s = self._unpack_item(item)
        if s is self.UnDef:
            s = (slice(None), slice(None))
        start_indices = np.arange(0, self.n - level + 1)
        end_indices = np.concatenate(
            [np.flip(self._end_indices_for_sslice(start)[:level, None], axis=0) for start in start_indices],
            axis=1
        )
        start_indices = start_indices[None, :]
        linear_indices = self.linear_from_start_end(start_indices, end_indices)[s]
        if self.linearise_blocks:
            index = (linear_indices.flatten(),) + tuple([slice(None)] * len(self.value_shape))
            return linear_indices, index
        else:
            return linear_indices, self.UnDef

    def get_sblock(self, item):
        """
        Return a block of sslices down from the specified level.
        """
        linear_indices, index = self._get_sblock_index(item)
        if self.linearise_blocks:
            return self.arr[index].reshape(linear_indices.shape + self.value_shape)
        else:
            return self.arr[linear_indices]

    def set_sblock(self, key, value):
        """
        Like get_sblock but set value.
        """
        linear_indices, index = self._get_sblock_index(key)
        if self.linearise_blocks:
            self.arr[index] = value
        else:
            self.arr[linear_indices] = value

    def _get_eblock_index(self, item):
        level, s = self._unpack_item(item)
        if s is self.UnDef:
            s = (slice(None), slice(None))
        end_indices = np.arange(level, self.n + 1)
        start_indices = np.concatenate(
            [self._start_indices_for_eslice(end)[-level:, None] for end in end_indices],
            axis=1
        )
        end_indices = end_indices[None, :]
        linear_indices = self.linear_from_start_end(start_indices, end_indices)[s]
        if self.linearise_blocks:
            index = (linear_indices.flatten(),) + tuple([slice(None)] * len(self.value_shape))
            return linear_indices, index
        else:
            return linear_indices, self.UnDef

    def get_eblock(self, item):
        """
        Return a block of eslices down from the specified level.
        """
        linear_indices, index = self._get_eblock_index(item)
        if self.linearise_blocks:
            return self.arr[index].reshape(linear_indices.shape + self.value_shape)
        else:
            return self.arr[linear_indices]

    def set_eblock(self, key, value):
        """
        Like get_sblock but set value.
        """
        linear_indices, index = self._get_eblock_index(key)
        if self.linearise_blocks:
            self.arr[index] = value
        else:
            self.arr[linear_indices] = value

    def flatten(self, order="-l+s"):
        """
        Return map in linear order. The different orders correspond to iteration using two nested for loops where the
        first letter indicates the outer dimension and the second the inner: s: start, e: end, l: level. A minus sign
        reverses the order of the respective dimension.

        :param order: string specifying order of linearisation: '+s+e', '+e+s', '+l+s' (+ can be omitted or replaced
         with -)
        :return: 1D array with values in given order
        """
        # get order info
        match = self._flatten_regex.match(order)
        if match is None:
            raise ValueError(f"Invalid order '{order}'")
        outer_dim = match['outer']
        inner_dim = match['inner']
        outer_sign = match['outer_sign']
        inner_sign = match['inner_sign']
        # check
        if (outer_dim, inner_dim) not in [('s', 'e'), ('e', 's'), ('l', 's')]:
            raise ValueError(f"Outer/inner dimension must be s/e, e/s or l/s but are {outer_dim}/{inner_dim}")
        # collect outer slices
        slices = []
        if outer_dim == 's':
            for start in range(self.n):
                slices.append(self.sslice[start])
        elif outer_dim == 'e':
            for end in range(1, self.n + 1):
                slices.append(self.eslice[end])
        else:
            assert outer_dim == 'l', outer_dim
            for level in range(1, self.n + 1):
                slices.append(self.lslice[level])
        # adjust sign for outer dimension
        if outer_sign == '-':
            slices = reversed(slices)
        else:
            assert not outer_sign or outer_sign == '+', outer_sign
        # adjust sign for inner dimension
        if inner_sign == '-':
            if self._is_pytorch():
                slices = [torch.flip(s, dims=(0,)) for s in slices]
            else:
                slices = [np.flip(s, axis=0) for s in slices]
        else:
            assert not inner_sign or inner_sign == '+', inner_sign
        # concatenate and return
        if self._is_pytorch():
            return torch.cat(tuple(slices))
        else:
            return np.concatenate(tuple(slices))

    def __repr__(self):
        return f"TMap(n={self.n}, {self.arr}, linearise_blocks={self.linearise_blocks})"

    def __str__(self):
        """
        Return a string representation of the map, consisting of consecutive dslices.
        """
        s = ""
        for depth in range(self.n):
            if s:
                s += "\n"
            try:
                s += str(self.dslice[depth])
            except TypeError:
                s += "["
                linear_start, linear_end = self.linear_start_end_from_level(self.level(depth))
                s += " ".join([str(self.arr[idx]) for idx in range(linear_start, linear_end + 1)])
                s += "]"
        return s

    def pretty(self, cut=None, str_func=None, detach_pytorch=True, scf=None, pos=None, rnd=None,
               align='r', crosses=False, cross_border=True, fill_char=" ", pad_char=" ", top_char=" ", bottom_char=" ",
               fill_lines=True, haxis=False, daxis=False, laxis=False):
        """
        Pretty-print a triangular map. See the gallery for usage examples.

        :param cut: cut at specified level, printing only the bottom 'cut' levels of the map
        :param str_func: function to convert values to strings (default: str)
        :param detach_pytorch: whether to detach tensors if the underlying array is a pytorch tensor
        :param scf: kwargs to use np.format_float_scientific to format value
        :param pos: kwargs to use np.format_float_positional to format value
        :param rnd: kwargs to use np.around to format value
        :param align: right-align ('r') or left-align ('l') content within cells
        :param crosses: use a different style for plotting the triangular map
        :param cross_border: in 'crosses' style, use crosses also at the border, otherwise use straight lines
        :param fill_char: character used for indenting lines (and filling lines; see ``fill_lines``)
        :param pad_char: character used for padding content (left or right; depending on ``align``)
        :param top_char: character used to fill remaining space at the top within cells
        :param bottom_char: character used to fill remaining space at the bottom within cells
        :param fill_lines: whether to fill lines to same length on the right side
        :param haxis: plot a horizontal axis with ticks and tick labels
        :param daxis: plot a depth axis on the right (not compatible with 'laxis')
        :param laxis: plot a level axis on the right (not compatible with 'daxis')
        :return: pretty-printed string
        """
        if daxis and laxis:
            raise ValueError("Only one of 'daxis' and 'laxis' may be true.")
        # if depth axis is used, lines have to be filled
        if daxis:
            fill_lines = True
        # get function to convert values to strings
        if str_func is None:
            if scf is not None:
                def str_func(val):
                    return np.format_float_scientific(val, **scf)
            elif pos is not None:
                def str_func(val):
                    return np.format_float_positional(val, **pos)
            elif rnd is not None:
                def str_func(val):
                    # when decimals is less or equal to zero (i.e. rounding to whole numbers)
                    # convert to integers for more compact printing
                    if rnd.setdefault("decimals", 0) <= 0:
                        return str(int(np.around(val, **rnd)))
                    else:
                        return str(np.around(val, **rnd))
            else:
                str_func = str
        # get values as strings
        # also get maximum width and number of lines
        str_slices = []
        max_width = 0
        max_lines = 1
        for depth in range(self.n):
            str_slices.append([])
            # level corresponding to depth
            level = self.level(depth)
            # cut at level
            if cut is not None and level > cut:
                continue
            for start in range(depth + 1):
                val = self[start, start + level]
                if self._is_pytorch() and detach_pytorch:
                    val = val.detach().numpy()
                str_lines = str_func(val).split("\n")
                if len(str_lines) > 1:
                    max_width = max(max_width, max(len(s) for s in str_lines))
                    max_lines = max(max_lines, len(str_lines))
                else:
                    max_width = max(max_width, len(str_lines[0]))
                str_slices[-1].append(str_lines)
        # adjust width for different styles
        if crosses:
            # width must be uneven
            max_width = int(2 * np.floor(max_width / 2)) + 1
        else:
            # width must be even
            max_width = int(2 * np.ceil(max_width / 2))
        # adjust width and number of lines of string elements (also align left or right)
        for slice_idx, str_slice in enumerate(str_slices):
            for line_idx, str_lines in enumerate(str_slice):
                while len(str_lines) < max_lines:
                    str_lines.append("")
                for value_idx, str_value in enumerate(str_lines):
                    if align == 'r':
                        str_slices[slice_idx][line_idx][value_idx] = str_value.rjust(max_width, pad_char)
                    elif align == 'l':
                        str_slices[slice_idx][line_idx][value_idx] = str_value.ljust(max_width, pad_char)
                    else:
                        raise ValueError(f"'align' has to be 'l' or 'r' but is '{align}'")
        # adapt max width to account for multi-line values (the actual strings are less wide and will be padded below)
        # max_width += 2 * (max_lines - 1)
        # for x in str_slices:
        #     print(x)
        # generate triangular matrix
        if crosses:
            lines_per_level = (max_width - 1) // 2 + 1
        else:
            lines_per_level = (max_width - 1) // 2 + 2
        s = ""
        # add top for cross-style
        if crosses and cut is None:
            depth_indent = fill_char * lines_per_level * (self.n - 1)
            line_indent = fill_char * (lines_per_level - 1)
            pad = depth_indent + line_indent + fill_char * (self.n * (max_lines - 1) + 1)
            s += pad + "╳"
            if fill_lines:
                s += pad
            # add depth axis label
            if daxis:
                s += " depth"
            if laxis:
                s += " level"
        for depth, str_slice in enumerate(str_slices):
            # level corresponding to depth
            level = self.level(depth)
            # cut at level
            if cut is not None and level > cut:
                continue
            # base indentation of this slice
            depth_indent = fill_char * (lines_per_level + max_lines - 1) * (level - 1)
            # add lines to draw structure
            for line in range(lines_per_level - 1):
                # additional indent for this line of slice
                line_indent = fill_char * (lines_per_level - line - 1 + max_lines - 1)
                # newline if not empty
                if s:
                    s += "\n"
                # spacing within and in between cells
                if crosses:
                    within_cell_spacing = top_char * (2 * line + 1)
                    in_between_cell_spacing = bottom_char * (2 * (lines_per_level + max_lines - line - 2) - 1)
                else:
                    within_cell_spacing = top_char * 2 * line
                    in_between_cell_spacing = bottom_char * 2 * (lines_per_level + max_lines - line - 2)
                # add indentation
                s += depth_indent + line_indent
                s += ("╱" + within_cell_spacing + "╲" + in_between_cell_spacing) * depth + "╱" + within_cell_spacing + "╲"
                if fill_lines:
                    s += line_indent + depth_indent
                # add depth axis label
                if not crosses and depth == 0 and line == 0:
                    if daxis:
                        s += " depth"
                    if laxis:
                        s += " level"
            # add line with actual content
            for sub_idx, sub_slice in enumerate(np.array(str_slice).T):
                sub_line_indent = fill_char * (max_lines - sub_idx - 1)
                padding = fill_char * sub_idx
                if crosses:
                    in_between_fill = fill_char * ((max_lines - sub_idx - 1) * 2 - 1)
                    if sub_idx == max_lines - 1:
                        if cross_border:
                            left_border = "╳"
                            right_border = "╳"
                        else:
                            left_border = "╱"
                            right_border = "╲"
                        s += "\n" + sub_line_indent + depth_indent + left_border + padding + (padding + "╳" + padding).join(sub_slice) + padding + right_border
                    else:
                        s += "\n" + sub_line_indent + depth_indent + "╱" + padding + (padding + "╲" + in_between_fill + "╱" + padding).join(sub_slice) + padding + "╲"
                else:
                    in_between_fill = fill_char * (max_lines - sub_idx - 1) * 2
                    s += "\n" + sub_line_indent + depth_indent + "╱" + padding + (padding + "╲" + in_between_fill + "╱" + padding).join(sub_slice) + padding + "╲"
                # fill lines
                if fill_lines:
                    s += depth_indent
                s += fill_char * (max_lines - sub_idx - 1)
                # add depth axis
                if daxis:
                    s += f" {depth}"
                if laxis:
                    s += f" {level}"
        if haxis:
            if crosses:
                tick_spacing = max_width + 2 * (max_lines - 1)
            else:
                tick_spacing = max_width + 1 + 2 * (max_lines - 1)
            just_width = tick_spacing + 1
            n = self.n
            a = [str(x).ljust(just_width) for x in range(n + 1)]
            s += "\n" + ("│" + " " * tick_spacing) * n + "│"
            s += "\n" + "".join(a)
        return s


def _get_size_and_kwargs(n, **kwargs):
    """
    Get the underlying storage ``size`` from ``n`` (:meth:`~size_from_n`); remove 'linearise_blocks' from ``kwargs``
    to create ``kwargs1``; if 'linearise_blocks' is in ``kwargs``, put it in ``kwargs2`` instead, otherwise make
    ``kwargs2`` an empty dict.

    :return: (``size``, ``kwargs1``, ``kwargs2``)
    """
    kwargs1 = kwargs
    kwargs2 = {}
    if 'linearise_blocks' in kwargs1:
        kwargs2['linearise_blocks'] = kwargs1['linearise_blocks']
        del kwargs1['linearise_blocks']
    return TMap.size_from_n(n), kwargs1, kwargs2


def array_tmap(n, value, *args, **kwargs):
    """
    Create a :class:`~TMap` of size ``n`` using a NumPy array initialised with ``value`` as underlying storage. The
    array is created with ``numpy.full``. The first argument (the shape of the array) is computed via
    :meth:`~size_from_n` (if ``n`` is a tuple of integers this will also be a tuple with only the first element being
    changed; see below); the second argument is ``value``; ``*args`` and ``**kwargs`` are passed on (except for
    ``linearise_blocks``, see below).

    :param n: int or tuple of int; if ``n`` is a single integer it specifies the size (width) of the map; if ``n`` is
     a tuple of integers, the first element specifies the size of the map and the remaining ones specify the shape of
     the values stored in the map (i.e. :attr:`~TMap.value_shape`).
    :param value: value to initialise the underlying array with ``numpy.full``
    :param args: additional arguments passed on to ``numpy.full``
    :param kwargs: additional arguments passed on to ``numpy.full``; if ``linearise_blocks`` is in ``kwargs`` this
     will be removed first and instead directly passed on to initialise the :class:`~TMap`.
    """
    if isinstance(n, tuple):
        n_ = n[0]
        value_shape = n[1:]
    else:
        n_ = n
        value_shape = ()
    size, kw1, kw2 = _get_size_and_kwargs(n_, **kwargs)
    return TMap(arr=np.full((size,) + value_shape, value, *args, **kw1), **kw2)


def tensor_tmap(n, value, *args, **kwargs):
    """
    Same as :meth:`~array_tmap` but use a PyTorch tensor (``torch.full``) instead.
    """
    if isinstance(n, tuple):
        n_ = n[0]
        value_shape = n[1:]
    else:
        n_ = n
        value_shape = ()
    size, kw1, kw2 = _get_size_and_kwargs(n_, **kwargs)
    return TMap(arr=torch.full((size,) + value_shape, value, *args, **kw1), **kw2)


def dict_tmap(n, default_factory, *args, **kwargs):
    """
    Create a :class:`~TMap` of size ``n`` using a :class:`defaultdict` as underlying storage. Elements will only be
    initialised (using the ``default_factory`` function) when accessed. Note that :meth:`~TMap.pretty` printing the
    map will typically access (and thus initialise) all elements.

    :param n: size of the map
    :param default_factory: function to initialise values in the dict/map (passed on to defaultdict as first argument)
    :param args: additional arguments passed on the defaultdict
    :param kwargs: additional arguments passed on the defaultdict
    """
    return TMap(arr=defaultdict(default_factory, *args, **kwargs), _n=n)

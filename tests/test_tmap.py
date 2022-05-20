#  Copyright (c) 2022 Robert Lieck.

from unittest import TestCase
from itertools import chain

import numpy as np
from numpy.testing import assert_array_equal
import torch

from tmap import TMap


class TestTMap(TestCase):

    list_values = [[0],
                   [1, 2],
                   [3, 4, 5],
                   [6, 7, 8, 9],
                   [10, 11, 12, 13, 14],
                   [15, 16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25, 26, 27],
                   [28, 29, 30, 31, 32, 33, 34, 35],
                   [36, 37, 38, 39, 40, 41, 42, 43, 44],
                   [45, 46, 47, 48, 49, 50, 51, 52, 53, 54]]

    str_values = "[0]\n" \
                 "[1 2]\n" \
                 "[3 4 5]\n" \
                 "[6 7 8 9]\n" \
                 "[10 11 12 13 14]\n" \
                 "[15 16 17 18 19 20]\n" \
                 "[21 22 23 24 25 26 27]\n" \
                 "[28 29 30 31 32 33 34 35]\n" \
                 "[36 37 38 39 40 41 42 43 44]\n" \
                 "[45 46 47 48 49 50 51 52 53 54]"

    def get_tmap(self, n, linearise=False, multi_dim=None):
        # construct triangular map with size n
        arr = np.arange(n * (n + 1) / 2).astype(int)
        # add multiple dimensions
        if multi_dim is not None:
            arr = np.concatenate([arr[..., None]] * multi_dim, axis=-1)
        # double-check size
        self.assertEqual(len(arr), TMap.size1d_from_n(n))
        # construct TMap
        tri = TMap(arr, linearise_blocks=linearise)
        # double-check layout for non-multidimensional case
        if multi_dim is None and n == 10:
            self.assertEqual("[0]\n"
                             "[1 2]\n"
                             "[3 4 5]\n"
                             "[6 7 8 9]\n"
                             "[10 11 12 13 14]\n"
                             "[15 16 17 18 19 20]\n"
                             "[21 22 23 24 25 26 27]\n"
                             "[28 29 30 31 32 33 34 35]\n"
                             "[36 37 38 39 40 41 42 43 44]\n"
                             "[45 46 47 48 49 50 51 52 53 54]", str(tri))
        return tri

    def test_size(self):
        for n in np.random.randint(2, 100, 100):
            size = TMap.size1d_from_n(n)
            # make sure inverse operations work
            self.assertEqual(n, TMap.n_from_size1d(size))
            # check bad size raises
            self.assertRaises(ValueError, lambda: TMap.n_from_size1d(size + 1))

    def test_copy(self):
        n = 10
        for type_mapper in [np.array, torch.from_numpy, list, tuple, lambda arr: {v: v for v in arr}]:
            tri = self.get_tmap(n)
            tri = TMap(type_mapper(tri.arr), linearise_blocks=tri.linearise_blocks)
            tri_ = tri.copy()
            # assert objects are not equal
            self.assertIsNot(tri, tri_)
            # assert underlying data are not equal (except for tuples for which copies with primitive types result in
            # identical objects)
            if type_mapper is not tuple:
                # check objects
                self.assertIsNot(tri.arr, tri_.arr)
                # check modification (randomly modify values)
                for start, end, val in np.random.randint(0, n + 1, (100, 3)):
                    if start == end:
                        continue
                    start, end = sorted((start, end))
                    # assert values are still equal
                    self.assertEqual(tri[start, end], tri_[start, end])
                    # modify copy
                    tri_[start, end] = val
                    # assert values are not equal anymore (unless random value happens to be existing value)
                    if val != tri[start, end]:
                        self.assertNotEqual(tri[start, end], tri_[start, end])
                    # modify original
                    tri[start, end] = val
                    # assert values are again equal
                    self.assertEqual(tri[start, end], tri_[start, end])

    def test_top(self):
        n = 10
        for n_dims in [None, 1, 2]:
            tri_n = self.get_tmap(n, multi_dim=n_dims)
            for depth in range(n):
                t = self.get_tmap(depth, multi_dim=n_dims)
                assert_array_equal(t.arr, tri_n.top(depth).arr)

    def test_level(self):
        for n in np.random.randint(3, 20, 100):
            tri = self.get_tmap(n)
            for depth in range(n):
                self.assertEqual(n - depth, tri.level(depth))
                width = n - depth
                for start in range(depth + 1):
                    end = start + width
                    self.assertEqual(n - depth, tri.level(start, end))
            self.assertRaises(TypeError, lambda: tri.level(1, 2, 3))

    def test_get_set_single_values(self):
        n = 10
        for n_dims in [None, 1, 2]:
            tri = self.get_tmap(n, multi_dim=n_dims)
            # test getting/setting single values
            idx = 0
            for depth in range(n):
                width = n - depth
                for start in range(depth + 1):
                    # expected value
                    val = idx
                    if n_dims is not None:
                        val = np.array([idx] * n_dims)
                    # end index
                    end = start + width
                    # get item
                    assert_array_equal(val, tri[start, end])
                    # set item and get new value
                    tri[start, end] = val + 1
                    assert_array_equal(val + 1, tri[start, end])
                    idx += 1
            # double-check index with size
            self.assertEqual(idx, n * (n + 1) / 2)
            # check out of bound
            self.assertRaises(IndexError, lambda: tri[-1, 2])
            self.assertRaises(IndexError, lambda: tri[0, n + 1])
            self.assertRaises(IndexError, lambda: tri[2, 2])

    def test_get_submaps(self):
        n = 10
        for type_mapper in [np.array, torch.from_numpy, list]:
            for n_dims in [None, 1, 2]:
                tri = self.get_tmap(n, multi_dim=n_dims)
                tri = TMap(type_mapper(tri.arr), linearise_blocks=tri.linearise_blocks)
                idx = 0
                for depth in range(n):
                    width = n - depth
                    for start in range(depth + 1):
                        # end index
                        end = start + width
                        # expected value
                        if depth == 0:
                            val = np.array(sum(self.list_values, []))
                        else:
                            if end != n:
                                rev_end = -(depth - start)
                            else:
                                rev_end = n + 1
                            val = np.array(
                                [v for sl in self.list_values[depth:] for v in np.array(sl)[start:rev_end]]
                            )
                        # multidimensional values
                        if n_dims is not None:
                            val = np.concatenate((val[:, None],) * n_dims, axis=1)
                        # get item
                        assert_array_equal(val, tri[start:end].arr)
                        idx += 1
                # double-check index with size
                self.assertEqual(idx, n * (n + 1) / 2)
                # check out of bound
                self.assertRaises(IndexError, lambda: tri[-1, 2])
                self.assertRaises(IndexError, lambda: tri[0, n + 1])
                self.assertRaises(IndexError, lambda: tri[2, 2])

    def test_slices(self):
        n = 10
        for n_dims in [None, 1, 2]:
            tri = self.get_tmap(n, multi_dim=n_dims)
            for level, depth, arr in zip(reversed(range(1, n + 1)), range(n), self.list_values):
                if n_dims is not None:
                    arr = np.concatenate([np.array(arr)[:, None]] * n_dims, axis=-1)
                assert_array_equal(arr, tri.lslice(level))
                assert_array_equal(arr, tri.dslice(depth))

    def test_print(self):
        n = 10
        for type_func in [np.array, lambda arr: {idx: v for idx, v in enumerate(arr)}]:
            tri = self.get_tmap(n)
            # potentially convert to dict (which cannot be sliced)
            tri = TMap(type_func(tri.arr), linearise_blocks=tri.linearise_blocks)
            # normal print
            self.assertEqual(self.str_values, str(tri))
            # pretty print
            pretty_str = "           ╱╲\n" \
                         "          ╱ 0╲\n" \
                         "         ╱╲  ╱╲\n" \
                         "        ╱ 1╲╱ 2╲\n" \
                         "       ╱╲  ╱╲  ╱╲\n" \
                         "      ╱ 3╲╱ 4╲╱ 5╲\n" \
                         "     ╱╲  ╱╲  ╱╲  ╱╲\n" \
                         "    ╱ 6╲╱ 7╲╱ 8╲╱ 9╲\n" \
                         "   ╱╲  ╱╲  ╱╲  ╱╲  ╱╲\n" \
                         "  ╱10╲╱11╲╱12╲╱13╲╱14╲\n" \
                         " ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲\n" \
                         "╱15╲╱16╲╱17╲╱18╲╱19╲╱20╲"
            self.assertEqual(pretty_str, self.get_tmap(6).pretty())

    def test_get_set_ranges_and_se_slice(self):
        n = 10
        # the first column [0, 1, 3, 6, ...]
        first_column = np.array([int(x * (x + 1) / 2) for x in reversed(range(n))])
        # counter to double-check
        counter = -1

        # check for different dimensions
        for n_dims in [None, 1, 2]:
            # slices for fixed start index
            tri = self.get_tmap(n, multi_dim=n_dims)
            for counter, start in enumerate(range(n)):
                # expected value
                if start == 0:
                    exp_val = first_column
                else:
                    exp_val = first_column[:-start] + start
                # add dimensions
                if n_dims is not None:
                    if n_dims == 1:
                        exp_val = exp_val[:, None]
                    elif n_dims > 1:
                        exp_val = np.concatenate([exp_val[:, None]] * n_dims, axis=-1)
                # check value
                assert_array_equal(exp_val, tri.sslice[start])
                # check with additional slice
                assert_array_equal(exp_val[1:-1], tri.sslice[start, 1:-1])
                assert_array_equal(exp_val[1:], tri.sslice[start, 1:])
                assert_array_equal(exp_val[:-1], tri.sslice[start, :-1])
                # set to different value with slicing and check new value (only if slice is at least of size 2)
                if start < n - 1:
                    tri.sslice[start, 1:-1] = exp_val[1:-1] + 1
                    assert_array_equal([exp_val[0]] + list(exp_val[1:-1] + 1) + [exp_val[-1]], tri.sslice[start])
                    tri.sslice[start, 1:] = exp_val[1:] + 2
                    assert_array_equal([exp_val[0]] + list(exp_val[1:-1] + 2) + [exp_val[-1] + 2], tri.sslice[start])
                    tri.sslice[start, :-1] = exp_val[:-1] + 3
                    assert_array_equal([exp_val[0] + 3] + list(exp_val[1:-1] + 3) + [exp_val[-1] + 2], tri.sslice[start])
                tri.sslice[start] = exp_val + 4
                assert_array_equal(exp_val + 4, tri.sslice[start])
            self.assertEqual(n - 1, counter)

            # slices for fixed end index
            tri = self.get_tmap(n, multi_dim=n_dims)
            for counter, end in enumerate(range(1, n + 1)):
                # expected value (flip for end slices)
                exp_val = np.flip(first_column)[n - end:] + np.arange(end)
                # add dimensions
                if n_dims is not None:
                    if n_dims == 1:
                        exp_val = exp_val[:, None]
                    elif n_dims > 1:
                        exp_val = np.concatenate([exp_val[:, None]] * n_dims, axis=-1)
                # check value
                assert_array_equal(exp_val, tri.eslice[end])
                # check with additional slice
                assert_array_equal(exp_val[1:-1], tri.eslice[end, 1:-1])
                assert_array_equal(exp_val[1:], tri.eslice[end, 1:])
                assert_array_equal(exp_val[:-1], tri.eslice[end, :-1])
                # set to different value with slicing and check new value (only if slice is at least of size 2)
                if end > 1:
                    tri.eslice[end, 1:-1] = exp_val[1:-1] + 1
                    assert_array_equal([exp_val[0]] + list(exp_val[1:-1] + 1) + [exp_val[-1]], tri.eslice[end])
                    tri.eslice[end, 1:] = exp_val[1:] + 2
                    assert_array_equal([exp_val[0]] + list(exp_val[1:-1] + 2) + [exp_val[-1] + 2], tri.eslice[end])
                    tri.eslice[end, :-1] = exp_val[:-1] + 3
                    assert_array_equal([exp_val[0] + 3] + list(exp_val[1:-1] + 3) + [exp_val[-1] + 2], tri.eslice[end])
                tri.eslice[end] = exp_val + 4
                assert_array_equal(exp_val + 4, tri.eslice[end])
            self.assertEqual(n - 1, counter)

    def test_se_blocks(self):
        for linearise in [False, True]:
            n = 10
            for n_dims in [None, 1]:
                # flat shape for reshaping (incl. shape of values for multi-dimensional values)
                if n_dims is None:
                    # just flatten
                    flat_shape = (-1,)
                else:
                    # flatten everything except for the value dimensions
                    flat_shape = (-1, n_dims)
                # slices for fixed start index
                tri = self.get_tmap(n, linearise, multi_dim=n_dims)
                for level in range(1, n + 1):
                    start_block = np.array([v[:n - level + 1] for v in self.list_values[n - level:]])
                    end_block = np.array([v[idx:] for idx, v in enumerate(self.list_values[n - level:])])
                    # adapt dimensions
                    if n_dims is not None:
                        if n_dims == 1:
                            start_block = start_block[:, :, None]
                            end_block = end_block[:, :, None]
                        elif n_dims > 1:
                            start_block = np.concatenate([start_block[:, :, None]] * n_dims, axis=-1)
                            end_block = np.concatenate([end_block[:, :, None]] * n_dims, axis=-1)
                    # check default
                    assert_array_equal(start_block, tri.sblock[level])
                    assert_array_equal(end_block, tri.eblock[level])
                    # cut beginning/end
                    # beginning
                    assert_array_equal(start_block[:, 1:], tri.sblock[level, :, 1:])
                    assert_array_equal(end_block[:, 1:], tri.eblock[level, :, 1:])
                    # end
                    assert_array_equal(start_block[:, :-1], tri.sblock[level, :, :-1])
                    assert_array_equal(end_block[:, :-1], tri.eblock[level, :, :-1])
                    # both
                    assert_array_equal(start_block[:, 1:-1], tri.sblock[level, :, 1:-1])
                    assert_array_equal(end_block[:, 1:-1], tri.eblock[level, :, 1:-1])
                    # cut front/back
                    # front
                    assert_array_equal(start_block[1:, :], tri.sblock[level, 1:, :])
                    assert_array_equal(end_block[1:, :], tri.eblock[level, 1:, :])
                    # back
                    assert_array_equal(start_block[:-1, :], tri.sblock[level, :-1, :])
                    assert_array_equal(end_block[:-1, :], tri.eblock[level, :-1, :])
                    # both
                    assert_array_equal(start_block[1:-1, :], tri.sblock[level, 1:-1, :])
                    assert_array_equal(end_block[1:-1, :], tri.eblock[level, 1:-1, :])

                    # set and reset blocks (with and without slicing)
                    if linearise:
                        # start block
                        tri.sblock[level, 1:-1, 1:-1] = start_block[1:-1, 1:-1].reshape(flat_shape) + 1
                        assert_array_equal(start_block[1:-1, 1:-1] + 1, tri.sblock[level, 1:-1, 1:-1])
                        tri.sblock[level] = start_block.reshape(flat_shape) + 2
                        assert_array_equal(start_block + 2, tri.sblock[level])
                        tri.sblock[level] = start_block.reshape(flat_shape)
                        # end block
                        tri.eblock[level, 1:-1, 1:-1] = end_block[1:-1, 1:-1].reshape(flat_shape) + 1
                        assert_array_equal(end_block[1:-1, 1:-1] + 1, tri.eblock[level, 1:-1, 1:-1])
                        tri.eblock[level] = end_block.reshape(flat_shape) + 2
                        assert_array_equal(end_block + 2, tri.eblock[level])
                        tri.eblock[level] = end_block.reshape(flat_shape)
                    else:
                        # start block
                        tri.sblock[level, 1:-1, 1:-1] = start_block[1:-1, 1:-1] + 1
                        assert_array_equal(start_block[1:-1, 1:-1] + 1, tri.sblock[level, 1:-1, 1:-1])
                        tri.sblock[level] = start_block + 2
                        assert_array_equal(start_block + 2, tri.sblock[level])
                        tri.sblock[level] = start_block
                        # end block
                        tri.eblock[level, 1:-1, 1:-1] = end_block[1:-1, 1:-1] + 1
                        assert_array_equal(end_block[1:-1, 1:-1] + 1, tri.eblock[level, 1:-1, 1:-1])
                        tri.eblock[level] = end_block + 2
                        assert_array_equal(end_block + 2, tri.eblock[level])
                        tri.eblock[level] = end_block

    def test_flatten(self):
        n = 10
        for n_dims in [None, 1, 2]:
            tri = self.get_tmap(n, multi_dim=n_dims)
            # check invalid values of order parameter
            self.assertRaises(ValueError, lambda: tri.flatten("xyz"))
            self.assertRaises(ValueError, lambda: tri.flatten("ss"))
            # expected output values (without additional dimensions)
            # '-ls'
            level_slices = [[0],
                            [1, 2],
                            [3, 4, 5],
                            [6, 7, 8, 9],
                            [10, 11, 12, 13, 14],
                            [15, 16, 17, 18, 19, 20],
                            [21, 22, 23, 24, 25, 26, 27],
                            [28, 29, 30, 31, 32, 33, 34, 35],
                            [36, 37, 38, 39, 40, 41, 42, 43, 44],
                            [45, 46, 47, 48, 49, 50, 51, 52, 53, 54]]
            # 's-e'
            start_slices = [[0, 1, 3, 6, 10, 15, 21, 28, 36, 45],
                            [2, 4, 7, 11, 16, 22, 29, 37, 46],
                            [5, 8, 12, 17, 23, 30, 38, 47],
                            [9, 13, 18, 24, 31, 39, 48],
                            [14, 19, 25, 32, 40, 49],
                            [20, 26, 33, 41, 50],
                            [27, 34, 42, 51],
                            [35, 43, 52],
                            [44, 53],
                            [54]]
            # 'es'
            end_slices = [[45],
                          [36, 46],
                          [28, 37, 47],
                          [21, 29, 38, 48],
                          [15, 22, 30, 39, 49],
                          [10, 16, 23, 31, 40, 50],
                          [6, 11, 17, 24, 32, 41, 51],
                          [3, 7, 12, 18, 25, 33, 42, 52],
                          [1, 4, 8, 13, 19, 26, 34, 43, 53],
                          [0, 2, 5, 9, 14, 20, 27, 35, 44, 54]]
            # add additional dimensions
            for exp_val_list in [level_slices, start_slices, end_slices]:
                if n_dims is not None:
                    for outer_idx in range(len(exp_val_list)):
                        for inner_idx in range(len(exp_val_list[outer_idx])):
                            exp_val_list[outer_idx][inner_idx] = [exp_val_list[outer_idx][inner_idx]] * n_dims
            # standard order is '-ls'
            assert_array_equal(list(chain(*level_slices)), tri.flatten())
            # check level-first orders
            assert_array_equal(list(chain(*level_slices)), tri.flatten('-ls'))
            assert_array_equal(list(chain(*level_slices)), tri.flatten('-l+s'))
            assert_array_equal(list(chain(*reversed(level_slices))), tri.flatten('ls'))
            assert_array_equal(list(chain(*reversed(level_slices))), tri.flatten('+l+s'))
            assert_array_equal(list(chain(*[reversed(l) for l in reversed(level_slices)])), tri.flatten('l-s'))
            assert_array_equal(list(chain(*[reversed(l) for l in reversed(level_slices)])), tri.flatten('+l-s'))
            assert_array_equal(list(chain(*[reversed(l) for l in level_slices])), tri.flatten('-l-s'))
            # check start-first orders
            assert_array_equal(list(chain(*start_slices)), tri.flatten('s-e'))
            assert_array_equal(list(chain(*start_slices)), tri.flatten('+s-e'))
            assert_array_equal(list(chain(*reversed(start_slices))), tri.flatten('-s-e'))
            assert_array_equal(list(chain(*[reversed(l) for l in reversed(start_slices)])), tri.flatten('-se'))
            assert_array_equal(list(chain(*[reversed(l) for l in reversed(start_slices)])), tri.flatten('-s+e'))
            assert_array_equal(list(chain(*[reversed(l) for l in start_slices])), tri.flatten('se'))
            assert_array_equal(list(chain(*[reversed(l) for l in start_slices])), tri.flatten('+s+e'))
            # check end-first orders
            assert_array_equal(list(chain(*end_slices)), tri.flatten('es'))
            assert_array_equal(list(chain(*end_slices)), tri.flatten('+e+s'))
            assert_array_equal(list(chain(*reversed(end_slices))), tri.flatten('-es'))
            assert_array_equal(list(chain(*reversed(end_slices))), tri.flatten('-e+s'))
            assert_array_equal(list(chain(*[reversed(l) for l in reversed(end_slices)])), tri.flatten('-e-s'))
            assert_array_equal(list(chain(*[reversed(l) for l in end_slices])), tri.flatten('e-s'))
            assert_array_equal(list(chain(*[reversed(l) for l in end_slices])), tri.flatten('+e-s'))

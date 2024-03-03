#  Copyright (c) 2022 Robert Lieck.

from unittest import TestCase, skip
from itertools import chain
from collections import defaultdict

import numpy as np
from numpy.testing import assert_array_equal
import torch

from triangularmap.tmap import TMap, array_tmap, tensor_tmap, dict_tmap


# @skip("Disabled to see example coverage")
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

    def to_pytorch(self, tri):
        return TMap(torch.from_numpy(tri.arr))

    def get_tmap(self, n, linearise=False, multi_dim=None):
        # construct triangular map with size n
        arr = np.arange(n * (n + 1) / 2).astype(int)
        # add multiple dimensions
        if multi_dim is not None:
            arr = np.concatenate([arr[..., None]] * multi_dim, axis=-1)
        # double-check size
        self.assertEqual(len(arr), TMap.size_from_n(n))
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

    def test_repr(self):
        tmap = self.get_tmap(3)
        self.assertEqual(tmap.__repr__(), "TMap(n=3, [0 1 2 3 4 5], linearise_blocks=False)")

    def test_arr(self):
        tmap = self.get_tmap(3)
        assert_array_equal(tmap.arr, np.array([0, 1, 2, 3, 4, 5]))

        def f():
            tmap.arr = []

        self.assertRaises(AttributeError, f)

    def test_size(self):
        for n in np.random.randint(2, 100, 100):
            size = TMap.size_from_n(n)
            # make sure inverse operations work
            self.assertEqual(n, TMap.n_from_size(size))
            # check bad size raises
            self.assertRaises(ValueError, lambda: TMap.n_from_size(size + 1))

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
            self.assertIs(tri_n.top(), tri_n)
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
                assert_array_equal(arr, tri.lslice[level])
                assert_array_equal(arr, tri.dslice[depth])

    def test_print(self):
        n = 10
        for type_func in [np.array, lambda arr: {idx: v for idx, v in enumerate(arr)}]:
            tri = self.get_tmap(n)
            # potentially convert to dict (which cannot be sliced)
            tri = TMap(type_func(tri.arr), linearise_blocks=tri.linearise_blocks)
            # normal print
            self.assertEqual(self.str_values, str(tri))
            # pretty print
            tmap = self.get_tmap(6)
            self.assertRaises(ValueError, lambda: tmap.pretty(daxis=True, laxis=True))
            pretty_str = "           ╱╲           \n" \
                         "          ╱ 0╲          \n" \
                         "         ╱╲  ╱╲         \n" \
                         "        ╱ 1╲╱ 2╲        \n" \
                         "       ╱╲  ╱╲  ╱╲       \n" \
                         "      ╱ 3╲╱ 4╲╱ 5╲      \n" \
                         "     ╱╲  ╱╲  ╱╲  ╱╲     \n" \
                         "    ╱ 6╲╱ 7╲╱ 8╲╱ 9╲    \n" \
                         "   ╱╲  ╱╲  ╱╲  ╱╲  ╱╲   \n" \
                         "  ╱10╲╱11╲╱12╲╱13╲╱14╲  \n" \
                         " ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲ \n" \
                         "╱15╲╱16╲╱17╲╱18╲╱19╲╱20╲"
            self.assertEqual(pretty_str, tmap.pretty())
            self.assertEqual(pretty_str, self.to_pytorch(tmap).pretty())
            pretty_str = "           ╱╲            depth\n" \
                         "          ╱ 0╲           0\n" \
                         "         ╱╲  ╱╲         \n" \
                         "        ╱ 1╲╱ 2╲         1\n" \
                         "       ╱╲  ╱╲  ╱╲       \n" \
                         "      ╱ 3╲╱ 4╲╱ 5╲       2\n" \
                         "     ╱╲  ╱╲  ╱╲  ╱╲     \n" \
                         "    ╱ 6╲╱ 7╲╱ 8╲╱ 9╲     3\n" \
                         "   ╱╲  ╱╲  ╱╲  ╱╲  ╱╲   \n" \
                         "  ╱10╲╱11╲╱12╲╱13╲╱14╲   4\n" \
                         " ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲ \n" \
                         "╱15╲╱16╲╱17╲╱18╲╱19╲╱20╲ 5"
            self.assertEqual(pretty_str, tmap.pretty(daxis=True))
            pretty_str = "           ╱╲            level\n" \
                         "          ╱ 0╲           6\n" \
                         "         ╱╲  ╱╲         \n" \
                         "        ╱ 1╲╱ 2╲         5\n" \
                         "       ╱╲  ╱╲  ╱╲       \n" \
                         "      ╱ 3╲╱ 4╲╱ 5╲       4\n" \
                         "     ╱╲  ╱╲  ╱╲  ╱╲     \n" \
                         "    ╱ 6╲╱ 7╲╱ 8╲╱ 9╲     3\n" \
                         "   ╱╲  ╱╲  ╱╲  ╱╲  ╱╲   \n" \
                         "  ╱10╲╱11╲╱12╲╱13╲╱14╲   2\n" \
                         " ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲ \n" \
                         "╱15╲╱16╲╱17╲╱18╲╱19╲╱20╲ 1"
            self.assertEqual(pretty_str, tmap.pretty(laxis=True))
            pretty_str = "         ╱╲  ╱╲\n" \
                         "        ╱1 ╲╱2 ╲\n" \
                         "       ╱╲  ╱╲  ╱╲\n" \
                         "      ╱3 ╲╱4 ╲╱5 ╲\n" \
                         "     ╱╲  ╱╲  ╱╲  ╱╲\n" \
                         "    ╱6 ╲╱7 ╲╱8 ╲╱9 ╲\n" \
                         "   ╱╲  ╱╲  ╱╲  ╱╲  ╱╲\n" \
                         "  ╱10╲╱11╲╱12╲╱13╲╱14╲\n" \
                         " ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲\n" \
                         "╱15╲╱16╲╱17╲╱18╲╱19╲╱20╲\n" \
                         "│   │   │   │   │   │   │\n" \
                         "0   1   2   3   4   5   6   "
            self.assertEqual(pretty_str, tmap.pretty(align='l', fill_lines=False, haxis=True, cut=5))
            self.assertRaises(ValueError, lambda: tmap.pretty(align='c'))
            pretty_str = "––––––––––––╳–––––––––––– depth\n" \
                         "–––––––––––╱.╲–––––––––––\n" \
                         "––––––––––╳..0╳–––––––––– 0\n" \
                         "–––––––––╱.╲.╱.╲–––––––––\n" \
                         "––––––––╳..1╳..2╳–––––––– 1\n" \
                         "–––––––╱.╲.╱.╲.╱.╲–––––––\n" \
                         "––––––╳..3╳..4╳..5╳–––––– 2\n" \
                         "–––––╱.╲.╱.╲.╱.╲.╱.╲–––––\n" \
                         "––––╳..6╳..7╳..8╳..9╳–––– 3\n" \
                         "–––╱.╲.╱.╲.╱.╲.╱.╲.╱.╲–––\n" \
                         "––╳.10╳.11╳.12╳.13╳.14╳–– 4\n" \
                         "–╱.╲.╱.╲.╱.╲.╱.╲.╱.╲.╱.╲–\n" \
                         "╳.15╳.16╳.17╳.18╳.19╳.20╳ 5\n" \
                         "│   │   │   │   │   │   │\n" \
                         "0   1   2   3   4   5   6   "
            self.assertEqual(pretty_str, tmap.pretty(crosses=True,
                                                     haxis=True, daxis=True,
                                                     fill_char="–",
                                                     pad_char='.', top_char='.', bottom_char='.'))
            self.assertEqual(tmap.pretty(crosses=True, cross_border=False, fill_char='-'),
                             """------------╳------------\n"""
                             """-----------╱ ╲-----------\n"""
                             """----------╱  0╲----------\n"""
                             """---------╱ ╲ ╱ ╲---------\n"""
                             """--------╱  1╳  2╲--------\n"""
                             """-------╱ ╲ ╱ ╲ ╱ ╲-------\n"""
                             """------╱  3╳  4╳  5╲------\n"""
                             """-----╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲-----\n"""
                             """----╱  6╳  7╳  8╳  9╲----\n"""
                             """---╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲---\n"""
                             """--╱ 10╳ 11╳ 12╳ 13╳ 14╲--\n"""
                             """-╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲-\n"""
                             """╱ 15╳ 16╳ 17╳ 18╳ 19╳ 20╲""")
            pretty_str = "            ╳             level\n" \
                         "           ╱ ╲           \n" \
                         "          ╳  0╳           6\n" \
                         "         ╱ ╲ ╱ ╲         \n" \
                         "        ╳  1╳  2╳         5\n" \
                         "       ╱ ╲ ╱ ╲ ╱ ╲       \n" \
                         "      ╳  3╳  4╳  5╳       4\n" \
                         "     ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲     \n" \
                         "    ╳  6╳  7╳  8╳  9╳     3\n" \
                         "   ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲   \n" \
                         "  ╳ 10╳ 11╳ 12╳ 13╳ 14╳   2\n" \
                         " ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ \n" \
                         "╳ 15╳ 16╳ 17╳ 18╳ 19╳ 20╳ 1"
            self.assertEqual(pretty_str, tmap.pretty(crosses=True, laxis=True))
            pretty_str = "–––––––––––––––––––––––––––––––––––––––––╱╲–––––––––––––––––––––––––––––––––––––––––\n" \
                         "––––––––––––––––––––––––––––––––––––––––╱..╲––––––––––––––––––––––––––––––––––––––––\n" \
                         "–––––––––––––––––––––––––––––––––––––––╱0000╲–––––––––––––––––––––––––––––––––––––––\n" \
                         "––––––––––––––––––––––––––––––––––––––╱╲||||╱╲––––––––––––––––––––––––––––––––––––––\n" \
                         "–––––––––––––––––––––––––––––––––––––╱..╲||╱..╲–––––––––––––––––––––––––––––––––––––\n" \
                         "––––––––––––––––––––––––––––––––––––╱0001╲╱0002╲––––––––––––––––––––––––––––––––––––\n" \
                         "–––––––––––––––––––––––––––––––––––╱╲||||╱╲||||╱╲–––––––––––––––––––––––––––––––––––\n" \
                         "––––––––––––––––––––––––––––––––––╱..╲||╱..╲||╱..╲––––––––––––––––––––––––––––––––––\n" \
                         "–––––––––––––––––––––––––––––––––╱0003╲╱0004╲╱0005╲–––––––––––––––––––––––––––––––––\n" \
                         "––––––––––––––––––––––––––––––––╱╲||||╱╲||||╱╲||||╱╲––––––––––––––––––––––––––––––––\n" \
                         "–––––––––––––––––––––––––––––––╱..╲||╱..╲||╱..╲||╱..╲–––––––––––––––––––––––––––––––\n" \
                         "––––––––––––––––––––––––––––––╱0006╲╱0007╲╱0008╲╱0009╲––––––––––––––––––––––––––––––\n" \
                         "–––––––––––––––––––––––––––––╱╲||||╱╲||||╱╲||||╱╲||||╱╲–––––––––––––––––––––––––––––\n" \
                         "––––––––––––––––––––––––––––╱..╲||╱..╲||╱..╲||╱..╲||╱..╲––––––––––––––––––––––––––––\n" \
                         "–––––––––––––––––––––––––––╱0010╲╱0011╲╱0012╲╱0013╲╱0014╲–––––––––––––––––––––––––––\n" \
                         "––––––––––––––––––––––––––╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲––––––––––––––––––––––––––\n" \
                         "–––––––––––––––––––––––––╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲–––––––––––––––––––––––––\n" \
                         "––––––––––––––––––––––––╱0015╲╱0016╲╱0017╲╱0018╲╱0019╲╱0020╲––––––––––––––––––––––––\n" \
                         "–––––––––––––––––––––––╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲–––––––––––––––––––––––\n" \
                         "––––––––––––––––––––––╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲––––––––––––––––––––––\n" \
                         "–––––––––––––––––––––╱0021╲╱0022╲╱0023╲╱0024╲╱0025╲╱0026╲╱0027╲–––––––––––––––––––––\n" \
                         "––––––––––––––––––––╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲––––––––––––––––––––\n" \
                         "–––––––––––––––––––╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲–––––––––––––––––––\n" \
                         "––––––––––––––––––╱0028╲╱0029╲╱0030╲╱0031╲╱0032╲╱0033╲╱0034╲╱0035╲––––––––––––––––––\n" \
                         "–––––––––––––––––╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲–––––––––––––––––\n" \
                         "––––––––––––––––╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲––––––––––––––––\n" \
                         "–––––––––––––––╱0036╲╱0037╲╱0038╲╱0039╲╱0040╲╱0041╲╱0042╲╱0043╲╱0044╲–––––––––––––––\n" \
                         "––––––––––––––╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲––––––––––––––\n" \
                         "–––––––––––––╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲–––––––––––––\n" \
                         "––––––––––––╱0045╲╱0046╲╱0047╲╱0048╲╱0049╲╱0050╲╱0051╲╱0052╲╱0053╲╱0054╲––––––––––––\n" \
                         "–––––––––––╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲–––––––––––\n" \
                         "––––––––––╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲––––––––––\n" \
                         "–––––––––╱0055╲╱0056╲╱0057╲╱0058╲╱0059╲╱0060╲╱0061╲╱0062╲╱0063╲╱0064╲╱0065╲–––––––––\n" \
                         "––––––––╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲––––––––\n" \
                         "–––––––╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲–––––––\n" \
                         "––––––╱0066╲╱0067╲╱0068╲╱0069╲╱0070╲╱0071╲╱0072╲╱0073╲╱0074╲╱0075╲╱0076╲╱0077╲––––––\n" \
                         "–––––╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲–––––\n" \
                         "––––╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲––––\n" \
                         "–––╱0078╲╱0079╲╱0080╲╱0081╲╱0082╲╱0083╲╱0084╲╱0085╲╱0086╲╱0087╲╱0088╲╱0089╲╱0090╲–––\n" \
                         "––╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲||||╱╲––\n" \
                         "–╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲||╱..╲–\n" \
                         "╱0091╲╱0092╲╱0093╲╱0094╲╱0095╲╱0096╲╱0097╲╱0098╲╱0099╲╱0100╲╱0101╲╱0102╲╱0103╲╱0104╲\n" \
                         "│     │     │     │     │     │     │     │     │     │     │     │     │     │     │\n" \
                         "0     1     2     3     4     5     6     7     8     9     10    11    12    13    14    "
            self.assertEqual(pretty_str, self.get_tmap(14).pretty(fill_char="–",
                                                                  pad_char='0',
                                                                  top_char='.',
                                                                  bottom_char='|',
                                                                  haxis=True))
            tmap = self.get_tmap(3)
            pretty_str = '           ╱╲           \n' \
                         '          ╱  ╲          \n' \
                         '         ╱    ╲         \n' \
                         '        ╱0.e+00╲        \n' \
                         '       ╱╲      ╱╲       \n' \
                         '      ╱  ╲    ╱  ╲      \n' \
                         '     ╱    ╲  ╱    ╲     \n' \
                         '    ╱1.e+00╲╱2.e+00╲    \n' \
                         '   ╱╲      ╱╲      ╱╲   \n' \
                         '  ╱  ╲    ╱  ╲    ╱  ╲  \n' \
                         ' ╱    ╲  ╱    ╲  ╱    ╲ \n' \
                         '╱3.e+00╲╱4.e+00╲╱5.e+00╲'
            self.assertEqual(pretty_str, tmap.pretty(scf=dict()))
            pretty_str = '     ╱╲     \n' \
                         '    ╱0.╲    \n' \
                         '   ╱╲  ╱╲   \n' \
                         '  ╱1.╲╱2.╲  \n' \
                         ' ╱╲  ╱╲  ╱╲ \n' \
                         '╱3.╲╱4.╲╱5.╲'
            self.assertEqual(pretty_str, tmap.pretty(pos=dict()))
            pretty_str = '     ╱╲     \n' \
                         '    ╱ 0╲    \n' \
                         '   ╱╲  ╱╲   \n' \
                         '  ╱ 1╲╱ 2╲  \n' \
                         ' ╱╲  ╱╲  ╱╲ \n' \
                         '╱ 3╲╱ 4╲╱ 5╲'
            tmap._arr = tmap.arr + 0.1
            self.assertNotEqual(pretty_str, tmap.pretty())
            self.assertEqual(pretty_str, tmap.pretty(rnd=dict(decimals=0)))
            pretty_str = '        ╱╲        \n' \
                         '       ╱  ╲       \n' \
                         '      ╱ 0.1╲      \n' \
                         '     ╱╲    ╱╲     \n' \
                         '    ╱  ╲  ╱  ╲    \n' \
                         '   ╱ 1.1╲╱ 2.1╲   \n' \
                         '  ╱╲    ╱╲    ╱╲  \n' \
                         ' ╱  ╲  ╱  ╲  ╱  ╲ \n' \
                         '╱ 3.1╲╱ 4.1╲╱ 5.1╲'
            self.assertEqual(pretty_str, tmap.pretty(rnd=dict(decimals=1)))

    def test_multiline_print(self):
        # multiline values
        tmap = TMap(np.zeros((3, 3, 3)))
        self.assertEqual(tmap.pretty(align='l', fill_char='-'),
                         """-----------------╱╲-----------------\n"""
                         """----------------╱  ╲----------------\n"""
                         """---------------╱    ╲---------------\n"""
                         """--------------╱      ╲--------------\n"""
                         """-------------╱        ╲-------------\n"""
                         """------------╱          ╲------------\n"""
                         """-----------╱[[0. 0. 0.] ╲-----------\n"""
                         """----------╱- [0. 0. 0.] -╲----------\n"""
                         """---------╱-- [0. 0. 0.]]--╲---------\n"""
                         """--------╱╲                ╱╲--------\n"""
                         """-------╱  ╲              ╱  ╲-------\n"""
                         """------╱    ╲            ╱    ╲------\n"""
                         """-----╱      ╲          ╱      ╲-----\n"""
                         """----╱        ╲        ╱        ╲----\n"""
                         """---╱          ╲      ╱          ╲---\n"""
                         """--╱[[0. 0. 0.] ╲----╱[[0. 0. 0.] ╲--\n"""
                         """-╱- [0. 0. 0.] -╲--╱- [0. 0. 0.] -╲-\n"""
                         """╱-- [0. 0. 0.]]--╲╱-- [0. 0. 0.]]--╲""")
        self.assertEqual(tmap.pretty(align='l', fill_char='-', crosses=True),
                         """------------------╳------------------\n"""
                         """-----------------╱ ╲-----------------\n"""
                         """----------------╱   ╲----------------\n"""
                         """---------------╱     ╲---------------\n"""
                         """--------------╱       ╲--------------\n"""
                         """-------------╱         ╲-------------\n"""
                         """------------╱           ╲------------\n"""
                         """-----------╱[[0. 0. 0.]  ╲-----------\n"""
                         """----------╱- [0. 0. 0.]  -╲----------\n"""
                         """---------╳-- [0. 0. 0.]] --╳---------\n"""
                         """--------╱ ╲               ╱ ╲--------\n"""
                         """-------╱   ╲             ╱   ╲-------\n"""
                         """------╱     ╲           ╱     ╲------\n"""
                         """-----╱       ╲         ╱       ╲-----\n"""
                         """----╱         ╲       ╱         ╲----\n"""
                         """---╱           ╲     ╱           ╲---\n"""
                         """--╱[[0. 0. 0.]  ╲---╱[[0. 0. 0.]  ╲--\n"""
                         """-╱- [0. 0. 0.]  -╲-╱- [0. 0. 0.]  -╲-\n"""
                         """╳-- [0. 0. 0.]] --╳-- [0. 0. 0.]] --╳""")
        tmap = TMap([" ONE ", " TWO \n LINES ", " THESE \n ARE \n FOUR \n LINES "])
        self.assertEqual(tmap.pretty(fill_char='1', pad_char='2', top_char='3', bottom_char='4', align='l'),
                         """111111111111111╱╲111111111111111\n"""
                         """11111111111111╱33╲11111111111111\n"""
                         """1111111111111╱3333╲1111111111111\n"""
                         """111111111111╱333333╲111111111111\n"""
                         """11111111111╱ ONE 222╲11111111111\n"""
                         """1111111111╱1222222221╲1111111111\n"""
                         """111111111╱112222222211╲111111111\n"""
                         """11111111╱11122222222111╲11111111\n"""
                         """1111111╱╲44444444444444╱╲1111111\n"""
                         """111111╱33╲444444444444╱33╲111111\n"""
                         """11111╱3333╲4444444444╱3333╲11111\n"""
                         """1111╱333333╲44444444╱333333╲1111\n"""
                         """111╱ TWO 222╲111111╱ THESE 2╲111\n"""
                         """11╱1 LINES 21╲1111╱1 ARE 2221╲11\n"""
                         """1╱112222222211╲11╱11 FOUR 2211╲1\n"""
                         """╱11122222222111╲╱111 LINES 2111╲""")
        self.assertEqual(tmap.pretty(fill_char='1', pad_char='2', top_char='3', bottom_char='4', align='l', crosses=True),
                         """11111111111111╳11111111111111\n"""
                         """1111111111111╱3╲1111111111111\n"""
                         """111111111111╱333╲111111111111\n"""
                         """11111111111╱33333╲11111111111\n"""
                         """1111111111╱ ONE 22╲1111111111\n"""
                         """111111111╱122222221╲111111111\n"""
                         """11111111╱11222222211╲11111111\n"""
                         """1111111╳1112222222111╳1111111\n"""
                         """111111╱3╲44444444444╱3╲111111\n"""
                         """11111╱333╲444444444╱333╲11111\n"""
                         """1111╱33333╲4444444╱33333╲1111\n"""
                         """111╱ TWO 22╲11111╱ THESE ╲111\n"""
                         """11╱1 LINES 1╲111╱1 ARE 221╲11\n"""
                         """1╱11222222211╲1╱11 FOUR 211╲1\n"""
                         """╳1112222222111╳111 LINES 111╳""")
        tmap = TMap(["1", "12\n34", "123\n456\n789"])
        self.assertEqual(tmap.pretty(fill_char='a', pad_char='b', top_char='c', bottom_char='d', align='l'),
                         """aaaaaaaaa╱╲aaaaaaaaa\n"""
                         """aaaaaaaa╱cc╲aaaaaaaa\n"""
                         """aaaaaaa╱1bbb╲aaaaaaa\n"""
                         """aaaaaa╱abbbba╲aaaaaa\n"""
                         """aaaaa╱aabbbbaa╲aaaaa\n"""
                         """aaaa╱╲dddddddd╱╲aaaa\n"""
                         """aaa╱cc╲dddddd╱cc╲aaa\n"""
                         """aa╱12bb╲aaaa╱123b╲aa\n"""
                         """a╱a34bba╲aa╱a456ba╲a\n"""
                         """╱aabbbbaa╲╱aa789baa╲""")
        self.assertEqual(tmap.pretty(fill_char='a', pad_char='b', top_char='c', bottom_char='d', align='l', crosses=True),
                         """aaaaaaaa╳aaaaaaaa\n"""
                         """aaaaaaa╱c╲aaaaaaa\n"""
                         """aaaaaa╱1bb╲aaaaaa\n"""
                         """aaaaa╱abbba╲aaaaa\n"""
                         """aaaa╳aabbbaa╳aaaa\n"""
                         """aaa╱c╲ddddd╱c╲aaa\n"""
                         """aa╱12b╲aaa╱123╲aa\n"""
                         """a╱a34ba╲a╱a456a╲a\n"""
                         """╳aabbbaa╳aa789aa╳""")
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
                    assert_array_equal([exp_val[0] + 3] + list(exp_val[1:-1] + 3) + [exp_val[-1] + 2],
                                       tri.sslice[start])
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
            tri_np = self.get_tmap(n, multi_dim=n_dims)
            tri_pt = self.to_pytorch(tri_np)
            # check invalid values of order parameter
            self.assertRaises(ValueError, lambda: tri_np.flatten("xyz"))
            self.assertRaises(ValueError, lambda: tri_np.flatten("ss"))
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
            # test function
            def test(slices, *args):
                l = list(chain(*slices))
                assert_array_equal(l, tri_np.flatten(*args))
                assert_array_equal(l, tri_pt.flatten(*args))
            # standard order is '-ls'
            test(level_slices)
            # check level-first orders
            test(level_slices, '-ls')
            test(level_slices, '-l+s')
            test(reversed(level_slices), 'ls')
            test(reversed(level_slices), '+l+s')
            test([reversed(l) for l in reversed(level_slices)], 'l-s')
            test([reversed(l) for l in reversed(level_slices)], '+l-s')
            test([reversed(l) for l in level_slices], '-l-s')
            # check start-first orders
            test(start_slices, 's-e')
            test(start_slices, '+s-e')
            test(reversed(start_slices), '-s-e')
            test([reversed(l) for l in reversed(start_slices)], '-se')
            test([reversed(l) for l in reversed(start_slices)], '-s+e')
            test([reversed(l) for l in start_slices], 'se')
            test([reversed(l) for l in start_slices], '+s+e')
            # check end-first orders
            test(end_slices, 'es')
            test(end_slices, '+e+s')
            test(reversed(end_slices), '-es')
            test(reversed(end_slices), '-e+s')
            test([reversed(l) for l in reversed(end_slices)], '-e-s')
            test([reversed(l) for l in end_slices], 'e-s')
            test([reversed(l) for l in end_slices], '+e-s')

    def test_reindex_top_down_start_end(self):
        seq = np.arange(10)
        # check they are inverse from each other
        assert_array_equal(TMap.reindex_from_start_end_to_top_down(TMap.reindex_from_top_down_to_start_end(seq)), seq)
        assert_array_equal(TMap.reindex_from_top_down_to_start_end(TMap.reindex_from_start_end_to_top_down(seq)), seq)
        # Top-down indexing is like this:
        #        ╱╲
        #       ╱ 0╲
        #      ╱╲  ╱╲
        #     ╱ 1╲╱ 2╲
        #    ╱╲  ╱╲  ╱╲
        #   ╱ 3╲╱ 4╲╱ 5╲
        #  ╱╲  ╱╲  ╱╲  ╱╲
        # ╱ 6╲╱ 7╲╱ 8╲╱ 9╲
        #
        # Start-end indexing is like this:
        #        ╱╲
        #       ╱ 3╲
        #      ╱╲  ╱╲
        #     ╱ 2╲╱ 6╲
        #    ╱╲  ╱╲  ╱╲
        #   ╱ 1╲╱ 5╲╱ 8╲
        #  ╱╲  ╱╲  ╱╲  ╱╲
        # ╱ 0╲╱ 4╲╱ 7╲╱ 9╲
        #
        # check they remapped correctly:
        assert_array_equal(TMap.reindex_from_top_down_to_start_end(seq), [6, 3, 1, 0, 7, 4, 2, 8, 5, 9])
        assert_array_equal(TMap.reindex_from_start_end_to_top_down(seq), [3, 2, 6, 1, 5, 8, 0, 4, 7, 9])

    def test_array_and_tensor_tmap(self):
        for tmap_func, arr_func in [(array_tmap, np.array),
                                    (tensor_tmap, torch.tensor)]:
            for linearise_blocks in [False, True]:
                for shape, init_value, mod_value in [
                    (4, 0, 1),  # simple map
                    ((3,), 1., 2.2),  # simple map but tuple for size
                    ((3, 2), 0, [1, 2]),  # map with 1D value shape
                    ((3, 2, 4), 0, [[1., 2., 3., 4.], [5., 6., 7., 8.]]),  # map with 2D value shape
                ]:
                    tmap = tmap_func(shape, init_value, linearise_blocks=linearise_blocks)
                    if isinstance(shape, tuple):
                        arr = init_value
                        for i, d in reversed(list(enumerate(shape))):
                            if i == 0:
                                d = TMap.size_from_n(d)
                            arr = [arr] * d
                    else:
                        arr = [init_value] * TMap.size_from_n(shape)
                    arr = arr_func(arr)
                    assert_array_equal(tmap.arr, arr)
                    mod_value = arr_func(mod_value)
                    tmap[2, 3] = mod_value
                    arr[tmap.linear_from_start_end(2, 3)] = mod_value
                    assert_array_equal(tmap.arr, arr)

    def test_dict_tmap(self):
        n = 4
        size = TMap.size_from_n(n)
        tmap = dict_tmap(n, list)
        self.assertEqual(tmap.arr, defaultdict(list))
        tmap[0, 1] = "X"
        tmap[0, 2] = 5
        self.assertEqual(set(tmap.arr.values()), {"X", 5})
        self.assertEqual(len(tmap.arr), 2)
        self.assertEqual(tmap.pretty(),
                         "       ╱╲       \n"
                         "      ╱[]╲      \n"
                         "     ╱╲  ╱╲     \n"
                         "    ╱[]╲╱[]╲    \n"
                         "   ╱╲  ╱╲  ╱╲   \n"
                         "  ╱ 5╲╱[]╲╱[]╲  \n"
                         " ╱╲  ╱╲  ╱╲  ╱╲ \n"
                         "╱ X╲╱[]╲╱[]╲╱[]╲")
        self.assertEqual(len(tmap.arr), size)

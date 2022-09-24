#  Copyright (c) 2022 Robert Lieck.

"""
Pretty Print Example
===========================

This example shows typical usages of the :func:`~tmap.TMap.pretty` function.
"""

# %%
# Typical Use Cases
# -----------------
# We create a TMap of width 6

from triangularmap import TMap

n = 6
arr = list(range(TMap.size1d_from_n(n)))
tmap = TMap(arr)

# %%
# and print it:

print(tmap.pretty())

# %%
# We can add a horizontal axis and a depth axis with labels:

print(tmap.pretty(haxis=True, daxis=True))

# %%
# or a level axis instead of the depth axis:

print(tmap.pretty(haxis=True, laxis=True))

# %%
# We can cut of the top and specify custom characters for filling the whitespace:

print(tmap.pretty(crosses=True, cut=4, fill_char="-", top_char='^', bottom_char='v', pad_char='0'))

# %%
# Two Different Styles
# --------------------
# The default style uses '╱' and '╲' characters to draw the map. We can also use a style that additionally uses '╳':

print(tmap.pretty(crosses=True, haxis=True, daxis=True))

# %%
# The 'crosses' style tends to be more compact as only one '╳' is used at the crossings (instead of '╲╱' or '╱╲' as in
# the default style). Moreover, content is padded to fill an uneven number of characters (while in the default style it
# is padded to an even number). The crosses style is therefore particularly compact if the content itself is
# of uneven width (e.g. just a single character wide):

print(TMap('abcdefghij').pretty(crosses=True))

# %%
# versus the default style:

print(TMap('abcdefghij').pretty())

# %%
# On the down-side it has a slightly more "messy" appearance (and therefore is not the default style), because there are
# no unicode character for the boundary, i.e., a '╳' with one of the upper arms missing.

# %%
# Known Issues
# ------------
# Plotting of the triangular map itself is not adapted to accommodate the axes, which may result in ill-formatted axes
# for some edge cases. Below, there are no spaces between two-digit tick labels on the horizontal axis, because the
# content is only one character wide (in more extreme cases, they may even become misaligned if there is not enough
# space):

print(TMap('.' * TMap.size1d_from_n(15)).pretty(crosses=True, daxis=True, haxis=True))
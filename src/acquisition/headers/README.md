# Notes on Analysis of .dat Files

Note we actually have to do a lot of matching here between files. Here for example is subject
p000625 layout.hea and matched .hea files:

```
# 3704341_layout.hea                     |  # p000625-2178-04-27-00-49.hea
                                         |
                                         |  p000625-2178-04-27-00-49/9 8 125 3900000 00:49:05.758 27/04/2178
3704341_layout 8 125 0    49:05.758      |  3704341_layout 0
~ 0 125/mV 8 0 -16384 0 0 II             |  ~ 14877
~ 0 125/mV 8 0 -128 0 0 III              |  3704341_0001 75000
~ 0 125/mV 15 0 -16384 0 0 AVL           |  3704341_0002 7500
~ 0 125/mV 15 0 -16384 0 0 AVF           |  3704341_0003 1792500
~ 0 125/mV 15 0 -16384 0 0 V             |  3704341_0004 1987500
~ 0 125/mV 8 0 -256 0 0 MCL1             |  3704341_0005 7500
~ 0 1.25(-100)/mmHg 8 0 -256 0 0 ABP     |  3704341_0006 15000
~ 0 5(-100)/mmHg 8 0 -1024 0 0 PAP       |  ~ 123
```

From the documentation (https://physionet.org/physiotools/wag/header-5.htm):

> For each database record, a header file specifies the names of the associated signal files and
> their attributes. [...] Header files contain at a minimum a record line, which specifies the
> record name, the number of segments, and **the number of signals**. Header files for ordinary
> records (those that contain one segment) also contain a signal specification line for each signal.
> Header files for multi-segment records (supported by WFDB library version 9.1 and later versions)
> contain a segment specification line for each segment.

We can see the number of signals in the header (not layout) file above. E.g. the
line '3704341_0004 1987500' suggests the `3704341_0004.dat` file should be
very large and have almost 2million samples. Loading the file with

```python
>>> dat = np.fromfile("3704341_0004.dat", dtype="byte")
>>> dat.dtype
int8
>>> dat.shape
(7950000,)
```

From `3704341_layout.hea` we know there are 8 signals here. So

```python
>>> dat = dat.reshape(-1, 8)
```
The .dat files are MIT Signal files. These files are **encoded** waveforms. E.g. they are just a big
array of ints, and have no time information and the the scales are not there (e.g. ABP might be in
-128 to 127  range, and needs an offset to convert to real range values.)

## Decoding the .dat files

The header files specify an sampling frequency and start time which in theory allows determining the
timepoints. Supposedly, these header files also specify the offset, if there is one. If we look at
the line for ABP , we have:

```
~ 0 1.25(-100)/mmHg 8 0 -256 0 0 ABP
```

My guess is that the `-100` is the offset, but WE CAN'T BE SURE. In general, trying to decode these
files ourself is a risk. BUT, one thing to keep in mind is that even if there is an offset, we are
going to subtract it in machine learning anyway (in normalization). So *really* we only care about
the time durations, i.e. the sampling frequency.  BUT if the offsets are byte offsets, it will cause
wrap-around, which is a problem. This appears to be the case:

```python
a = np.fromfile("p000625/3704341_0004.dat", dtype="byte")  # dtype=np.int8 also fine
# this file has 8 signals according to header
a = a.reshape(-1, 8)
# ABP waves are in index 6
abp = a[:, 6]
# if we plot below, we see a signal that has some strange discontinuities
plt.plot(abp[:1000]); plt.show()
# but if we use the "offset" in the brackets
plt.plot(abp[:1000] + 100); plt.show()
# this looks a LOT like an ABP wave (but flipped), e.g. dicrotic notch seems to
# PRECEDE a peak, but it should FOLLOW that peak
# NOTE: also a negative offset doesn't look much different:
plt.plot(abp[:1000] - 100); plt.show()
# also
>>> np.array_equiv(abp + 100, -(abp - 100))
False
>>> np.array_equiv(abp + 100, abp - 100)
False
# Also flipping (reversing) the signal yields a more typical form
plt.plot(np.flip(abp[:1000] - 100)); plt.show()
```

So we need to use the `wfdb` library `rdrecord` function to see what the heck is going on here.

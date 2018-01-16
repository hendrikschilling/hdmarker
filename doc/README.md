HDmarker
================

[TOC]

This library is considering all about the calibration target. It contains the hdmarker generation executable which automatically generates the needed target pattern. Besides the libray also detects the pattern in several recursion steps. In this documentation we detail all necessary functionality, how to use it in C++ and provide an example code \ref demo_usage. Furthermore we detail how to generate a matching calibration target.
<br>
@image html target.png
<br>
<br>
A ready to use target can be found here: <br>
&nbsp; <STRONG>data/target.pdf</STRONG><br>
or<br>
&nbsp; <STRONG>data/target.pdf</STRONG>
<br>
<br>

## How to cut the target to the desired shape
## How to cut the target to the desired shape {#cut_target}

When cutting the target to the desired shape you can use the png version of the target. Afterwards, please vectorize the image before printing the target, i.e., convert the image to pdf. Otherwise the resulting image is blurred. Thus please use inkscape, import the png image and goto: Path->Trace Bitmap. Disable all optimizations like edge smoothing. Then press OK. Finally delete the imported image and save the traced result as pdf.

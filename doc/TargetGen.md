Target generation
================

The HDMarker calibration pattern consists of two (relatively) independent parts. The first relates to the robust, self identifiying calibration pattern which is given by the large coded pattern. The second relates to the dense fractal calibration points which can repeat several times, dependent on the used recursion level. To generate the calibration target please use the following commands:

\note
**usage generate target for specific page size:** <br>
&nbsp; marker_gen <VAR> [page_ID] [recursion_level] [save_name] </VAR>
<br>
**usage generate target to explicite size:** <br> 
&nbsp; marker_gen <VAR> [page_ID] [recursion_level] [target_width] [target_height] [save_name] </VAR>
<br>
<br>
<DIV style="font-size:12px;"><VAR>
[page_ID]         defines page number. please always use 0 <br>
[recursion_level] how many fractal calibration points are generated. <br>
[target_width]    width of the generated target. <br>
[target_height]   height of the generated target. <br>
[save_name]       where to save the target <br>
</VAR>
</DIV>

Example: <VAR> marker_gen 0 2 demoTarget.png </VAR> <br>
<br>
generates the following target:

@image html demoTarget.png


\attention
- Please vectorize the output image before printing the target (convert to pdf). Otherwise the resulting image is blurred. <br> e.g. use inkscape: Path->Trace Bitmap and disable all optimizations like edge smoothing (see also: \ref cut_target)
<br>
<br>
-  [page_ID], [target_width] and [target_height] are not used, they are only experimental feature
<br>
They can be used to define independent targets which can be printed on different boards. This feature is only present in hdmarker and not considered in the entire calibration. Thus always use the example code to generate the target and cut the needed regions from it.


### Bitdepth

Note that all methods in this library assume \b 8bit images. For higher bitdepths, rescale them to \b 8bit, using acutal image min/max values.

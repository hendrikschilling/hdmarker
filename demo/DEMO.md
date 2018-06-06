Demo marker detection {#demo_usage}
================

The following demo provides an c++ code example about how to extract marker information from an given input image.
<br>
&nbsp; <VAR>see cpp file:</VAR>  \ref Demo_hdmarker
<br>

\note
<VAR> extractMarker </VAR> can be found in the bin folder at the build location


Example Usage:
 @code
     extractMarker target.png Result.png; // Extracts marker with refinement and saves an image of all found markers
 @endcode
 @param [in] extractDemo.png // please use the the provided target image located in <STRONG> demo/extractDemo.png </STRONG>
 @param [out]  Result.png // will visualize the detected markers.
 <br>
<br>
<br>

<STRONG>Konsole output after running the example:</STRONG>
<br><br>
~~~
count 1 scale 16<br>
count 4 scale 8<br>
count 6 scale 4<br>
found 2 valid markers<br>
final score 8 corners<br>
[============================================================- 0 subs<br>
found 75 intitial corners<br>
added 44 corners=============================================/ 75 corners, range 1, added 44 checked 53 skipped 63<br>
added 52 corners============================================-] 119 corners, range 1, added 52 checked 67 skipped 70<br>
added 60 corners============================================|] 171 corners, range 1, added 60 checked 72 skipped 92<br>
added 68 corners=============================================- 231 corners, range 1, added 68 checked 84 skipped 104<br>
added 0 corners=============================================|] 299 corners, range 1, added 0 checked 0 skipped 0<br>
added 0 corners=============================================/] 299 corners, range 2, added 0 checked 0 skipped 0<br>
found 299 corners with extra search<br>
[===========================================================/] 0 subs<br>
found 0 intitial corners<br>
corners: 307<br>
homography rms: 0.066078<br>
findHomography: 307 inliers of 307 calibration points (100.00%)<br>
rms 0.065783 with full distortion correction<br>
[0.1355999434112855, 0.01433714395492138, 0, 0, 1.72106180276787e-06, 2.802771688347869,<br>
-0.01433389696960985, -1.720787289021778e-06, 0, 0, 0, 0, 0, 0]
~~~
<br>
<br>
<STRONG>Result images:</STRONG>
@image html off_hdm.jpg result.png shows the detected subpatterns
@image latex off_hdm.jpg result.png shows the detected subpatterns
<br>
<br>
@image html result.png shows the detected marker corners
@image latex result.png shows the detected marker corners
<br>

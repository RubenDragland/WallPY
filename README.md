# WallPY

Repo for handling AFM data from a wide range of different microscopes. 
Currently, this is enabled by combining Gwyddion and the python package gwyfile, but with added structure and code-awareness.
Additionally, .ibw-files can be read directly using the hystorian package. 

The repo presents a structured hdf5 file format for storing raw, sorted, and processed data.
It also holds a lot of functionality for analysing SPM data, such as flattening, filtering, crossectioning, and advanced algorithms.
For displaying data, the includes an overhead of the matplotlib library, making frequently used functionality more accessible and easier to use.



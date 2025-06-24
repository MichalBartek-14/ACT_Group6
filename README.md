# Analyzing GPR and photogrammetry data in Python and R
This project is an analysis of GPR (ground penetrating radar) data and photogrammetry data. 
The goal is to retrieve specific signals that belong to tree roots 
underground. The data was received in LAZ and PLY formats. In an 
ideal situation, the code would generate a 3D visualisation of tree roots on a desired location that is georeferenced, 
with preferably no signals of other clutter (pipes, watertanks, watertables, rocks etc.) that was picked up on by the GPR in the ground.
However, situations are rarely ideal and therefore the following code is more or less a collection of
attempts at achieving this 3D model instead of a solution. The attempts are listed in the 01_models folder. 

## Preprocessing
Preprocessing encorporates the conversion of the GPR signal from the large original location to the desired section for which the analysis is conducted later. The file is assigned the coordinate system. Preprocessing is mainly about changing the format of the data in a 
way that it can be easily visualized. _00_RetrievingGPRsignal.py_ attempts 
to convert LAZ file of the larger area to LAS of the desired validation trench location and add a georeference to it. _00_Visualising_LoadingLaz.py_
explores how to best visualize and load a LAZ/LAS files. _00_VisualisingMesh.py_ experiments 
with two ways of visualising a mesh validation trench file created through photogrammetry. 

![Mesh vis](https://github.com/MichalBartek-14/ACT_Group6/blob/master/pictures/Mesh.png)

## Models
The following three models are three attempts at retrieving/isolating tree root signals.
### 01a_ValueFiltering.py
The idea here is to isolate certain values that are corresponding to tree roots.
This is done by using **Numpy** to store the values of all points and then using masks 
(which are also **Numpy** data structures containing 'true/false') to be able to adjust 
just the points for which certain conditions are true. For example, filtering out just high values, could
be achieved with the following code: `mask_high = (gray > 0.8) & (z < -0.01)`
The gray value in this case is the sum or the RGB values that were retrieved from the LAZ file. 
After masking, the points for which the mask states "true" are then colored differently, 
and visualized with the **Open3d** package. After evaluation, it became clear that filtering out roots 
based on specific values is impossible because there is a strong overlap in values with other items. This method
was only able to take away those signals that are very different from roots. It did improve the visualisation of roots,
but wasn't able to filter out points with a high enough accuracy. There were always points present 
that couldn't have belonged to roots.

<div align="center">
  <img src="https://github.com/MichalBartek-14/ACT_Group6/blob/master/pictures/01a_Value_Filtering.png?raw=true" alt="Mesh vis" width="300">
</div>

### 01b_Kernel_and_DBSCAN.py

This method attempts to detect the tree roots by identifying root edges which would be seen in the reflectivity signal between two volumes (soil and tree root). 

First the values of the point cloud are loaded and then voxelised. Voxelisation is used for 2 main reasons: 

A) **Deal with the noise** which is present heavily in the GPR signal. Since our dataset is a GPR data converted into a lidar-like point cloud, it includes only the reflectivity of the individual pixels stored in rgb variable. Since the noise causes rapid unexplainable changes in the signal it is vital to smooth out these difference to detect actual object in ground. Therefore the voxelisation is applied and the voxel processed values simulate the reality better after voxelisation.

B) **Computationally more efficient** to make analyses on the voxels. Since the voxels aggregate the data into fewer entities with the original information included, it makes the calculations much faster.
This is especially efficient when processing larger files.

After the voxelisation the z-gradient is computed (with manual difference convolution kernel) to detect sharp edges between the voxels that would indicate the change in signal (potential shift of signal in other volume than soil).
Since the signal decreases with the depth of the soil due to attenuation, we apply compensation for the depth of the signal.

Once the voxels with sharp contrast are identified as potential root objects, the clustering from **DBSCAN** method is applied to detect the 
clusters. Since one root ideally shows similar values of the change (from soil to root) along its length the clustering should pick up its shape also laterally.
DBSCAN clustering (with fine tuned parameters) should thus potentially separate independent root instances.

<div align="center">
    <img src="https://github.com/MichalBartek-14/ACT_Group6/blob/master/pictures/01b_Edge.png?raw=true" alt="Edge vis" width="300">
</div>

### 01c_Slicing_Approach.py

Approach 01C similarly to 01B uses voxels as method to deal with the noise and make the computation process more efficient.
After the voxels are computed the z-gradient is computed, however contrary to the 01B, the z-gradient is not manually computed with the convolution kernel , but
with **Sobel** operator that is run on the XZ, YZ slices of the data retrieved from the point cloud.
This method also considers the magnitude of the z-gradient and Sobel computes gradients along both axes since it works from 2D slices.

<div align="center">
  <img src="https://github.com/MichalBartek-14/ACT_Group6/blob/master/pictures/01c_Slices.png?raw=true" alt="Slice vis" width="300">
</div>

## R compatibility

This file is an attempt at trying similar processing methods in R.
Unfortunately, we ran into computation related problems. R has its own system for allocating memory. 
02_R_Compatibility.R was run on a macbook with 16GB ram and a windows laptop with 16GB ram. For both cases, the systems crashed.

# Project Folder Structure
```
📁 00 preprocessing  
├── Retrieving_GPRsignal  
├── Visualizing_LoadingLaz 
├── VisualisingMesh
📁 01 models  
├── a_ValueFiltering  
├── b_Kernel_and_DBSCAN
├── c_Slicing_Approach
📁 02 R compatibility  
└── R_compatibility  
```
# how to use
Most python files in this directory are structured in the following way:
```
import x
import y
import z


def function_a():
    ...


def function_b():
    ...


def main():
    function_a()
    function_b()
    
    
if __name__ == "__main__":
    main()

```
It is possible to run different functions in the main function to figure 
out what each function does. Very rarely do functions depend on each other. 
In most cases they are standalone. Some functions are quite long and could 
benefit from subfunctions, however in this way, the different methods 
used are very clearly visible. 

# Credits & AI Statement
AI LLM's were used to generate quite a lot of Python code. This was deemed fitting because 
there was too little time to find appropriate packages and study their functionalities 
in an effective way to write our own code. However, Most time was spent on modifying the generated code to our data. 
This entails changing filter values, tweaking clustering parameters and tinkering 
with matplotlib visualisations of slices of the data. 

The same goes for the R code, but this code was parted with much faster because of 
the problem with memory allocation, which is a core problem of the R coding language. The code was still included, to 
serve as an example that R is not the way to go for this kind of problem.

Credits go out to [Michal Bartek](https://github.com/MichalBartek-14) and [Mees Ike](https://github.com/M-Ike007) 
for the Python code and to [Eleanor Hammond](https://github.com/Eh6708) and [Dimitra tzedaki](https://github.com/Dimitra-tzedaki) 
for the R code.
# License
This project falls under an MIT License 

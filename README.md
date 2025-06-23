TODO:
1. should include some nice pictures
2. should add citations to papers
3. should update the file structure (the names are a bit wrong)
4. add license

# Analyzing GPR and photogrammetry data in Python and R
this project is an analysis of GPR (ground penetrating radar) data and photogrammetry data. 
The goal is to retrieve specific signals that belong to tree roots 
underground. The data was received in LAZ and PLY formats. In an 
ideal situation, the code below would generate a 3D model of tree roots that is georeferenced, 
with no signals of other clutter (pipes, watertanks, watertables, rocks etc.) that was picked up on by the GPR in the ground.
However, situations are rarely ideal and therefore the following code is more or less a collection of
attempts at achieving this 3D model instead of a solution. The attempts are listed in the 01_models folder. 
It will be explained in further detail that some of the methods used are similar to
literature, which will be cited. 
## Preprocessing
Preprocessing is mainly about changing the format of the data in a 
way that it can be easily visualized. _00_RetrievingGPRsignal.py_ attempts 
to convert LAZ to LAS and add a georeference to it. _00_VisualisingLAZ.py_
explores how to best visualize a LAZ file. _00_VisualisingMesh.py_ experiments 
with two ways of visualising a mesh file created through photogrammetry. 
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
### 01b_EdgeDetectionModel.py
...j4opinvrpknlfjqefn3klkne
cite papers
### 01c_Slices_Approach.py
...
cite papers
## R compatibility
...
explain why it didn't work and that this is a fundamental flaw with R code
# Project Folder Structure
```
📁 00 preprocessing  
├── retrieving_GPR  
├── visualizingLaz 
├── VisualisingMesh
📁 01 models  
├── ValueFiltering  
├── EdgeDetectionModel
├── Slices_Approach
📁 02 R compatibility  
└── R_compatibility  
```
# how to use

# Credits

# License


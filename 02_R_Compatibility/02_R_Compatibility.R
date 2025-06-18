#install.packages("geomorph")
library(geomorph)
#read.ply("~/Documents/WAGENINGEN/GIS_RS_Integration/ACT/WUR_ACT_PG_250515/Trial_trenches/3D_modellen/51025869_Proefsleuf1_DaCosta_16042025_JCTSimons.ply", ShowSpecimen = TRUE, addNormals = TRUE)

#install.packages(c("lidR", "dbscan", "rgl", "RColorBrewer", "pracma"))

library(lidR)
library(dbscan)
library(rgl)
library(RColorBrewer)
library(pracma)

library(parallel)
library(abind)

#las <- readLAS("~/Documents/WAGENINGEN/GIS_RS_Integration/ACT/WUR_ACT_PG_250515/LAZ_Euroradar/Bomen-1-6.laz")

#make a function to read in the laz file
load_las_point_cloud <- function(file_path) {
  las <- readLAS(file_path)
  if (is.empty(las)) stop("Empty LAS file or could not load.")
  
  points <- cbind(las@data$X, las@data$Y, las@data$Z)
  reflectivity <- las@data$R / max(las@data$R, na.rm = TRUE)
  return(list(points = points, reflectivity = reflectivity))
}


# A function to voxelize the reflectivity data
voxelize_reflectivity <- function(points, reflectivity, voxel_size) {
  min_bound <- apply(points, 2, min)
  max_bound <- apply(points, 2, max)
  dims <- ceiling((max_bound - min_bound) / voxel_size)
  
  voxel_grid <- array(NA, dim = dims)
  count_grid <- array(0, dim = dims)
  
  indices <- floor((t(t(points) - min_bound)) / voxel_size) + 1  # R index starts at 1
  
  for (i in seq_len(nrow(indices))) {
    idx <- indices[i, ]
    if (all(idx > 0 & idx <= dims)) {
      if (is.na(voxel_grid[idx[1], idx[2], idx[3]])) {
        voxel_grid[idx[1], idx[2], idx[3]] <- reflectivity[i]
      } else {
        voxel_grid[idx[1], idx[2], idx[3]] <- voxel_grid[idx[1], idx[2], idx[3]] + reflectivity[i]
      }
      count_grid[idx[1], idx[2], idx[3]] <- count_grid[idx[1], idx[2], idx[3]] + 1
    }
  }
  
  voxel_grid <- voxel_grid / ifelse(count_grid == 0, NA, count_grid)
  return(list(voxel_grid = voxel_grid, min_bound = min_bound, voxel_size = voxel_size))
}

# A function to compute Z-gradient with compensation
#compute_z_gradient <- function(voxel_grid, compensation_power, base_threshold) {
  dims <- dim(voxel_grid)
  filled <- voxel_grid
  filled[is.na(filled)] <- 0
  
  kernel <- array(0, dim = c(3, 3, 3))
  kernel[2, 2, 1] <- -1
  kernel[2, 2, 3] <- 1
  
  grad_z <- convn(filled, kernel, type = "same")
  
  z_indices <- array(rep(1:dims[3], each = dims[1]*dims[2]), dim = dims)
  compensation <- (z_indices)^compensation_power
  grad_z <- abs(grad_z * compensation)
  
  cat(sprintf("grad_z stats: min=%.6f, max=%.6f, mean=%.6f\n", min(grad_z), max(grad_z), mean(grad_z)))
  edges <- which(grad_z > base_threshold, arr.ind = TRUE)
  return(edges)
}

# Actually, the above function does not work. Use this instead: function to compute Z-gradient with compensation
compute_z_gradient2 <- function(voxel_grid, compensation_power, base_threshold) {
  filled <- voxel_grid
  filled[is.na(filled)] <- 0
  
  dims <- dim(filled)
  grad_z <- array(0, dim = dims)
  
  # Compute forward difference along Z
  grad_z[, , 1:(dims[3] - 1)] <- abs(filled[, , 2:dims[3]] - filled[, , 1:(dims[3] - 1)])
  
  # Depth compensation
  z_indices <- array(rep(1:dims[3], each = dims[1] * dims[2]), dim = dims)
  compensation <- (z_indices + 1)^compensation_power
  grad_z <- grad_z * compensation
  
  cat(sprintf("grad_z stats: min=%.6f, max=%.6f, mean=%.6f\n", min(grad_z), max(grad_z), mean(grad_z)))
  
  # Threshold
  edges <- which(grad_z > base_threshold, arr.ind = TRUE)
  return(edges)
}
# Make a function for clustering using DBSCAN
cluster_voxels <- function(coords, eps, min_samples) {
  if (nrow(coords) == 0) return(list(coords = coords, labels = integer(0)))
  db <- dbscan(coords, eps = eps, minPts = min_samples)
  return(list(coords = coords, labels = db$cluster))
}

# A function for creating a labeled voxel grid
create_labeled_voxel_grid <- function(shape, coords, labels) {
  label_grid <- array(0, dim = shape)
  for (i in seq_len(nrow(coords))) {
    if (labels[i] != 0) {
      idx <- coords[i, ]
      label_grid[idx[1], idx[2], idx[3]] <- 1
    }
  }
  return(label_grid)
}

# Make a function for visualizing clusters with rgl
visualize_clusters <- function(coords, labels, min_bound, voxel_size) {
  colors <- brewer.pal(8, "Set1")
  n_colors <- length(colors)
  valid <- labels > 0
  labels <- labels[valid]
  coords <- coords[valid, , drop = FALSE]
  
  points <- sweep(coords - 1, 2, voxel_size, "*")
  points <- sweep(points, 2, min_bound, "+")
  
  color_vals <- colors[(labels %% n_colors) + 1]
  rgb_cols <- t(col2rgb(color_vals)) / 255
  
  open3d()
  plot3d(points, col = rgb_cols, size = 3, xlab = "X", ylab = "Y", zlab = "Z")
}

min_bound <- apply(points, 2, min)
max_bound <- apply(points, 2, max)
dims <- ceiling((max_bound - min_bound) / voxel_size)
voxelize_reflectivity_parallel <- function(points, reflectivity, voxel_size, n_cores = detectCores() - 1) {
  min_bound <- apply(points, 2, min)
  max_bound <- apply(points, 2, max)
  dims <- ceiling((max_bound - min_bound) / voxel_size)
  
  chunk_indices <- split(seq_len(nrow(points)), cut(seq_len(nrow(points)), n_cores, labels = FALSE))
  
  # Worker function
  voxel_worker <- function(chunk_idx) {
    voxel_grid <- array(NA, dim = dims)
    count_grid <- array(0, dim = dims)
    
    for (i in chunk_idx) {
      idx <- floor((points[i, ] - min_bound) / voxel_size) + 1
      if (all(idx > 0 & idx <= dims)) {
        if (is.na(voxel_grid[idx[1], idx[2], idx[3]])) {
          voxel_grid[idx[1], idx[2], idx[3]] <- reflectivity[i]
        } else {
          voxel_grid[idx[1], idx[2], idx[3]] <- voxel_grid[idx[1], idx[2], idx[3]] + reflectivity[i]
        }
        count_grid[idx[1], idx[2], idx[3]] <- count_grid[idx[1], idx[2], idx[3]] + 1
      }
    }
    return(list(voxel_grid = voxel_grid, count_grid = count_grid))
  }
  
  cl <- makeCluster(n_cores)
  clusterExport(cl, varlist = c("points", "reflectivity", "voxel_size", "min_bound", "dims"))
  partials <- parLapply(cl, chunk_indices, voxel_worker)
  stopCluster(cl)
  
  # Merge voxel grids and count grids
  final_voxel <- array(NA, dim = dims)
  final_count <- array(0, dim = dims)
  
  for (part in partials) {
    v <- part$voxel_grid
    c <- part$count_grid
    nan_mask <- is.na(final_voxel) & !is.na(v)
    final_voxel[nan_mask] <- 0
    v[is.na(v)] <- 0
    final_voxel <- final_voxel + v
    final_count <- final_count + c
  }
  
  # Average reflectivity
  final_voxel <- final_voxel / ifelse(final_count == 0, NA, final_count)
  
  return(list(voxel_grid = final_voxel, min_bound = min_bound, voxel_size = voxel_size))
}

#Now running the main code using the above-made functions
file_path <- "~/Documents/WAGENINGEN/GIS_RS_Integration/ACT/WUR_ACT_PG_250515/LAZ_Euroradar/Bomen-1-6.laz"

# Parameters
voxel_size <- 0.02
compensation_power <- 0.1
base_threshold <- 500
eps <- 4
min_samples <- 10

las_data <- load_las_point_cloud(file_path)
points <- las_data$points
reflectivity <- las_data$reflectivity

#voxel_size of 0.5 (defined in the vox_data function arguments below) is much too large to get meaningful voxels. 
#Therefore, no edges are detected in the subsequent code. But a smaller voxel size results in memory issues (even when using parellisation)
vox_data <- voxelize_reflectivity_parallel(points, reflectivity, voxel_size = 0.5, n_cores = 8)
voxel_grid <- vox_data$voxel_grid
min_bound <- vox_data$min_bound
voxel_size <- vox_data$voxel_size

non_empty_voxels <- sum(!is.na(voxel_grid))

#Computing Z-gradient (with depth compensation)
edge_voxels <- compute_z_gradient2(voxel_grid, compensation_power, base_threshold)

if (nrow(edge_voxels) == 0) {
  cat("No edge voxels detected. Adjust gradient threshold or check data.\n")
  return()
}

#Clustering edge voxels with DBSCAN
cluster_result <- cluster_voxels(edge_voxels, defined_eps, defined_min_samples)
coords <- cluster_result$coords
labels <- cluster_result$labels
num_clusters <- length(unique(labels[labels > 0]))

voxel_labels <- create_labeled_voxel_grid(dim(voxel_grid), coords, labels)

#Visualizing the clusters
visualize_clusters(coords, labels, min_bound, voxel_size)


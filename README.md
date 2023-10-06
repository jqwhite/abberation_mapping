# Aberration mapping 

Python based notebooks for analyzing optical aberration modes in confocal light microscopy using Z-stacks of standard fluorescent beads.

See https://en.wikipedia.org/wiki/Optical_aberration

Based on Debayan Saha's Ph.D. thesis [ref].

Excecute the notebooks in order:

1. `01_process_patches.ipynb`
2. `02_analyse_zola_results.ipynb`

## Further details

The `01_process_patches.ipynb` selects the beads, processes the patches around the beads, and runs a PSF model on each bead.

**Logistics**

The logistics of the `01_process_patches.ipynb` notebook are complicated.  

It is intended to run on a local computer that has `ssh` access to the CBG cluster (falcon).

It does the pre-processing locally and generates a folder `patches_for_zola` with 
50 sub-folders, each containing TIFF files of a Z-stack of each selected bead 
(eg `planes_14_z_35_y_1227_x_369.tif`) .

The notebook copies these local bead files to the remote (`falcon`), 
along with a fiji macro and a bash shell script necessary for processing 
on the HPC cluster.

To process the beads, the notebook starts a number of batch jobs on gpu nodes of the cluster as follows:

` ! ssh falcon sbatch {remote_zola_script_path} '{remote_fiji_path} {remote_fiji_macro_path} {remote_imageJ_macro_parameter_string}'   `

1. SLURM calls a bashscript (currently `bashscript_fiji3.sh`) to start the processes.  The script takes as paramters the location of the ImageJ binary `remote_fiji_path`, the macro to be run `zola_macro_cluster_params_passed.ijm` and a string containing the parameters for the PSF modeling plugin (the plugin is `ZOLA_-0.2.8-SNAPSHOT.jar`).
2. The bashscript starts a headless instance of ImageJ, 
3. The headless instance of ImageJ is passed a command line argument that runs a macro to start the Zola analysis.
3. The parameters for the plugin and the filename for the results are passed as a single string which is parsed by the macro into a filepath on the remote for the results, and the parameters needed for the "Calibration: PSF modeling" plugin.   

After the SLURM jobs are done running, there is a folder called `zola_raw` in the `patches_for_zola` folder on the remote.  For each bead, there is a `.json` file with the amplitudes of each Zernike mode, as a list, along with some metadata. 




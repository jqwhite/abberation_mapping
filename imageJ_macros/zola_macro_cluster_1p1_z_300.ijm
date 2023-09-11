/*
 * @Debayan MPI-CBG
 */

file_path = getArgument();
//file_path = "/Volumes/project-dsaha/insitu_psf_data/droso/2021_02_01_droso_low_mag_5/time_0min/ch_gfp/depth_0_20_um/processed/y_0_x_0_good_1/patches_0"
print("In macro");
print(file_path);
suffix = ".tif";
processFolder(file_path); 
// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	start = getTime();
	print(input)
	list = getFileList(input);
	list = Array.sort(list);
	print(list.length)
	for (i = 0; i < list.length; i++) {
		print(list[i]);
		if(File.isDirectory(input + File.separator + list[i])){
			l = substring(list[i], 0, lengthOf(list[i])-1);
			processFolder(input + File.separator + l + File.separator);
		}
		if(endsWith(list[i], suffix)){
			processFile(input, list[i]);
		}
	}
	print("Totoal Time: "+(getTime()-start)/1000);   
}

function processFile(input, file) {
	print("Processing: "+ input + File.separator + file);
	if (indexOf(input, "patches") > 0) {
		if (!File.exists(File.getParent(input) + File.separator + "zola_raw")){
			File.makeDirectory(File.getParent(input) + File.separator + "zola_raw"); 
		}
	//File.makeDirectory(input + File.separator + "result")
	output = File.getParent(input) + File.separator + "zola_raw"+ File.separator + substring(file, 0, indexOf(file, '.tif'))+'.json';
	print(output);

	open(input + File.separator + file);
	//run("Subtract Background...", "rolling=10 stack");
	run("Properties...", "channels=1 slices="+nSlices+" frames=1 unit=pixel pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");
	setSlice((nSlices+1)/2); 
	makePoint(getWidth()/2, getHeight()/2);
	run(" Calibration: PSF modeling", "run_on_gpu gain=1 pixel_size=226 z_step=300 bead_moving=[far -> close to objective] numerical_aperture=1.1 immersion_refractive=1.33 wavelength=0.515 patch_size=16 zernike_coefficient=[Zernike 15 coefs] iteration=30 result_calibration_file="+output);
	run("Close All");
	}
	else{print("Np patches");
	}
}
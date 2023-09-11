/*
 * @Debayan MPI-CBG
 */

file_path = getArgument();
print("In macro");
print(file_path);
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
		if(endsWith(list[i], ".tif")){
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
	output = File.getParent(input) + File.separator + "zola_raw"+ File.separator + substring(file, 0, indexOf(file, '.tif'))+'.json';
	print(output);

	open(input + File.separator + file);
	setSlice((nSlices+1)/2); 
	makePoint(getWidth()/2, getHeight()/2);
	wid = 64;
	run(" Calibration: PSF modeling", "run_on_gpu gain=1 pixel_size=113 z_step=100 bead_moving=[far -> close to objective] numerical_aperture=1.0 immersion_refractive=1.33 wavelength=0.515 patch_size="+wid+" zernike_coefficient=[Zernike 15 coefs] iteration=20 result_calibration_file="+output);
	run("Close All");
	}
	else{print("No patches");
	}
}
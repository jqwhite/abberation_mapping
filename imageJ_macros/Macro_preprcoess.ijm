saveAs("Tiff", "C:/Users/lmfuser/Desktop/uns.tif");
run("Z Project...", "projection=[Max Intensity]");
run("Find Maxima...", "prominence=300 output=List");
saveAs("Results", "C:/Users/lmfuser/Desktop/Results.csv");
close();
run("Subtract Background...", "rolling=2 sliding stack");
run("Gaussian Blur 3D...", "x=0.5 y=0.5 z=0.5");
saveAs("Tiff", "C:/Users/lmfuser/Desktop/uns_bck_sub_blur.tif");
close();


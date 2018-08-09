PImage original;
PImage convolved;

// Define a sharpening kernel; To play with different kernels, try this: http://en.wikipedia.org/wiki/Kernel_(image_processing)
int [][] K = new int [][] {{0,-1,0},
                           {-1,5,-1},
                           {0,-1,0}}; 


void setup(){
    // Normalise your kernel K at first if the sum of weights does not equal to 1
    
    // Read in the source image
    original = loadImage("operaHouse.jpg");
    
    // Initialize convolved result image
    convolved = createImage(original.width, original.height, RGB);
    size(550, 183);
        
    for (int x=1; x<original.width-1; x++)
        for (int y=1; y<original.height-1; y++){
            color c = conv(original, K, x, y);
            int loc = y * original.width + x; // Convert from 2D coordinates to 1D
            convolved.pixels[loc] = c;
        }
}


void draw(){
    image(original, 0, 0);
    image(convolved, original.width + 1, 0);
}

color conv(PImage img, int[][] K, int x, int y){
    int radius = int(K.length / 2);
 
    // You need to convolve each RGB channel separately
    float rsum = 0;
    float gsum = 0;
    float bsum = 0;
    // Iterate through each element of the kernel as well as the image   
    for (int kx=0, ix=x-radius; kx<K.length; kx++, ix++)
        for (int ky=0, iy=y-radius; ky<K.length; ky++, iy++) {
             int loc = iy * img.width + ix;
             rsum += K[ky][kx] * red(img.pixels[loc]);
             gsum += K[ky][kx] * green(img.pixels[loc]);
             bsum += K[ky][kx] * blue(img.pixels[loc]);
        }
    
    return color(rsum, gsum, bsum); // Note: the normalisation was done within kernel
}
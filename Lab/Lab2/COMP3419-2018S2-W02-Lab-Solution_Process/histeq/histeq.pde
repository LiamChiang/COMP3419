PImage oimg; // orignial img
PImage gimg; // target img
PImage himg;

void setup() {
  oimg = loadImage("sample_image.jpg");
  gimg = createImage(oimg.width, oimg.height, RGB); // gray scaled
  himg = createImage(oimg.width, oimg.height, RGB); // equalised
  
  size(1024, 384); // display gray & equalised img at the same time

  // convert the img to gray scale at first
  for (int y = 0; y < oimg.height - 1; y++)
    for (int x = 0; x < oimg.width - 1; x++){
      int loc = y * oimg.width + x;
      color c = oimg.pixels[loc];
      int greyValue = (int)(0.212671 * red(c) + 0.715160 * green(c) 
                                        + 0.072169 * blue(c));
      color greyColor = color(greyValue);
      gimg.pixels[loc] = greyColor;      
    }

  int [] histg = new int [256];

  // get the distribution
  for (int y = 0; y < oimg.height; y++)
    for (int x = 0; x < oimg.width; x++){
      int loc = y * gimg.width + x;
      color c = gimg.pixels[loc];
      int greyValue = (int) red(c);
      histg[greyValue]++;
    }
    
  int [] cdf = new int [256];  
  int cdfmin = histg[0];
  cdf[0] = histg[0];
  for (int i = 1; i < histg.length; i++){
    cdf[i] = cdf[i - 1] + histg[i];
  }
    
  // ** Key thing: caculate the new value for each gray scale
  int [] h = new int [256];
  for (int i = 0; i < histg.length; i++){
    h[i] = (int)((float)(cdf[i] - cdfmin) * 255 / 
                  (float)(oimg.width * oimg.height - cdfmin));
  }
  
  // modify the original image to equalised image
  // get the histogram
  for (int y = 0; y < oimg.height; y++)
    for (int x = 0; x < oimg.width; x++){
      int loc = y * oimg.width + x;
      color c = gimg.pixels[loc];
      int grayValue = (int) red(c);
      himg.pixels[loc] = color(h[grayValue]);      
    } 
  
}

void draw(){
  image(gimg, 0, 0);
  image(himg, oimg.width, 0);
  save("equalised.jpg");
}
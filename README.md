# Image Pattern Search on CUDA

This project is developed for the term project of the CMP641 - Software Development for Parallel Computers lecture taaught by [Kayahan Mustafa Imre](https://web.cs.hacettepe.edu.tr/~kimre/).

In this implementation, 5x5 pattern search on a 16-bit grayscale image is implemented. Search is done on upper and lower triangle of the image matrix seperately. This implementation uses CUDA to parallelize the search algorithm.

The algorithm resembles basically convolution process except it makes comparison instead of multiplication while searching the pattern on the image matrix.

## Dependencies

The implementation needs only [lodepng](https://lodev.org/lodepng/) PNG image processing library, yet it is embedded into the project.

## Compilation
### Windows
You may need [GnuMake32](https://gnuwin32.sourceforge.net/downlinks/make.php). Please download and install it first if you haven't.

Then, in the project root folder, run

```
make
```

After invoking the "make", it will start to compile the project according to your OS. 

#### Run

After the compilation, you can run the program with a command line statement like this:

```
make run ARGS="../images/image_8192.png ../images/pattern_8192_1.png"
```

Or

```
psearch.exe ../images/image_8192.png ../images/pattern_8192_1.png
```

### Linux
PLease make sure you installed build-essential, llvm-14, lldb-14 and clang-14 packages. If you haven't you can install it by following:

```
sudo apt update
sudo apt install build-essential llvm-14 lldb-14 clang-14
```

Then, in the project root folder, run

```
make
```

After invoking the "make", it will start to compile the project according to your OS. 

#### Run

After the compilation, you can run the program with a command line statement like this:

```
make run ARGS="./images/image_8192.png ./images/pattern_8192_1.png"
```

Or

```
./build/release/psearch ./images/image_8192.png ./images/pattern_8192_1.png
```

## Resources
You may need test images. You can download [here](https://drive.google.com/file/d/14GBRgUel3fOdJVDOdlZ3HJh3_1vUsWZt/view?usp=sharing). These all random checker images of different sizes.

You can also create your own test images [here](https://onlinetools.com/image/generate-random-image). Note that you may need some edit & conversion for the images newly created. Firstly you sould convert color space to 16-bit Grayscale/Alpha. Then, you should extract some 5x5 patches from the image for the test pattern. 

If you need an image editor I recommend you free and open-source [Krita](https://krita.org/en/) image editing & painting application.


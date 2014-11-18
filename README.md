##Dependencies
1. pillow
2. Numpy
3. matplotlib

The source code can be run under windows or linux with python 2.7+ and the libraries above.

##Directory structure

    ./
    ├── doc
    │   └── report.pdf  (the report)
    ├── img
    │   ├── best.png  (SA-denoised result)
    │   ├── flipped.png  (flipped image)
    │   ├── ICM-energy-time.png  (time-energy plot of ICM)
    │   ├── icm.png  (ICM-denoised result)
    │   ├── in.png  (original image)
    │   └── SA-energy-time.png  (time-energy plot of SA)
    └── src
        ├── binarize.py  (script to convert the input to a binary image)
        ├── count.py  (script to evaluate the denoised results)
        ├── denoise.py  (script to denoise the results with either SA or ICM)
        ├── flip.py  (script to flip the image)
        └── util.py  (utilities. Configurable arguments are defined here.)


##How to generate the results

Note: python scripts should be run under the `src` directory. All images will be placed under the `img` directory.

1. Place the original image called `in.png` under `img` directory.
2. Enter the `src` directory, run `python binarize.py`. It will convert the `in.png` to a binary image and overwrite it.
3. Run `python flip.py`, which will generate the flipped image named `flipped.png`.
4. Run `python denoise.py` to denoise the flipped image using simulated annealing. The result will be named `best.png`. Temporary results (`temp-*.png`) and the time-energy plot (`SA-energy-time.png`) will be saved, too.
5. Run `python denoise.py -m "ICM" -o "icm.png"` to denoise the flipped image using ICM. The result will be named `icm.png`. Temporary results (`icm-temp-*.png`) and the time-energy plot (`ICM-energy-time.png`) will be saved, too.
6. Run `python count.py` to see how many pixels of the output of SA agree to the original image. To do the same evaluation for ICM, run `python count.py -o "icm.png"`

You can run `python denoise.py -h` to see what arguments are configurable.

##About
* [Github repository](https://github.com/joyeecheung/simulated-annealing-denoising)
* Time: Nov. 2014

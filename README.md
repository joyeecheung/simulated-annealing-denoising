##Dependencies
1. pillow
2. Numpy
3. Scipy >= 0.14.0
3. matplotlib

##How to generate the results
Enter the `src` directory, run `python main.py`. It will use the `02.png` under `img` directory as default to produce the results.

To use another source image, put the image under `img` directory, then run `python main.py -s [filename]`. For example, to use a `03.png`, put it under `img`, then run `python main.py -s 03.png`.

The results will show up in `result` directory.

##Directory structure

    .
    ├─ README.md
    ├─ requirements.txt
    ├─ doc
    │   └─ report.pdf
    ├─ img (source image)
    │   └─ 02.png
    ├─ result (the results)
    │   ├─  equalize.png
    │   ├─  filter-laplacian.png
    │   ├─  filter-smooth-11-11.png
    │   ├─  filter-smooth-3-3.png
    │   ├─  filter-smooth-7-7.png
    │   ├─  filter-sobel-0.png
    │   ├─  filter-sobel-1.png
    │   ├─  hist-equalize.png
    │   ├─  hist.png
    │   ├─  patch-50-50-0.png
    │   ├─  patch-50-50-1.png
    │   ├─  patch-50-50-2.png
    │   ├─  patch-50-50-3.png
    │   ├─  patch-50-50-4.png
    │   ├─  patch-50-50-5.png
    │   ├─  patch-50-50-6.png
    │   ├─  patch-50-50-7.png
    │   ├─  patch-96-64-0.png
    │   ├─  patch-96-64-1.png
    │   ├─  patch-96-64-2.png
    │   ├─  patch-96-64-3.png
    │   ├─  patch-96-64-4.png
    │   ├─  patch-96-64-5.png
    │   ├─  patch-96-64-6.png
    │   └─  patch-96-64-7.png
    └─src (the python source code)
        ├─  main.py (entry point)
        ├─  hist.py
        ├─  patch.py
        ├─  filter.py
        └─  util.py

##About
* [Github repository](https://github.com/joyeecheung/simulated-annealing-denoising)
* Time: Nov. 2014
**Pilot Project in GCS**
======

Before starting, run "BaseEnvironment.bat" file which will create anaconda environment using the environment.yml file with add dependencies. Once virtual environment is created, run the code under that.
If you just need dependencies, refer requirements.txt and install it via pip3

The model folder does not contain weights, please make sure to download the weights as name [model folder].weights before execution

Install FFMPEG software additionally and add its "bin" directory to path environment variable

## **Execution Steps**

##### Run each step as standalone

> - python launch.py -f [file name path] -p metadata
> - python launch.py -f [file name path] -p scan
> - python launch.py -f [file name path] -p prepare
> - python launch.py -f [file name path] -p model -m [model folder path]
> - python launch.py -f [file name path] -p wrap 
> - python launch.py -f [file name path] -p output
> - python launch.py -f [file name path] -p validate
> - python launch.py -f [file name path] -p tracker 


##### To run all above steps in order

Ignore the process switch and all steps will run in order

> - python launch.py -f [file name path] -m [model folder path]


##### To restart from a failed step or process. 
The "resume" flag would start from the step provided (in this case runmodel) and continue executing subsequent steps

> - python launch.py -f [file name path] -p runmodel -m [model folder path] --resume
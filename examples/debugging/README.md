# Debugging

To find a bug or to understand what the pipeline is doing, it is possible to run BlenderProc from inside Blender.
Beforehand the config should have been run at least once via the `run.py` script to make sure the correct blender version and the required additional python packages are installed.


To start the pipeline from inside Blender, the `src/debug.py` script has to be opened and executed in Blender's scripting tab:

![](blender.png)

Per default this loads and runs the config file located in `examples/debugging/config.yaml`.
As blender does not allow passing arguments to the script, all paths need to be defined inside the configuration file. 



python3 run.py examples/debugging/config.yaml examples/semantic_segmentation/scene.blend examples/debugging/output



======Adding {} fg objects======= 4
added sample_fg_objects2 : textured.065
added sample_fg_objects2 : textured.066
added sample_fg_objects2 : textured.067
ADDING FOREGROUND OBJECTS
Info: Deleted 1 object(s)
Info: Deleted 1 object(s)
crop rule passed, visible area =  1.2392787204598708
marking  <bpy_struct, Object("textured.064") at 0x7fd482ad4c08>  placed!!
#### Finished - Running module CurriculumSampler (took 129.348 seconds) ####
#### Finished - Running blender pipeline (took 152.330 seconds) ####
Traceback (most recent call last):
  File "/home/andrew/ai/upwork_clients/Ahsan/NewBlenderProc/src/run.py", line 40, in <module>
    pipeline.run()
  File "./src/main/Pipeline.py", line 86, in run
    module.run()
  File "./src/composite/CurriculumSampler.py", line 259, in run
    verbose=True)
  File "./src/composite/CurriculumSampler.py", line 648, in scale_and_place_fg_objs
    base_polys[r_idx].get_area()) > \
ZeroDivisionError: float division by zero

Error: script failed, file: '/home/andrew/ai/upwork_clients/Ahsan/NewBlenderProc/src/run.py', exiting.
Cleaning temporary directory
going to place again
crop rule passed, visible area =  0.9945911052140002
overlap rule not violated with  <bpy_struct, Object("textured.064") at 0x7f63a8c2a808>  at only  0 intersection
overlap rule not violated with  <bpy_struct, Object("textured.065") at 0x7f63a8c2b408>  at only  0 intersection
overlap rule not violated with  <bpy_struct, Object("textured.066") at 0x7f63a9ebb008>  at only  0 intersection
marking  <bpy_struct, Object("textured.067") at 0x7f63a9ebbc08>  placed!!
#### Finished - Running module CurriculumSampler (took 448.379 seconds) ####
#### Finished - Running blender pipeline (took 471.001 seconds) ####
Traceback (most recent call last):
  File "/home/andrew/ai/upwork_clients/Ahsan/NewBlenderProc/src/run.py", line 40, in <module>
    pipeline.run()
  File "./src/main/Pipeline.py", line 86, in run
    module.run()
  File "./src/composite/CurriculumSampler.py", line 260, in run
    verbose=True)
  File "./src/composite/CurriculumSampler.py", line 666, in scale_and_place_fg_objs
    base_polys[r_idx].get_area()) > \
ZeroDivisionError: float division by zero



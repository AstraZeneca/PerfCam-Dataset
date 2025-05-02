# Details of Experiment 1

## Cameras

The videos included in this dataset had been taken by different cameras:

1. '1.mp4': Apple M3 MacBook Pro with a FaceTime HD Camera with a moderate FPS
2. '2.mp4': Nicla Vision camera with a very low FPS
3. '3.mp4': Intel RealSense d435i RGB video with a very high FPS
4. '3_depth.mp4':  Intel RealSense d435i depth video with a very high FPS
5. '4.mp4': Pixel 7 Pro phone with Android v15 with highest FPS possible

## Description

In this experiment, we tried to emulate a scenario where there are 3 nodes:

1. A start node, where products start their journey on the conveyor belt
2. A QA node, where bad products is being removed from the line
3. An end node, where good products are ending their journey and ready to be shipped

The experimental scenario is structured so that products move along a conveyor belt from Node 1 to Node 3, passing through two edges. The data collected from this experiment is intended to serve as a benchmark for creating digital twins, training models for object detection and counting, and analyzing operator behavior.


## Events

There were several events that occurred during this experiment, categorized into the following pairs, specifically related to the production line operations:

| Event Start                        | Event End / Outcome                     | Description                                                                                 |
|------------------------------------|-----------------------------------------|---------------------------------------------------------------------------------------------|
| Cross adjustment starts            | Cross adjustment ends                   | The operator begins and finishes moving a product over another product during adjustment.    |
| Crawled inside room                | Crawled outside room                    | The operator crawls into and out of the room where a piece of equipment or node is located. |
| Camera rotation starts             | Camera rotation ends                    | The camera begins and completes its rotation to focus on different parts of the production line. |
| Double handed adjusting starts     | Double handed adjusting ends            | The operator uses both hands to adjust a product, beginning and completing the task.        |
| Line jamming starts                | Line jammed                             | The production line starts to jam and becomes completely obstructed.                        |
| Line preparing to start            | Line started                            | Preparations for starting the production line are made, and the line begins operating.      |
| Door closing starts                | Door opening starts, Door closed        | The process of closing the door to the room where a node is located begins and either opens again or closes completely. |
| Exiting room                       | Exited room                             | The operator completes leaving the room where equipment or a node is located.               |
| Entering room                      | Entered room                            | The operator enters the room where equipment or a node is located.                |
| Preparing to focus starts          | Focused                                 | The operator begins to focus on a specific node on the production line to perform their job, achieving full focus.      |
| Door opening starts                | Door closing starts, Door opened        | The door to the room where a node is located begins to open and either closes again or opens fully. |
| Double handed pickup starts        | Double handed pickup ends               | The operator uses both hands to pick up a product, starting and completing the task.        |
| Line unjamming                     | Line unjammed                           | The process of clearing jams in the production line is completed.                           |
| Pickup starts                      | Pickup ends                             | The operator begins and completes the pickup of a product from the line.                    |
| Leaving starts                     | Left                                    | The operator begins leaving and eventually leaves entirely, becoming untrackable.|
| Triple pickup starts               | Triple pickup ends                      | The operator begins and completes picking up three products simultaneously.                 |
| Distracting starts                 | Distracted                              | The operator becomes distracted from focusing on their job on the production line             |
| Line preparing to stop             | Line stopped                            | Preparations for stopping the production line are made, and the line is halted.             |
| Adjusting starts                   | Adjusting ends                          | The operator starts and finishes adjusting a product on the line at a specific node.        |
| Returning starts                   | Returned                                | The operator begins the process of returning to a node on the production line and becomes trackable.        |
| Double pickup starts               | Double pickup ends                      | The operator begins and completes picking up two products simultaneously.                   |
| Triple-double pickup starts        | Triple-double pickup ends               | The operator begins and completes a complex pickup involving multiple products.             |


The `ground-truth-events.csv` file was manually crafted to precisely mark the timestamps for each of these events, including their start and end times. Subsequently, using the `scripts/generate_event_cuts.py`, videos of every individual event were produced and are available in the `videos/event_cuts` folder. These can be utilized for training or evaluating various machine learning-based action detection algorithms and tests.

## Counting and stats

There are two files that indicate the ground truth for products moving along each edge: `ground-truth-products-edge1.csv` (counted using camera 4) and `ground-truth-products-edge2.csv` (counted using camera 3). These counts were performed manually by a human and can be used to measure the accuracy of ML-based counting and tracking.

The red areas indicated below represent the regions where products passed. Each `ground-truth-products-edgeX.csv` CSV file contains the exact time when the products passed.

<img src="miscellaneous/area1.png" alt="The area where the ground truth data has been collected for edge 1." style="max-width: 500px; width: 100%;">

<img src="miscellaneous/area2.png" alt="The area where the ground truth data has been collected for edge 2." style="max-width: 500px; width: 100%;">

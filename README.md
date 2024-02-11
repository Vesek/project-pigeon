# project-pigeon
A project for the AstroPi hackathon in Pardubice
The task is to calculate the speed of the ISS using pictures taken from a RPi HQ camera aboard.

## Mission Zero
Our twist on mission zero is a sinewave modulated by the intesity of the RGB components ilustrating a real EM wave.
Made by [Franta Slepička](https://github.com/FrteenCZ)

## Mission Space Lab
In mission space lab we used an approach that differentiates itself from the other ones because it doesn't use much of the original code from the instructions.

We plan to add a TFLite model for the Coral NPU on board the Astro Pi and also try to measure the g-forces caused by the slight acceleration of the station.

During the development we used a lot of internet resources but the most helpful were Python, OpenCV and RPi docs.

## Usage
To make a flight ready `.zip` file, remove the _iss from desired files and compress them.
Those files have some of the debugging stripped from them, so they only run in the simulator.

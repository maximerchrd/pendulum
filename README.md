# AI Pendulum

To try it out, run `train_IMU-Distance_export.py` (make sure you have installed the necessary packages).  
Adapt the relevant parameters to your physical pendulum.

You can then run `visualize.py` to test the weights calculated during training.

If you want to try it on a microcontroller and a physical pendulum, upload the `.ino` file in `rp2040_PendulumAI` (don’t forget to add the trained model in the same folder).

However, I was not able to achieve a physically balanced pendulum yet.

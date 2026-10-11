# CartPole DQN project

This repository implements and evaluates a Deep Q-Network for the CartPole-v1 task. It accompanies the Part 2 presentation. The included `results_full` folder contains the measured runs, not expected or invented values.

## Run

Requires Python 3.10 or newer. In a virtual environment, install `numpy`, `matplotlib`, and `Pillow` (the last package is only needed for GIF generation):

```bash
python -m pip install -r requirements.txt
python train.py --steps 100000 --output results_full
python train.py --steps 100000 --double --learning-rate 0.0002 --output results_double
python make_assets.py --output results_full
```

The project works without Gymnasium because `cartpole.py` implements the standard CartPole-v1 equations, Euler integration, initial-state range, failure bounds, reward, and 500-step limit. For an official Gymnasium benchmark run, install `gymnasium` and add `--gym` to the training command. Keep results from the two environments labeled separately. The GIF creator always uses the bundled physics.

## Files

- `cartpole.py`: simulator with `reset` and `step` methods.
- `dqn.py`: two-hidden-layer (64 units each) NumPy network, Huber loss, Adam, and replay storage.
- `train.py`: five independent seeded runs, random and angle-rule baselines, 20-episode evaluations, CSV logs, and final model checkpoints.
- `make_assets.py`: figures and a playback GIF produced from a final checkpoint.
- `results_full/summary.json`: exact numeric results; `evaluations.csv` has the learning curves.
- `results_double/`: a labeled follow-up sensitivity experiment using Double DQN and a lower learning rate.

The 20 evaluation episodes use initial-state seeds 10000–10019. Each training run saves its final checkpoint, with no best-checkpoint selection. Terminal failures stop bootstrapping; reaching the time limit retains bootstrap value. Baseline random action sequences are seeded independently.

## Part 1 comparison

The original plan used PyTorch and Gymnasium. This reproducible run uses NumPy and bundled CartPole physics because those packages were unavailable in the authoring environment. It retains the DQN architecture, replay, target network, epsilon schedule, 100,000 steps per seed, five training seeds, and evaluation schedule. Optimization updates occur every four steps to keep execution practical. A separately labeled follow-up tried Double DQN with a smaller learning rate after the original DQN run was unstable. Both versions fell short of the 400-point target. The threshold was set in Part 1 and is not changed after seeing the results. This is a self-contained simulation result, not a verified Gymnasium leaderboard score or physical-control claim.

## Recording the submission video

Open the slide deck, enable your webcam and microphone in ScreenPal (or another screen recorder), and keep the camera visible while you share the slides. On the demo slide, play `results_full/agent_demo.gif` or the MP4 copy. Follow the slide speaker notes and the timing guide included with the presentation. Verify that the exported recording is 13–15 minutes and that your face, audio, and slides remain visible.


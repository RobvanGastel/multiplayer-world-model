# Multiplayer Interactive World Models for Air Hockey

This repository is a small-scale reproduction of MIRA (Hu et al., 2026), a multiplayer interactive world model, on a two-player air hockey game instead of Rocket League. A world model learns to simulate a game from pixels and actions alone: given a few frames of context and the keys each player presses, it generates the next frames for both players, which makes the game playable without the game engine. MIRA's recipe has two parts. First, a codec compresses each frame into a compact latent with a frozen DINOv3 (Siméoni et al., 2025) encoder, following representation autoencoders (Zheng et al., 2025). Then a causal diffusion transformer, trained with flow matching (Lipman et al., 2023), predicts the next latent frame from the previous ones and the players' actions. 

Air hockey keeps the problem small enough to train on a single GPU: the full model has 83M parameters, of which only the 37M-parameter diffusion transformer is trained in the world-model stage.

Check out the `Explanation.ipynb` notebook for a walkthrough of the code and ideas behind it.

**Generation quality improvement over training time.** The left column is the real game; the other two are the world model halfway through training (67k steps) and fully trained (140k steps), given the same starting frames and the keys that were really pressed. Each row is one player's view, and midway we switch to the generated future state. Halfway through training, the model ends this game in a draw that never happened. Fully trained, it generates the real goal, and both players' scoreboards update together.

![](/assets/progression_no_border.gif?raw=true)

**The world model follows the keys.** After the real context frames, the blue player switches to random keys that never occurred with this game. The yellow arrow shows the direction the held keys push the mallet, and the generated mallet moves accordingly: right along the centre line, then down towards its own goal.

![](/assets/random_keys.gif?raw=true)

**The world model plays both players itself.** Here it gets no keys at all: both players' actions are replaced by the learned "unknown" token, which the model also sees on 10% of training steps (action dropout, as in MIRA). Starting from the countdown of a real game, it plays both sides. It is noticeable that the quality here is usually not as good, random moves might affect the performance. Additionally, the PPO agents are not perfect.

![](/assets/no_actions.gif?raw=true)

## Setup
Install the packages using the `requirements.txt` file.

```bash
# using conda
conda create --name wm python=3.11
conda activate wm
# Install the package for wm imports,
pip install -e .
```

The codec uses the pretrained DINOv3 ViT-S/16 weights. Download them from the [DINOv3 repository](https://github.com/facebookresearch/dinov3) and set `dino_weights_dir` in `configs/codec/rae_encoder.yml` to the folder containing them.

## Usage
The pipeline has four stages. Each stage is one command, and the output of each stage is the input of the next.

**1. Train the agents.** The game data is played by PPO (Schulman et al., 2017) agents, first initialised by behaviour cloning from a scripted player and then trained with self-play.

```bash
python -m tools.agent.train_ppo --config configs/ppo/ppo_self_play.yml --load-path runs/distilled.pt
```

**2. Collect matches.** The agents play matches against each other and every frame is rendered from both players' points of view. The recipe, which agents play, how actions are recorded and how often moves are random, is in `configs/data/collect.yml`; command-line flags override it.

```bash
python -m tools.agent.collect_matches --num-matches 1500 --seed 42 --random-frac 0.3 --out-dir data/train
```

**3. Train the codec.** The codec encodes each 160×128 frame into a 5×4×32 latent. A frozen DINOv3 encoder extracts features, a strided convolution compresses them, and a ViT decoder learns to reconstruct the frame.

```bash
python -m tools.train_codec --encoder-config configs/codec/rae_encoder.yml --decoder-config configs/codec/vit_decoder.yml --loss-config configs/codec/codec_loss.yml
```

**4. Train the world model.** Finally, train the combined model by first encoding the latents. Every collected game is encoded once, so world-model training reads small latents from memory instead of decoding video and running the codec on every step. 

```bash
python -m tools.encode_latents --root data/train
python -m tools.train_wm --config configs/world_model/train.yml
```

**5. Evaluate.** This script evaluate the performance of the model, for example the pixel error between the true rollout and the generated rollout.

```bash
python -m tools.evaluation.evaluate_wm --checkpoint runs/wm/world_model.pt --data-root data/test

```

## Results
The world model is trained for 140k steps on 14,270 games: 7,278 of pure agent play and 6,992 with 30% random moves. All results are on held-out games with random moves, which are the hardest to predict without reading the keys.

**Pixel error.** We locate the puck and each player's mallet in the generated frames by their colour and compare the positions with the real frames, after 19 real context frames. A step is one latent, which covers two frames; 30 steps is one second. The codec itself already places the puck 1.25 px off, which is the lowest error a generated frame can reach.

| | 1 step | 8 steps | 16 steps | 30 steps |
|---|---|---|---|---|
| Puck, world model | 0.45 px | 1.16 px | 1.53 px | 1.82 px |
| Puck, repeat the last frame | 1.16 px | 8.11 px | 14.92 px | 16.28 px |
| Own mallet, true keys | 0.37 px | 2.62 px | 5.01 px | 11.08 px |
| Own mallet, another game's keys | 0.58 px | 10.23 px | 15.11 px | 26.14 px |

**Action Recoverability Ratio (ARR).** MIRA measures controllability with ARR: a small probe on frozen DINOv3 features is trained to detect which keys are pressed in a short window of real video. ARR divides how well the probe recovers the commanded keys from the generated video by how well it recovers them from the codec's reconstruction of the real video. An ARR of 1 means the keys are as visible in the generated video as in the real one. Our probe reaches 0.87 mean average precision on real held-out video. Besides MIRA's version, which uses each clip's real keys, we also condition each clip on another game's keys, because the real keys can partly be predicted from the context frames alone.

| model | ARR, real keys | ARR, another game's keys |
|---|---|---|
| world model | 0.97 | 0.84 |
| chance level | 0.64 | 0.64 |

**Where it still fails.** The world model sometimes ends a game too early. In 3 of 8 held-out rollouts, it generated the "TIME'S UP - DRAW" banner while the real game played on or ended in a goal.

![](/assets/failure_time_up.gif?raw=true)


## Citing
If you reference or use the codebase in your research, please cite:

```
@misc{2026airhockey_wm,
    title={Multiplayer Interactive World Models for Air Hockey},
    author = {Van Gastel, Rob},
    year={2026}
}
```

## References
Hu, A., Volhejn, V., Rahary, A. R., Mulder, C., Makkar, A., Liao, A., Royer, A., Orsini, M., Jelley, A., Alonso, E., Laurent, F., Norén, F., Swingos, J., Hünermann, J., Rollins, K., Hosseini, L., Cauchois, M. L., Peter, M., de Witte, P., … Pérez, P. (2026). Multiplayer Interactive World Models with Representation Autoencoders (Version 2). arXiv. https://doi.org/10.48550/ARXIV.2607.05352

Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., … Bojanowski, P. (2025). DINOv3 (arXiv:2508.10104). arXiv. https://doi.org/10.48550/arXiv.2508.10104

Zheng, B., Ma, N., Tong, S., & Xie, S. (2025). Diffusion Transformers with Representation Autoencoders (arXiv:2510.11690). arXiv. https://doi.org/10.48550/arXiv.2510.11690

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling (arXiv:2210.02747). arXiv. https://doi.org/10.48550/arXiv.2210.02747

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms (arXiv:1707.06347). arXiv. https://doi.org/10.48550/arXiv.1707.06347
DONGS — Evolving AI Creatures

DONGS is a Python/Pygame simulation of tiny evolving creatures that live in a simple 2D world.
Each creature (a “Dong”) is controlled by a genetic code and a neural network brain. They eat, explore, play, breed, and even “talk” to the user through a terminal block in their world. Over time, their genes and neural networks combine and mutate, producing new generations with different traits.

🧬 Genetics

Every Dong is born with a genetic code that defines its tendencies and natural limits.
Genes affect how it explores, eats, plays, breeds, and how much intelligence it can develop.

Genes and Their Roles

Each Dong has the following genes (randomised at birth and sometimes reshuffled on breeding):

max_intel – Caps how quickly the creature can improve and how long it might live.
nn_growth – Governs how strongly exploration and interactions improve fitness and neural net adaptation.
pref_eat – Appetite bias. Higher = stronger drive towards food.
pref_play – Playfulness. Higher = more likely to pick up balls or push toys.
pref_breed – Breeding urge. Higher = stronger drive to seek mates.
explore_bias – Curiosity. Higher = more exploration and wandering.
move_scalar – General physical movement speed scaling.
social_drive – Tendency to stay near other Dongs and coordinate movement.
risk_sense – Caution factor that increases survival behaviour at low energy.
memory_span – Strength of remembered food locations when none are nearby.
chattyness – How frequently the Dong initiates terminal conversations.

These values are floats within ranges (e.g. 0.6–1.4), so no two Dongs are genetically identical.
Genetics interact with the neural net, meaning two creatures with the same brain weights may behave differently depending on their genes.

🧠 Neural Network

Each Dong has a compact multi-layer perceptron (MLP) controlling its movement decisions.

Inputs (4 total):

Relative X distance to nearest food
Relative Y distance to nearest food
Current energy level (scaled 0–1)
Number of nearby Dongs (scaled)

Hidden Layer: 8 nodes (tanh activation)

Outputs (2 total):

Desired movement in X
Desired movement in Y

Weights and Biases:

w1: shape (8×4)
b1: shape (8,)
w2: shape (2×8)
b2: shape (2,)

Total parameters per Dong: 74 trainable values

On breeding, offspring inherit a crossover of the parents’ weights with random mutations.
Weights are logged to disk so you can analyse neural net evolution over time.

🌍 Simulation World

The world is a 1280×720 grid with:

Food items that spawn randomly — eaten to restore energy.
Toys (balls and blocks) — playing boosts fitness and satisfies play preference.
Terminal block — Dongs occasionally dock with it to “speak” to the user. The user can also type messages, and the creatures reply via a connected Large Language Model (LLM).
Terrain — includes grass, trees, rocks, and lakes. Movement is slower in some regions, and creatures avoid water.
Creatures wander, play, explore, and breed. If their energy runs out or they exceed lifespan, they die.
New generations continue with mixed and mutated genes and neural nets.

🎮 Controls

Start — begins a new simulation (IDs reset from DONG0001, genes and brains randomised).
Pause/Resume — freezes the world so you can hover over Dongs to inspect them or drag them around.
Stop — ends the simulation and finalises the log file.
Dropdown — select which LLM the terminal block uses (disabled while simulation runs).
Terminal panel — type messages to chat with the creatures (only user ↔ Dong dialogue is shown).

📑 Logging

Each run generates a log file under Programs/:

Filename format: DONG-YYYYMMDD-HHMMSS.txt
Tab-delimited table with:

Timestamp

Dong ID, age, energy, fitness

All genetic values
All 74 neural net weights/biases

This allows tracking evolution across the whole run.

🔄 Development History / Improvements

During development, many improvements and fixes were made:

Graphics/UI

Fixed panel overlap (LLM panel height adjusted, INSPECTOR panel fully visible).
Terminal input line no longer hides text.
Only chat messages (user + Dongs) are displayed in terminal (system logs removed).
Pause/resume added with drag-and-inspect during pause.

Behaviour

Dongs originally hugged borders — exploration bias and edge repulsion added.
Wandering improved with waypoints and gentle randomness.
Playing with toys now boosts stats, improving evolution.
Breeding consumes 30–40% energy, preventing infinite offspring.

Simulation

Randomised genes and weights at each Start.
IDs reset each new game (DONG0001 upwards).
Per-run log files added with timestamped names.

Terminal

User messages produce a green scanline on the in-world terminal block.
Replies are colour-coded: white for user, cyan for Dongs/LLM.

Running
pip install pygame numpy
python main.py

LLM setup is preconfigured for your k3s/MetalLB Ollama endpoint `http://192.168.0.202:11434` (AI NUC node `192.168.0.20`) via `config.json` (`ollama.base_url`, `provider`, `cluster_hint`).
Quick test: `curl http://192.168.0.202:11434/api/tags`.


Requires Python 3.10+ and a display environment.


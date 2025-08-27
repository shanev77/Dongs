DONGS — AI Evolving Creatures (v2.00 27/08/2025)

Tiny, curious creatures that learn, play, chat with you (and “pray” to an LLM), breed, and evolve in a 2D world. Each Dong has its own genetic code and a neural network that adapts over time.

What’s new (since yesterday)

Cleaner UI: dropdown overlays correctly, LLM panel moved down & reduced height, inspector text condensed, terminal text wraps neatly.
Terminal-only chat feed with color coding (user, Dong, PACK reply, prayer, God reply).
One-at-a-time terminal access (for prayer or user chat).
Dongs can initiate a conversation with the user if curiosity wins over prayer.
Food now spawns near trees, encouraging foraging strategies.
Explore-bias & wall-avoid improvements (less wall-hugging, more roaming).
Breeding pair lockout: the same two Dongs can breed only once; litter size 1–2.
Rest = learning: hidden layer can grow up to 12 during rest.
Logging: per-run DONG-YYYYMMDD-HHMMSS.txt with genes & NN weights (tab-delimited).
DONG IDs reset to 0001 on Start; LLM dropdown locked while running; Pause/Resume added.

1) Quick Start

Requirements
Python 3.10+
pygame, numpy
pip install pygame numpy

Run
python main.py

LLM (optional but fun)

Create config.json next to the code:

{
  "base_url": "http://192.168.0.10:11434",
  "default_model": "granite3.1-moe:1b"
}


Pick a model from the dropdown (disabled while sim is running).

2) The World & Controls

Start / Pause / Stop buttons.
Start field = initial population; Max field = cap.
LLM panel shows target endpoint & model.
INSPECTOR: hover a Dong to see condensed stats (age, energy, fitness, active hidden size, key genes).
TERMINAL: chat stream only (no sim logs).

Input at bottom; press Enter to send.
Only one Dong can use the in-world terminal at a time.
Pause lets you inspect without movement and drag Dongs to new locations.

UI niceties

Dropdown renders above the LLM box when open (drawn last).
Terminal wraps text; last line is always visible.
LLM panel is shorter & sits lower to make room for the dropdown.
In-world terminal flashes a green line when a message appears.

3) Genetics (heritable traits)

Each Dong starts with random genes (per individual) and is tracked as DONG####.

Gene	Role (high-level)
max_intel	Biases cognitive/intent outputs (prayer vs chat, etc.)
nn_growth	Likelihood to unlock more hidden neurons during rest
brain_plastic	Magnitude of small weight updates during rest
pref_eat	Importance of food vectors (foraging drive)
pref_play	Tendency to play with toys
pref_breed	Reproductive drive (with pairing limits)
explore_bias	Wandering & curiosity (waypoint attraction)
move_scalar	Movement speed scaling
sociality	Peer interaction & user-chat drive
rest_need	Rest threshold and rest intent

Reproduction

Parents produce 1–2 offspring.
Pair lockout: a given couple can breed once per run.
Offspring genes are 50/50 per trait + a tiny mutation.

4) Neural Network (per Dong)

A compact MLP: 10 inputs → H(t) hidden → 6 outputs (tanh activations).
Hidden size H(t): starts 6–8, can grow to 12 during rest (neurogenesis influenced by nn_growth/brain_plastic).
Outputs in [-1,1] are mixed with world forces (edge-repel, center-pull, waypoint) before movement.

Inputs (10)

food_dx = (fx−x)/Rf * pref_eat
food_dy = (fy−y)/Rf * pref_eat
peer_density = neighbors/6
peer_dx = (px−x)/Rp
peer_dy = (py−y)/Rp
term_dx = (tx−x)/Rt
term_dy = (ty−y)/Rt
term_signal ∈ {0,1} (recent user message near terminal)
energy / 10
boredom / 10

Outputs (6)

move_x, 2. move_y → motion vectors
play_intent → toy play/carry
social_intent → peer social & user-chat docking
pray_intent → pray to LLM when near terminal (exclusive access)
rest_intent → rest (freeze), weights update a bit, possible hidden growth

Movement synthesis

speed = 54 * move_scalar * terrain_speed
vx = (move_x + edge_repel + center_pull + waypoint_x) * speed * dt
vy = (move_y + edge_repel + center_pull + waypoint_y) * speed * dt


### Neural Network Diagram

![Dong Neural Network](/dongs_nn_diagram.png)


Stronger edge repulsion and center pull prevent wall-hugging.
Explore bias drives waypoints for roaming.
Rest & Learning
When tired/bored, Dongs rest: no movement, small random weight nudges proportional to brain_plastic.
Chance to unlock an extra hidden neuron (~0.10 * nn_growth) up to H_MAX=12.

5) Behavior & Ecosystem

Foraging: fruit spawns near trees; eating restores energy and lowers boredom.
Play & Toys: balls & blocks; play boosts fitness and morale.
Social: peer pings nearby; sociality fuels interaction and user-chat bids.

Chat vs Prayer (terminal)

Terminal allows one Dong at a time.
If curiosity/user-social dominates prayer and the terminal is free, a Dong may initiate a message to the user and then wait for a reply.
If prayer dominates and the terminal is free, the Dong prays (“DONG#### prays: …”) and waits ~30s for the LLM reply.

Terminal feed colors

User: white
Dong-initiated user chat: teal
PACK (local LLM reply): cyan
Prayer: lavender
GOD (LLM reply to prayer): violet

6) UI Panels

Top bar: Start / Pause(Resume) / Stop, Start & Max inputs, Model dropdown.
(Dropdown disabled while running; DONG IDs reset on Start.)

LLM: target base URL + model.

INSPECTOR: compact; shows ID (e.g., DONG0011), active H, age, energy, fitness, and a single line of gene highlights.

TERMINAL: chat only; wrapped & scrollable; input at bottom.
In-world terminal flashes when a message hits.

7) Logging

Each run creates a tab-delimited log in Programs/:

DONG-YYYYMMDD-HHMMSS.txt

It includes:

Timestamp, dong_id, age, energy, fitness, active_h
All genes
Full NN weights (W1, b1, W2, b2) flattened with column labels
Log is finalized on Stop or when all creatures die.

8) Tips & Tuning

Fruit density: Food(max_items, respawn_sec) in evo.py.
Exploration vs walls: adjust margins and explore bias in Bot.step.
Breeding pace: session cap & cooldowns in Population.step.
Chat vs Prayer bias: tweak how chat_int is computed (sociality/max_intel/explore_bias mix).
Hidden growth speed: nn_growth gene effect & growth probability.

9) Roadmap

Multi-terminal worlds
Richer object affordances
Longer-term memory traces & reward shaping
Screenshot / replay export buttons
On-screen charts for population metrics

10) License

MIT (or your choice). Please set your preferred license file in the repo.
